"""Test whether gradients flow through the dynamics model for RL training.

The existing `dynamics_forward` is wrapped with @torch.no_grad() and uses
in-place tensor assignments — both break the computation graph.  This script
implements a differentiable single-step dynamics wrapper that replicates the
same math (noise init → consistency-model denoise → L2 normalize) without
those graph-breaking ops, then checks whether an actor network receives
gradients through it.

Usage (from repo root):
    conda run -n iws python tests/gradient_flow/test_gradient_flow.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from omegaconf import OmegaConf
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from interactive_world_sim.utils.normalizer import LinearNormalizer

# ── paths & constants ────────────────────────────────────────────────
CKPT_PATH = "outputs/pusht_cam1/checkpoints/best.ckpt"
EPISODE_PATH = "data/mini/pusht/val/episode_0.hdf5"
GOAL_PATH = "tests/goal_selection/z_goal.pt"
RESULTS_DIR = Path("tests/gradient_flow/results")
OBS_KEY = "camera_1_color"
RESOLUTION = 128
DEVICE = "cuda:0"
ACTION_DIM = 4
LATENT_SHAPE = (4, 32, 32)  # C, H_lat, W_lat
LATENT_DIM = 4 * 32 * 32    # 4096

HORIZONS = [1, 5, 10, 15, 20]
USE_GRAD_CHECKPOINT = True  # trade compute for memory on longer horizons


# ── model loading ────────────────────────────────────────────────────
def _patch_attention_backends():
    from torch.nn.attention import SDPBackend
    from interactive_world_sim.algorithms.models.attention import Attention

    cap = torch.cuda.get_device_capability()
    if cap[0] >= 8 and cap[0] != 8:
        _orig_init = Attention.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.cuda_backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

        Attention.__init__ = _patched_init


def load_model(ckpt_path: str) -> LatentWorldModel:
    _patch_attention_backends()
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "torch", lambda x: getattr(torch, x), replace=True
    )
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    cfg.n_frames = 10
    cfg.algorithm.n_frames = 10
    if "diffusion" in cfg.algorithm and "sampling_timesteps" in cfg.algorithm.diffusion:
        cfg.algorithm.diffusion.sampling_timesteps = 10
    if (
        "diffusion" in cfg.algorithm.dynamics
        and "sampling_timesteps" in cfg.algorithm.dynamics.diffusion
    ):
        cfg.algorithm.dynamics.diffusion.sampling_timesteps = 10
    cfg.algorithm.load_ae = None

    algo = LatentWorldModel.load_from_checkpoint(
        ckpt_path,
        cfg=cfg.algorithm,
        map_location=DEVICE,
        dtype=dtype,
        strict=False,
        weights_only=False,
    )
    algo.dynamics = algo.dynamics.to(dtype)
    algo.eval()
    algo.dynamics.eval()
    return algo


# ── differentiable single-step dynamics ──────────────────────────────
def dynamics_step_differentiable(
    model: LatentWorldModel,
    z_hist: torch.Tensor,        # (1, T_hist, C, H, W)
    action_hist: torch.Tensor,   # (1, T_hist+1, A)  — includes new action
) -> torch.Tensor:
    """One-step dynamics prediction that preserves the computation graph.

    Replicates the math of dynamics_forward for a single prediction step:
      1. Sample noise chunk (no grad needed on noise itself)
      2. Concatenate with history
      3. Build timestep schedule
      4. Run consistency model denoising via _forward
      5. L2-normalize per view
      6. Return the single predicted latent

    No @torch.no_grad, no in-place assignment on graph tensors.
    """
    # rearrange to (T, B, C, H, W) for internal dynamics convention
    z_seq = rearrange(z_hist, "b t c h w -> t b c h w")
    action_seq = rearrange(action_hist, "b t a -> t b a")

    T_hist = z_seq.shape[0]
    batch_size = z_seq.shape[1]

    # 1. Sample random noise for the new frame (not part of actor graph)
    noise = torch.randn(
        (1, batch_size, *z_seq.shape[2:]),
        device=z_seq.device, dtype=z_seq.dtype,
    )
    noise = torch.clamp(noise, -model.clip_noise, model.clip_noise)

    # 2. Concatenate history + noise (this IS differentiable w.r.t. z_seq
    #    because torch.cat preserves gradients for its inputs)
    xs = torch.cat([z_seq, noise], dim=0)  # (T_hist+1, B, C, H, W)

    # 3. Sliding window
    n_tokens = model.n_tokens
    curr_end = T_hist + 1
    curr_start = max(0, curr_end - n_tokens)
    xs_window = xs[curr_start:]
    action_window = action_seq[curr_start:curr_end]

    # 4. Timestep schedule
    window_len = xs_window.shape[0]
    clean_t = (
        torch.ones((window_len - 1,), device=DEVICE)
        * model.noise_scheduler.stabilization_level
    )
    timesteps = torch.linspace(
        model.noise_scheduler.timesteps - 1, 0,
        model.dyn_infer_steps + 1, device=DEVICE,
    )

    # 5. Denoising loop (dyn_infer_steps=1 for PushT)
    xs_updated = xs_window
    for step_i in range(model.dyn_infer_steps):
        t = timesteps[step_i].unsqueeze(0)
        s = timesteps[step_i + 1].unsqueeze(0)
        t = torch.cat([clean_t, t], 0)
        t = t[:, None].expand(-1, batch_size).long()
        s = torch.cat([clean_t, s], 0)
        s = s[:, None].expand(-1, batch_size).long()

        # _forward calls the dynamics model and noise_scheduler.CTM_calc_out
        # Both are differentiable: CTM_calc_out does linear interpolation
        # between input and model prediction.
        def _denoise_and_mix(xs_in, t_in, s_in, act_window):
            denoise_fn = lambda x, tt, ss: model.dynamics(
                x, tt, ss, external_cond=act_window
            )
            return model.noise_scheduler.CTM_calc_out(
                denoise_fn, xs_in, t_in, s_in
            )

        if USE_GRAD_CHECKPOINT:
            xs_updated = checkpoint(
                _denoise_and_mix, xs_updated, t, s, action_window,
                use_reentrant=False,
            )
        else:
            xs_updated = _denoise_and_mix(xs_updated, t, s, action_window)

    # 6. Extract the predicted frame (last in sequence)
    z_pred = xs_updated[-1:]  # (1, B, C, H, W)

    # 7. L2-normalize per view
    num_views = len(model.obs_keys)
    c_per_v = z_pred.shape[2] // num_views
    z_norm_parts = []
    for i in range(num_views):
        chunk = z_pred[:, :, i * c_per_v : (i + 1) * c_per_v]
        chunk = chunk / (torch.norm(chunk, dim=2, keepdim=True) + 1e-8)
        z_norm_parts.append(chunk)
    z_pred = torch.cat(z_norm_parts, dim=2)

    # rearrange back to (B, 1, C, H, W)
    z_pred = rearrange(z_pred, "t b c h w -> b t c h w")
    return z_pred  # (1, 1, C, H, W)


# ── actor network ───────────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),  # actions in [-1, 1]
        )

    def forward(self, z_flat: torch.Tensor) -> torch.Tensor:
        return self.net(z_flat)


# ── main test ────────────────────────────────────────────────────────
def run_gradient_test(
    model: LatentWorldModel,
    actor: Actor,
    init_latent: torch.Tensor,   # (1, 1, C, H, W) detached
    z_goal: torch.Tensor,        # (1, C, H, W)
    H: int,
) -> dict:
    """Run imagination rollout of H steps, compute loss, check grads."""
    actor.zero_grad()

    z = init_latent.detach()  # (1, 1, C, H, W) — no encoder grad
    action_hist_list = []

    # build up action history with a dummy initial action (zeros)
    dummy_action = torch.zeros(1, 1, ACTION_DIM, device=DEVICE, dtype=torch.float32)
    action_hist_list.append(dummy_action)

    for t in range(H):
        # actor produces action from current (latest) latent
        z_curr = z[:, -1]  # (1, C, H, W)
        z_flat = z_curr.reshape(1, -1)  # (1, LATENT_DIM)
        a = actor(z_flat)  # (1, ACTION_DIM)

        # append new action to history
        action_hist_list.append(a.unsqueeze(1))  # (1, 1, A)

        # build action tensor for dynamics: need T_hist + 1 actions
        T_hist = z.shape[1]
        # action_hist is all actions so far; take last T_hist+1
        all_actions = torch.cat(action_hist_list, dim=1)  # (1, t+2, A)
        action_input = all_actions[:, -(T_hist + 1):]     # (1, T_hist+1, A)

        # differentiable dynamics step
        z_next = dynamics_step_differentiable(model, z, action_input)
        # (1, 1, C, H, W)

        # update latent history (keep up to 10 for sliding window)
        z = torch.cat([z, z_next], dim=1)
        z = z[:, -10:]  # hist_context

    # loss: negative cosine similarity to goal
    z_final = z[:, -1].reshape(1, -1)  # (1, D)
    z_goal_flat = z_goal.reshape(1, -1)  # (1, D)
    cos_sim = F.cosine_similarity(z_final, z_goal_flat, dim=1)
    loss = -cos_sim.mean()

    # backward
    loss.backward()

    # collect gradient info
    grad_norms = {}
    all_finite = True
    any_nonzero = False
    for name, p in actor.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            grad_norms[name] = gn
            if not np.isfinite(gn):
                all_finite = False
            if gn > 0:
                any_nonzero = True
        else:
            grad_norms[name] = None
            all_finite = False

    total_grad_norm = sum(
        p.grad.norm().item() ** 2
        for p in actor.parameters()
        if p.grad is not None
    ) ** 0.5

    result = {
        "H": H,
        "loss": loss.item(),
        "loss_finite": bool(np.isfinite(loss.item())),
        "total_grad_norm": total_grad_norm,
        "grads_finite": all_finite,
        "grads_nonzero": any_nonzero,
        "gradients_ok": all_finite and any_nonzero,
        "per_layer": grad_norms,
    }
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load model ───────────────────────────────────────────────────
    print("Loading model …")
    model = load_model(CKPT_PATH)
    normalizer: LinearNormalizer = model.normalizer

    # freeze all dynamics/encoder/decoder params
    for p in model.parameters():
        p.requires_grad_(False)

    # ── load goal ────────────────────────────────────────────────────
    z_goal = torch.load(GOAL_PATH, map_location=DEVICE, weights_only=True)
    z_goal = z_goal.float()
    print(f"Goal latent loaded: {tuple(z_goal.shape)}, norm={z_goal.flatten().norm():.2f}")

    # ── encode initial frame ─────────────────────────────────────────
    print("Encoding initial frame …")
    epi_data, _ = load_dict_from_hdf5(EPISODE_PATH)
    raw_img = epi_data["obs"]["images"][OBS_KEY][0]
    img = center_crop(raw_img, (RESOLUTION, RESOLUTION))
    img = cv2.resize(img, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img_tensor = normalizer[OBS_KEY].normalize(img_tensor).to(DEVICE)

    with torch.no_grad():
        init_latent = model.encoder_forward(img_tensor)[:, None]  # (1,1,C,H,W)
    init_latent = init_latent.float()
    print(f"Initial latent: {tuple(init_latent.shape)}, norm={init_latent.flatten().norm():.2f}")

    # ── first: diagnose the original dynamics_forward ────────────────
    print("\n" + "=" * 72)
    print("DIAGNOSIS: why dynamics_forward blocks gradients")
    print("=" * 72)
    print("1. @torch.no_grad() decorator on dynamics_forward  → kills all grad tracking")
    print("2. In-place assignment: xs_pred[curr_start:] = xs_pred_updated")
    print("   → overwrites a slice of a tensor that could be in the graph")
    print("3. .clone() on xs_pred_chunk in normalization → safe, but moot under no_grad")
    print()
    print("Solution: differentiable single-step wrapper that avoids all three issues.")
    print("=" * 72)

    # ── run gradient tests at multiple horizons ──────────────────────
    print("\nRunning gradient flow tests …\n")
    results = []

    for H in HORIZONS:
        # fresh actor each time so grads don't accumulate across H tests
        torch.manual_seed(42)
        actor = Actor(LATENT_DIM, ACTION_DIM).to(DEVICE).float()
        actor.train()

        r = run_gradient_test(model, actor, init_latent, z_goal, H)
        results.append(r)

        status = "OK" if r["gradients_ok"] else "FAIL"
        print(f"  H={H:3d}  loss={r['loss']:+.6f}  "
              f"grad_norm={r['total_grad_norm']:.6e}  "
              f"finite={r['grads_finite']}  nonzero={r['grads_nonzero']}  [{status}]")

    # ── summary table ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("GRADIENT FLOW SUMMARY")
    print("=" * 72)
    print(f"{'H':>4} | {'loss':>12} | {'grad_norm':>14} | {'gradients_ok':>13}")
    print("-" * 72)
    for r in results:
        print(f"{r['H']:4d} | {r['loss']:+12.6f} | {r['total_grad_norm']:14.6e} | "
              f"{'True' if r['gradients_ok'] else 'FALSE':>13}")
    print("=" * 72)

    # per-layer details for the longest horizon
    last = results[-1]
    print(f"\nPer-layer gradient norms (H={last['H']}):")
    for name, gn in last["per_layer"].items():
        print(f"  {name:30s}  {gn:.6e}" if gn is not None else f"  {name:30s}  None")

    # ── save results ─────────────────────────────────────────────────
    json_path = RESULTS_DIR / "gradient_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
