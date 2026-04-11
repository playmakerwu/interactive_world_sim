"""Test the stochasticity of the learned dynamics model.

Given the SAME initial latent state and the SAME action sequence, run the
dynamics model 5 times and measure how much the predicted latent trajectories
diverge.  This isolates the randomness introduced by the consistency-model
noise initialization inside `dynamics_forward`.

Usage (from repo root):
    conda run -n iws python tests/latent_stochasticity/test_latent_variance.py
"""

from itertools import combinations
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from interactive_world_sim.utils.normalizer import LinearNormalizer

# ── constants ────────────────────────────────────────────────────────
CKPT_PATH = "outputs/pusht_cam1/checkpoints/best.ckpt"
EPISODE_PATH = "data/mini/pusht/val/episode_0.hdf5"
RESULTS_DIR = Path("tests/latent_stochasticity/results")
OBS_KEY = "camera_1_color"
RESOLUTION = 128
HIST_CONTEXT = 10
DEVICE = "cuda:0"
N_RUNS = 5


# ── model loading (from replay_trajectory.py) ───────────────────────
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


def load_model(ckpt_path: str):
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


def preprocess_img(raw_img: np.ndarray) -> np.ndarray:
    img = center_crop(raw_img, (RESOLUTION, RESOLUTION))
    img = cv2.resize(img, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


# ── autoregressive rollout (latent-only, no decoding) ────────────────
@torch.no_grad()
def rollout_latent(
    model: LatentWorldModel,
    init_latent: torch.Tensor,   # (1, 1, C, H, W)
    actions_norm: torch.Tensor,  # (T, A)  — already normalized, on device
) -> np.ndarray:
    """Run one autoregressive rollout.  Returns latents as np array (T, C*H*W)."""
    dtype = model.dtype
    T = actions_norm.shape[0]

    curr_latent = init_latent.clone()               # (1, 1, C, H, W)
    action_hist: list[torch.Tensor] = []

    latents_flat: list[np.ndarray] = []
    # record initial latent
    latents_flat.append(
        curr_latent[0, -1].flatten().cpu().float().numpy()
    )

    for t_idx in range(T):
        action_chunk = actions_norm[t_idx].unsqueeze(0)  # (1, A)
        action_hist.append(action_chunk)

        action_seq = torch.cat(action_hist, dim=0)[-(HIST_CONTEXT + 1) :]
        action_seq = rearrange(action_seq, "t a -> 1 t a")

        latent_pred = model.dynamics_forward(curr_latent, action_seq)
        curr_latent = torch.cat([curr_latent, latent_pred], dim=1)
        curr_latent = curr_latent[:, -HIST_CONTEXT:]

        latents_flat.append(
            curr_latent[0, -1].flatten().cpu().float().numpy()
        )

    return np.stack(latents_flat, axis=0)  # (T+1, D)


# ── analysis helpers ─────────────────────────────────────────────────
def pairwise_l2(latents_all: np.ndarray) -> np.ndarray:
    """latents_all: (N_RUNS, T+1, D).  Returns (T+1,) mean pairwise L2."""
    T1 = latents_all.shape[1]
    mean_pw = np.zeros(T1)
    max_pw = np.zeros(T1)
    for t in range(T1):
        dists = []
        for i, j in combinations(range(N_RUNS), 2):
            d = np.linalg.norm(latents_all[i, t] - latents_all[j, t])
            dists.append(d)
        mean_pw[t] = np.mean(dists)
        max_pw[t] = np.max(dists)
    return mean_pw, max_pw


def print_summary(mean_pw: np.ndarray, max_pw: np.ndarray, norms: np.ndarray):
    T1 = mean_pw.shape[0]
    print("\n" + "=" * 72)
    print("LATENT STOCHASTICITY SUMMARY")
    print("=" * 72)
    print(f"{'t':>5}  {'mean_pw_L2':>12}  {'max_pw_L2':>12}  "
          f"{'norm_mean':>12}  {'norm_std':>12}")
    print("-" * 72)
    step = max(1, T1 // 20)  # ~20 rows
    for t in range(0, T1, step):
        nm = norms[:, t]  # (N_RUNS,)
        print(f"{t:5d}  {mean_pw[t]:12.6f}  {max_pw[t]:12.6f}  "
              f"{nm.mean():12.4f}  {nm.std():12.6f}")
    # always print last
    t = T1 - 1
    nm = norms[:, t]
    print(f"{t:5d}  {mean_pw[t]:12.6f}  {max_pw[t]:12.6f}  "
          f"{nm.mean():12.4f}  {nm.std():12.6f}")
    print("=" * 72)
    print(f"Final mean pairwise L2: {mean_pw[-1]:.6f}")
    print(f"Final max  pairwise L2: {max_pw[-1]:.6f}")
    print(f"Max mean pairwise L2 across all t: {mean_pw.max():.6f} (at t={mean_pw.argmax()})")
    print()


# ── main ─────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load model ───────────────────────────────────────────────────
    print("Loading model …")
    model = load_model(CKPT_PATH)
    normalizer: LinearNormalizer = model.normalizer
    dtype = model.dtype

    # ── load episode ─────────────────────────────────────────────────
    print("Loading episode …")
    epi_data, _ = load_dict_from_hdf5(EPISODE_PATH)
    gt_images = epi_data["obs"]["images"][OBS_KEY][()]
    actions_raw = epi_data["action"][()]  # (T, 4)
    T_total = actions_raw.shape[0]
    print(f"Episode length: {T_total} frames")

    # ── encode initial frame ─────────────────────────────────────────
    img0 = preprocess_img(gt_images[0])
    img_tensor = (
        torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0)
    )  # (1, 3, 128, 128)
    img_tensor = normalizer[OBS_KEY].normalize(img_tensor).to(DEVICE)

    with torch.no_grad():
        init_latent_a = model.encoder_forward(img_tensor)[:, None]
        init_latent_b = model.encoder_forward(img_tensor)[:, None]
    enc_diff = (init_latent_a - init_latent_b).abs().max().item()
    print(f"Encoder determinism check: max |z_a - z_b| = {enc_diff:.2e}  "
          f"({'PASS' if enc_diff == 0.0 else 'WARN'})")

    init_latent = init_latent_a.to(dtype)

    # ── normalize actions ────────────────────────────────────────────
    actions_norm = normalizer["action"].normalize(
        torch.from_numpy(actions_raw)
    ).to(DEVICE).to(dtype)  # (T, A)

    # ── run N_RUNS rollouts ──────────────────────────────────────────
    all_latents = []
    for run_i in range(N_RUNS):
        print(f"Rollout {run_i + 1}/{N_RUNS} …")
        lats = rollout_latent(model, init_latent, actions_norm)
        all_latents.append(lats)

    all_latents = np.stack(all_latents, axis=0)  # (N_RUNS, T+1, D)
    T1 = all_latents.shape[1]

    # ── analysis ─────────────────────────────────────────────────────
    norms = np.linalg.norm(all_latents, axis=2)  # (N_RUNS, T+1)
    mean_pw, max_pw = pairwise_l2(all_latents)

    print_summary(mean_pw, max_pw, norms)

    # ── save raw data ────────────────────────────────────────────────
    npz_path = RESULTS_DIR / "latent_stochasticity.npz"
    np.savez(
        npz_path,
        all_latents=all_latents,
        mean_pairwise_l2=mean_pw,
        max_pairwise_l2=max_pw,
        norms=norms,
    )
    print(f"Raw data saved to {npz_path}")

    # ── plot 1: divergence over time ─────────────────────────────────
    timesteps = np.arange(T1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, mean_pw, label="Mean pairwise L2", color="tab:blue")
    ax.fill_between(timesteps, 0, max_pw, alpha=0.2, color="tab:blue",
                    label="Max pairwise L2")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Pairwise L2 distance")
    ax.set_title("Latent divergence across 5 rollouts (same init, same actions)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    div_path = RESULTS_DIR / "latent_divergence.png"
    fig.savefig(div_path, dpi=150)
    plt.close(fig)
    print(f"Divergence plot saved to {div_path}")

    # ── plot 2: latent norms over time ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_i in range(N_RUNS):
        ax.plot(timesteps, norms[run_i], label=f"Run {run_i}", alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("L2 norm of latent")
    ax.set_title("Latent L2 norms over time (5 runs overlaid)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    norm_path = RESULTS_DIR / "latent_norms.png"
    fig.savefig(norm_path, dpi=150)
    plt.close(fig)
    print(f"Norms plot saved to {norm_path}")


if __name__ == "__main__":
    main()
