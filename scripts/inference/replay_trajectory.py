"""Replay a real trajectory through the PushT world model and produce a
side-by-side comparison video (ground truth vs. world-model prediction).

Usage (from repo root):
    conda run -n iws python scripts/inference/replay_trajectory.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

# ── project imports ──────────────────────────────────────────────────
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from interactive_world_sim.utils.normalizer import LinearNormalizer


# ── helpers (copied from teleoperate_keyboard.py) ────────────────────
def _patch_attention_backends():
    """Patch Attention class to use MATH backend on GPUs that falsely match
    the A100 detection heuristic (major>=8, minor==0) but don't support
    flash attention with float32 (e.g. RTX 5070 Ti, compute 12.0)."""
    from torch.nn.attention import SDPBackend
    from interactive_world_sim.algorithms.models.attention import Attention

    cap = torch.cuda.get_device_capability()
    if cap[0] >= 8 and cap[0] != 8:
        # Not actually A100/H100 — override to MATH
        Attention.__init__orig = Attention.__init__
        _orig_init = Attention.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.cuda_backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

        Attention.__init__ = _patched_init


def _register_resolvers():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)


def load_model(ckpt_path: str):
    _patch_attention_backends()
    _register_resolvers()
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
        map_location="cuda:0",
        dtype=dtype,
        strict=False,
        weights_only=False,
    )
    algo.dynamics = algo.dynamics.to(dtype)
    algo.eval()
    algo.dynamics.eval()
    return algo


def preprocess_img(raw_img: np.ndarray, resolution: int = 128) -> np.ndarray:
    """Crop and resize to (resolution, resolution), return float32 in [0,1]."""
    img = center_crop(raw_img, (resolution, resolution))
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def main():
    # ── paths ────────────────────────────────────────────────────────
    ckpt_path = "outputs/pusht_cam1/checkpoints/best.ckpt"
    episode_path = "data/mini/pusht/val/episode_0.hdf5"
    output_dir = Path("data/wm_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "replay_comparison.mp4"

    obs_key = "camera_1_color"
    resolution = 128
    hist_context = 10
    device = "cuda:0"

    # ── load model ───────────────────────────────────────────────────
    print("Loading model …")
    model = load_model(ckpt_path)
    normalizer: LinearNormalizer = model.normalizer
    dtype = model.dtype

    # ── load episode ─────────────────────────────────────────────────
    print("Loading episode …")
    epi_data, _ = load_dict_from_hdf5(episode_path)
    gt_images = epi_data["obs"]["images"][obs_key]  # (T, H, W, 3) uint8
    actions_raw = epi_data["action"]  # (T, 4) float32
    T_total = gt_images.shape[0]
    print(f"Episode length: {T_total} frames")

    # ── initial frame → latent ───────────────────────────────────────
    t0 = 0
    img0 = preprocess_img(gt_images[t0], resolution)  # (128,128,3) float32 [0,1]
    img_tensor = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0)  # (1,3,128,128)
    img_tensor = normalizer[obs_key].normalize(img_tensor).to(device)

    with torch.no_grad():
        curr_latent = model.encoder_forward(img_tensor)[:, None]  # (1,1,C,H,W)
    curr_latent = curr_latent.to(dtype)

    # initial action (normalized)
    curr_action_raw = actions_raw[t0]
    curr_action = normalizer["action"].normalize(
        torch.from_numpy(curr_action_raw)
    ).to(device).to(dtype)

    # ── autoregressive rollout ───────────────────────────────────────
    print("Running autoregressive rollout …")
    gt_frames = []
    pred_frames = []

    # render initial predicted frame
    with torch.no_grad():
        xs_pred = render_img_cm(
            model, curr_latent[:, -1], resolution,
            normalizer=normalizer, num_views=1,
        )
    pred_np = (xs_pred[0].permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
    gt_np = (img0 * 255).clip(0, 255).astype(np.uint8)
    gt_frames.append(gt_np)
    pred_frames.append(pred_np)

    action_hist = []
    skip_frame = 1

    for t_idx in range(1, T_total):
        # ground truth frame
        gt_img = preprocess_img(gt_images[t_idx], resolution)
        gt_frames.append((gt_img * 255).clip(0, 255).astype(np.uint8))

        # get next action from dataset (normalized)
        next_action_raw = actions_raw[t_idx]
        next_action = normalizer["action"].normalize(
            torch.from_numpy(next_action_raw)
        ).to(device).to(dtype)

        curr_action = next_action
        action_chunk = curr_action.unsqueeze(0)  # (1, A)
        action_hist.append(action_chunk)

        # stack action history for dynamics
        action_seq = torch.cat(action_hist, dim=0)[-(hist_context + 1):]  # (<=hist+1, A)
        action_seq = rearrange(action_seq, "t a -> 1 t a")

        # dynamics forward
        with torch.no_grad():
            latent_pred = model.dynamics_forward(curr_latent, action_seq)
        curr_latent = torch.cat([curr_latent, latent_pred], dim=1)
        curr_latent = curr_latent[:, -hist_context:]

        # render predicted frame
        with torch.no_grad():
            xs_pred = render_img_cm(
                model, curr_latent[:, -1], resolution,
                normalizer=normalizer, num_views=1,
            )
        pred_np = (xs_pred[0].permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
        pred_frames.append(pred_np)

        if t_idx % 20 == 0:
            print(f"  frame {t_idx}/{T_total}")

    # ── write side-by-side video ─────────────────────────────────────
    print(f"Writing video to {output_path} …")
    h, w = resolution, resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 10, (w * 2, h))

    for gt_f, pred_f in zip(gt_frames, pred_frames):
        # gt and pred are RGB, cv2 expects BGR
        gt_bgr = cv2.cvtColor(gt_f, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.cvtColor(pred_f, cv2.COLOR_RGB2BGR)
        side_by_side = np.concatenate([gt_bgr, pred_bgr], axis=1)
        writer.write(side_by_side)

    writer.release()
    print(f"Done! Video saved to {output_path} ({len(gt_frames)} frames)")


if __name__ == "__main__":
    main()
