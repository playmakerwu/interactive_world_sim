"""Encode a chosen goal image into a latent vector using the frozen encoder.

Usage (from repo root):
    conda run -n iws python tests/goal_selection/encode_goal.py \
        --image_path tests/goal_selection/frames/episode_3_final.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (
    LatentWorldModel,
)
from interactive_world_sim.utils.normalizer import LinearNormalizer

CKPT_PATH = "outputs/pusht_cam1/checkpoints/best.ckpt"
OBS_KEY = "camera_1_color"
RESOLUTION = 128
DEVICE = "cuda:0"
OUTPUT_PATH = Path("tests/goal_selection/z_goal.pt")


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


def main():
    parser = argparse.ArgumentParser(description="Encode a goal image to latent.")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to goal PNG"
    )
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # ── load model ───────────────────────────────────────────────────
    print("Loading model …")
    model = load_model(CKPT_PATH)
    normalizer: LinearNormalizer = model.normalizer
    dtype = model.dtype

    # ── load and preprocess image ────────────────────────────────────
    # The saved PNGs are already 128x128, stored as BGR by cv2.imwrite.
    # Replicate the exact pipeline the encoder sees during inference:
    #   raw uint8 RGB → /255 float32 → permute → normalizer → device
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"cv2.imread returned None for {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8

    # the PNG is already 128x128, but guard against other sizes
    if img_rgb.shape[:2] != (RESOLUTION, RESOLUTION):
        img_rgb = cv2.resize(
            img_rgb, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA
        )

    img_float = img_rgb.astype(np.float32) / 255.0  # [0, 1]
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
    # (1, 3, 128, 128) float32 in [0, 1]

    img_tensor = normalizer[OBS_KEY].normalize(img_tensor).to(DEVICE)
    # after normalizer: [-1, 1], on device

    # ── encode ───────────────────────────────────────────────────────
    with torch.no_grad():
        z_goal = model.encoder_forward(img_tensor)  # (1, C, H_lat, W_lat)

    z_goal_cpu = z_goal.cpu()
    l2_norm = z_goal_cpu.float().flatten().norm().item()

    # ── save ─────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z_goal_cpu, OUTPUT_PATH)

    print(f"Image:        {img_path}")
    print(f"Latent shape: {tuple(z_goal_cpu.shape)}")
    print(f"Latent dtype: {z_goal_cpu.dtype}")
    print(f"L2 norm:      {l2_norm:.4f}")
    print(f"Saved to:     {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
