#!/usr/bin/env python3
"""Pre-encode RGB trajectories into latent tensors using a frozen encoder.

Usage:
    # Encode mini pusht dataset (for testing)
    python scripts/pre_encode_latents.py \
        --ckpt_path outputs/pusht_cam1/checkpoints/best.ckpt \
        --dataset_dir data/mini/pusht \
        --output_dir data/mini/pusht_latent \
        --obs_keys camera_1_color \
        --batch_size 64

    # Encode full dataset
    python scripts/pre_encode_latents.py \
        --ckpt_path outputs/pusht_cam1/checkpoints/best.ckpt \
        --dataset_dir data/full/pusht \
        --output_dir data/full/pusht_latent \
        --obs_keys camera_1_color \
        --batch_size 128
"""

import argparse
import glob
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


def _register_resolvers():
    """Register custom OmegaConf resolvers used by this project's configs."""
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
    if not OmegaConf.has_resolver("torch"):
        import torch as _torch
        OmegaConf.register_new_resolver("torch", lambda x: getattr(_torch, x))


def load_frozen_encoder(ckpt_path: str, device: str = "cuda"):
    """Load encoder from checkpoint, freeze it, return (encoder_forward_fn, cfg)."""
    from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel

    _register_resolvers()

    ckpt_dir = os.path.dirname(os.path.dirname(ckpt_path))
    cfg_path = os.path.join(ckpt_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    # Temporarily disable AE loading to avoid recursion
    cfg.algorithm.load_ae = None
    model = LatentWorldModel.load_from_checkpoint(
        ckpt_path,
        cfg=cfg.algorithm,
        map_location=device,
        weights_only=False,
    )
    model.eval()

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    obs_keys = list(cfg.algorithm.obs_keys)
    num_views = len(obs_keys)

    @torch.no_grad()
    def encode_batch(images: torch.Tensor) -> torch.Tensor:
        """Encode (B, C, H, W) float32 [0,1] images to latent."""
        return model.encoder_forward(images.to(device))

    return encode_batch, cfg, obs_keys, num_views


def process_split(
    encode_fn,
    split_dir: str,
    output_split_dir: str,
    obs_keys: list,
    batch_size: int,
    resolution: int,
):
    """Process all episodes in a split directory."""
    episode_paths = sorted(
        glob.glob(os.path.join(split_dir, "episode_*.hdf5")),
        key=lambda p: int(Path(p).stem.split("_")[-1]),
    )
    if not episode_paths:
        print(f"  No episodes found in {split_dir}, skipping.")
        return

    os.makedirs(output_split_dir, exist_ok=True)
    episode_ends = []
    cumulative_len = 0

    for ep_path in tqdm(episode_paths, desc=f"  Encoding {os.path.basename(split_dir)}"):
        ep_name = Path(ep_path).stem
        with h5py.File(ep_path, "r") as f:
            # Load and concatenate camera images → (T, C_total, H, W)
            obs_list = []
            for key in obs_keys:
                # HDF5 stores images under obs/images/<key>
                imgs = f["obs"]["images"][key][:]  # (T, H_orig, W_orig, C) uint8
                T = imgs.shape[0]
                # Resize to model resolution if needed
                if imgs.shape[1] != resolution or imgs.shape[2] != resolution:
                    resized = np.empty((T, resolution, resolution, 3), dtype=np.uint8)
                    for t in range(T):
                        resized[t] = cv2.resize(
                            imgs[t], (resolution, resolution),
                            interpolation=cv2.INTER_AREA,
                        )
                    imgs = resized
                # (T, H, W, C) → (T, C, H, W) float32 [0, 1]
                imgs = np.moveaxis(imgs, -1, 1).astype(np.float32) / 255.0
                obs_list.append(imgs)
            obs_all = np.concatenate(obs_list, axis=1)  # (T, C*num_views, H, W)
            T = obs_all.shape[0]

            # Load action
            action = f["action"][:].astype(np.float32)  # (T, A)

        # Encode in batches
        latent_chunks = []
        obs_tensor = torch.from_numpy(obs_all)
        for i in range(0, T, batch_size):
            chunk = obs_tensor[i : i + batch_size]
            z = encode_fn(chunk)  # (B, C_lat, H_lat, W_lat)
            latent_chunks.append(z.cpu())
        latent = torch.cat(latent_chunks, dim=0)  # (T, C_lat, H_lat, W_lat)

        # Save per-episode .pt
        torch.save(
            {"latent": latent, "action": torch.from_numpy(action)},
            os.path.join(output_split_dir, f"{ep_name}.pt"),
        )

        cumulative_len += T
        episode_ends.append(cumulative_len)

    # Save metadata
    torch.save(
        {
            "episode_ends": np.array(episode_ends, dtype=np.int64),
            "n_episodes": len(episode_paths),
            "obs_keys": obs_keys,
        },
        os.path.join(output_split_dir, "metadata.pt"),
    )
    print(f"  Saved {len(episode_paths)} episodes to {output_split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pre-encode RGB to latent tensors")
    parser.add_argument("--ckpt_path", required=True, help="Path to trained checkpoint")
    parser.add_argument("--dataset_dir", required=True, help="Root dataset dir (contains train/ and val/)")
    parser.add_argument("--output_dir", required=True, help="Output dir for latent .pt files")
    parser.add_argument("--obs_keys", nargs="+", default=["camera_1_color"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading encoder from {args.ckpt_path} ...")
    encode_fn, cfg, obs_keys, num_views = load_frozen_encoder(args.ckpt_path, args.device)
    resolution = int(cfg.dataset.resolution)
    print(f"  obs_keys={args.obs_keys}, resolution={resolution}, num_views={num_views}")

    for split in ["train", "val"]:
        split_dir = os.path.join(args.dataset_dir, split)
        if os.path.isdir(split_dir):
            output_split_dir = os.path.join(args.output_dir, split)
            process_split(encode_fn, split_dir, output_split_dir, args.obs_keys, args.batch_size, resolution)
        else:
            print(f"  Split dir {split_dir} not found, skipping.")

    print("Done.")


if __name__ == "__main__":
    main()
