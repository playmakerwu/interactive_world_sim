#!/usr/bin/env python3
"""Profile VRAM usage of Stage 2 prebaked training step.

Identifies where memory is consumed:
  - Model weights (decoder, dynamics, encoder, FVD)
  - Forward pass activations
  - Backward pass gradients

Usage:
    python scripts/profile_vram.py \
        --ckpt_path outputs/pusht_cam1/checkpoints/best.ckpt \
        --latent_dir data/mini/pusht_latent \
        --batch_size 32
"""

import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))

from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel
from interactive_world_sim.datasets.latent_dynamics.latent_dataset import LatentDataset


def mb(bytes_val):
    return f"{bytes_val / 1024**2:.1f} MB"


def vram_report(label):
    alloc = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    print(f"  [{label}] allocated={mb(alloc)}, peak={mb(peak)}")
    return alloc


def count_params(module, name=""):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = total - trainable
    size_mb = total * 4 / 1024**2  # float32
    print(f"    {name}: {total:,} params ({size_mb:.1f} MB), trainable={trainable:,}, frozen={frozen:,}")
    return total, size_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--latent_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()

    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    # ============================================================
    # PHASE 1: Model weight analysis
    # ============================================================
    print("=" * 60)
    print("PHASE 1: Model Weight Analysis")
    print("=" * 60)

    ds_cfg = OmegaConf.create({
        "dataset_dir": args.latent_dir, "horizon": args.horizon,
        "val_horizon": args.horizon, "skip_frame": 1, "pad_before": 1,
        "pad_after": 7, "skip_idx": 1, "goal_sample": "intermediate",
        "action_mode": "bimanual_push", "bootstrap_seed": None, "debug": False,
    })
    dataset = LatentDataset(ds_cfg)
    normalizer = dataset.get_normalizer()

    ckpt_dir = os.path.dirname(os.path.dirname(args.ckpt_path))
    model_cfg = OmegaConf.load(os.path.join(ckpt_dir, ".hydra", "config.yaml"))
    model_cfg.algorithm.training_stage = 2
    model_cfg.algorithm.load_ae = args.ckpt_path
    model_cfg.algorithm.dynamics_init_seed = 42
    model_cfg.algorithm.use_prebaked_latent = True

    vram_report("Before model creation")

    model = LatentWorldModel(model_cfg.algorithm).to(device)
    model.set_normalizer(normalizer)
    model.train()

    v_after_model = vram_report("After model.to(cuda)")

    print("\n  --- Per-component breakdown ---")
    count_params(model.decoder, "decoder")
    count_params(model.dynamics, "dynamics")
    count_params(model.encoder, "encoder")
    if model.validation_fvd_model is not None:
        count_params(model.validation_fvd_model, "FVD (I3D)")
    if model.validation_fid_model is not None:
        count_params(model.validation_fid_model, "FID")
    if model.validation_lpips_model is not None:
        count_params(model.validation_lpips_model, "LPIPS")
    count_params(model.noise_scheduler, "noise_scheduler")

    total_model_params = sum(p.numel() for p in model.parameters())
    total_model_mb = total_model_params * 4 / 1024**2
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  TOTAL: {total_model_params:,} params = {total_model_mb:.1f} MB (float32)")
    print(f"  Trainable: {trainable_params:,} = {trainable_params * 4 / 1024**2:.1f} MB")
    # Optimizer states (Adam: 2x param for m and v)
    print(f"  Adam optimizer states: ~{trainable_params * 4 * 2 / 1024**2:.1f} MB")

    # ============================================================
    # PHASE 2: Forward pass profiling
    # ============================================================
    print("\n" + "=" * 60)
    print(f"PHASE 2: Forward Pass (batch_size={args.batch_size}, horizon={args.horizon})")
    print("=" * 60)

    # Build batch
    batch = {
        k: torch.stack([dataset[i % len(dataset)][k] for i in range(args.batch_size)]).to(device)
        for k in ["latent", "action"]
    }
    print(f"  Batch: latent={batch['latent'].shape}, action={batch['action'].shape}")

    torch.cuda.reset_peak_memory_stats()
    v_before_fwd = vram_report("Before forward")

    # Call _training_step_prebaked but capture intermediate points
    from einops import rearrange

    z = batch["latent"].float().to(device)
    action = model.normalizer["action"].normalize(batch["action"]).float().to(device)
    z = rearrange(z, "b t c h w -> t b c h w")
    action = rearrange(action, "b t a -> t b a")
    vram_report("After rearrange")

    t, s = model._generate_noise_levels(z, model.dyn_infer_steps)
    noisy_z_t, noisy_z_s = model.noise_scheduler.add_noise_to_t_s(z, t, s)
    vram_report("After noise generation")

    # The critical forward pass
    pred_s = model._forward(model.dynamics, noisy_z_t, t, s, external_cond=action)
    v_after_fwd = vram_report("After dynamics forward")

    import torch.nn.functional as F
    loss = F.mse_loss(pred_s, noisy_z_s.detach())
    vram_report("After loss computation")

    # ============================================================
    # PHASE 3: Backward pass
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Backward Pass")
    print("=" * 60)

    torch.cuda.reset_peak_memory_stats()
    v_before_bwd = vram_report("Before backward")
    loss.backward()
    v_after_bwd = vram_report("After backward")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model weights on GPU:       {mb(v_after_model)}")
    print(f"  Forward activations:        {mb(v_after_fwd - v_before_fwd)}")
    print(f"  Backward peak:              {mb(torch.cuda.max_memory_allocated())}")
    print(f"  Overall peak VRAM:          {mb(torch.cuda.max_memory_allocated())}")
    print()

    # Precision check
    print("  --- Precision check ---")
    for name, p in list(model.dynamics.named_parameters())[:3]:
        print(f"    {name}: dtype={p.dtype}")

    # Config precision
    print(f"  Training precision config: {model_cfg.get('experiment', {}).get('training', {}).get('precision', 'NOT SET')}")
    print()
    print("  --- Recommendations ---")
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    if peak_gb > 20:
        print(f"  WARNING: Peak VRAM = {peak_gb:.1f} GB — too high for batch_size={args.batch_size}")
        print(f"  FIX 1: Use bf16-mixed precision (halves activation memory)")
        print(f"  FIX 2: Remove FVD model during training (it's only for validation)")
        if total_model_mb > 200:
            frozen_mb = (total_model_params - trainable_params) * 4 / 1024**2
            print(f"  FIX 3: Frozen weights consuming {frozen_mb:.0f} MB — consider not loading decoder")


if __name__ == "__main__":
    main()
