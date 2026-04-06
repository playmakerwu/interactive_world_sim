#!/usr/bin/env python3
"""Smoke test: run forward + backward with pre-baked latent data.

Usage (after Phase 1 pre-encoding):
    python scripts/test_training_step.py \
        --ckpt_path outputs/pusht_cam1/checkpoints/best.ckpt \
        --latent_dir data/mini/pusht_latent
"""

import argparse
import os

import numpy as np
import torch
from omegaconf import OmegaConf

# Register resolvers BEFORE any config loading
OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))

from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel
from interactive_world_sim.datasets.latent_dynamics.latent_dataset import LatentDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--latent_dir", required=True)
    parser.add_argument("--dynamics_init_seed", type=int, default=42)
    parser.add_argument("--bootstrap_seed", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Build dataset ---
    ds_cfg = OmegaConf.create({
        "dataset_dir": args.latent_dir,
        "horizon": 16,
        "val_horizon": 16,
        "skip_frame": 1,
        "pad_before": 1,
        "pad_after": 7,
        "skip_idx": 1,
        "goal_sample": "intermediate",
        "action_mode": "bimanual_push",
        "bootstrap_seed": args.bootstrap_seed,
        "debug": False,
    })
    dataset = LatentDataset(ds_cfg)
    normalizer = dataset.get_normalizer()
    print(f"Dataset size: {len(dataset)}")

    # --- 2. Build model from checkpoint config ---
    ckpt_dir = os.path.dirname(os.path.dirname(args.ckpt_path))
    model_cfg = OmegaConf.load(os.path.join(ckpt_dir, ".hydra", "config.yaml"))

    # Override for Stage 2 + pre-baked latent
    model_cfg.algorithm.training_stage = 2
    model_cfg.algorithm.load_ae = args.ckpt_path
    model_cfg.algorithm.dynamics_init_seed = args.dynamics_init_seed
    model_cfg.algorithm.use_prebaked_latent = True

    print("Building model...")
    model = LatentWorldModel(model_cfg.algorithm).to(device)
    model.set_normalizer(normalizer)
    model.train()

    # --- 3. Verify freeze ---
    enc_params = list(model.encoder.parameters())
    enc_grad = sum(p.requires_grad for p in enc_params)
    dec_grad = sum(p.requires_grad for p in model.decoder.parameters())
    dyn_grad = sum(p.requires_grad for p in model.dynamics.parameters())
    print(f"Encoder params: {len(enc_params)} tensors, {enc_grad} with grad (expect 0)")
    print(f"Decoder params with grad: {dec_grad} (expect 0)")
    print(f"Dynamics params with grad: {dyn_grad} (expect >0)")
    assert enc_grad == 0, "Encoder should be frozen!"
    assert dec_grad == 0, "Decoder should be frozen!"
    assert dyn_grad > 0, "Dynamics should be trainable!"

    # --- 4. Verify dynamics_init_seed reproducibility ---
    model_cfg2 = model_cfg.copy()
    model2 = LatentWorldModel(model_cfg2.algorithm)
    p1 = list(model.dynamics.parameters())[0].data.flatten()[:5]
    p2 = list(model2.dynamics.parameters())[0].data.flatten()[:5]
    print(f"Dynamics weights (seed={args.dynamics_init_seed}):")
    print(f"  Model 1: {p1}")
    print(f"  Model 2: {p2}")
    assert torch.allclose(p1.cpu(), p2.cpu()), "Same seed should produce same weights!"
    print("  Seed reproducibility verified!")

    # --- 5. Run forward + backward ---
    batch = {
        k: torch.stack([dataset[i][k] for i in range(2)]).to(device)
        for k in ["latent", "action"]
    }
    print(f"\nBatch: latent={batch['latent'].shape}, action={batch['action'].shape}")

    # VRAM before
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated() / 1024**2

    output = model._training_step_prebaked(batch, batch_idx=0)
    loss = output["loss"]
    print(f"Forward pass loss: {loss.item():.6f}")

    loss.backward()
    vram_peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Backward pass completed.")
    print(f"VRAM: baseline={vram_before:.0f}MB, peak={vram_peak:.0f}MB")

    # Verify gradients only on dynamics
    dyn_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.dynamics.parameters()
    )
    enc_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    dec_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.decoder.parameters()
    )
    print(f"Dynamics has gradients: {dyn_has_grad} (expect True)")
    print(f"Encoder has gradients:  {enc_has_grad} (expect False)")
    print(f"Decoder has gradients:  {dec_has_grad} (expect False)")
    assert dyn_has_grad, "Dynamics must receive gradients!"
    assert not enc_has_grad, "Encoder must NOT receive gradients!"
    assert not dec_has_grad, "Decoder must NOT receive gradients!"

    print("\n=== Phase 3 冒烟测试全部通过! ===")


if __name__ == "__main__":
    main()
