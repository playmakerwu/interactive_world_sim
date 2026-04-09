#!/usr/bin/env python3
"""Validate Ensemble Epistemic Uncertainty via Prediction Variance.

Methodology:
    1. Load N trained dynamics models (ensemble members) in eval mode
    2. Sample real (z_t, a_t) pairs from the validation set
    3. Synthesize abnormal actions (OOD) by noise injection + boundary violation
    4. Forward both normal and abnormal actions through all ensemble members
    5. Compute prediction variance across the ensemble dimension
    6. A well-calibrated ensemble should show:
       - LOW variance on in-distribution (normal) actions
       - HIGH variance on out-of-distribution (abnormal) actions

Usage:
    python scripts/validate_ensemble_variance.py \
        --ckpt_dir outputs/universe \
        --latent_dir data/mini/pusht_latent \
        --ae_ckpt outputs/pusht_cam1/checkpoints/best.ckpt

    # With more prediction steps
    python scripts/validate_ensemble_variance.py \
        --ckpt_dir outputs/universe \
        --latent_dir data/full/pusht_latent \
        --ae_ckpt outputs/pusht_cam1/checkpoints/best.ckpt \
        --n_samples 32 \
        --pred_horizon 8
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Register resolvers before any config loading
OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))

from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel
from interactive_world_sim.datasets.latent_dynamics.latent_dataset import LatentDataset


def load_ensemble(ckpt_dir: str, ae_ckpt: str, device: str) -> list:
    """Load all ensemble checkpoints from a directory."""
    ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not ckpt_paths:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")

    # Use the AE checkpoint's config as base
    ae_dir = os.path.dirname(os.path.dirname(ae_ckpt))
    base_cfg = OmegaConf.load(os.path.join(ae_dir, ".hydra", "config.yaml"))
    base_cfg.algorithm.training_stage = 2
    base_cfg.algorithm.load_ae = None  # weights already in checkpoint
    base_cfg.algorithm.use_prebaked_latent = True

    models = []
    for cp in ckpt_paths:
        # Try strict load first; if it fails (e.g. best.ckpt has encoder keys
        # but our config uses placeholder encoder), retry with strict=False
        try:
            model = LatentWorldModel.load_from_checkpoint(
                cp, cfg=base_cfg.algorithm, map_location=device,
                weights_only=False, strict=True,
            )
        except RuntimeError:
            model = LatentWorldModel.load_from_checkpoint(
                cp, cfg=base_cfg.algorithm, map_location=device,
                weights_only=False, strict=False,
            )
        model.eval()
        model.to(device)
        models.append(model)
        print(f"  Loaded: {Path(cp).name}")

    return models


def sample_validation_data(
    latent_dir: str, n_samples: int, horizon: int
) -> tuple:
    """Sample (z_0, action_sequence) pairs from validation set."""
    cfg = OmegaConf.create({
        "dataset_dir": latent_dir,
        "horizon": horizon + 1,  # +1 because we need z_0 + horizon actions
        "val_horizon": horizon + 1,
        "skip_frame": 1,
        "pad_before": 0,
        "pad_after": 0,
        "skip_idx": 1,
        "goal_sample": "intermediate",
        "action_mode": "single_ee",
        "bootstrap_seed": None,
        "debug": False,
    })
    ds = LatentDataset(cfg)
    normalizer = ds.get_normalizer()

    # Sample n_samples trajectories
    indices = np.random.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    samples = [ds[i] for i in indices]

    latents = torch.stack([s["latent"] for s in samples])  # (N, T, C, H, W)
    actions = torch.stack([s["action"] for s in samples])  # (N, T, A)

    z_0 = latents[:, 0:1]           # (N, 1, C, H, W) — initial state
    action_seq = actions[:, :horizon + 1]  # (N, horizon+1, A) — includes history

    return z_0, action_seq, normalizer, ds


def synthesize_abnormal_actions(real_actions: torch.Tensor):
    """Create truly OOD actions that are far outside the training distribution.

    The key insight: PushT actions have a very narrow range (~[-0.39, 0.36])
    with std ~0.05-0.1. To be genuinely OOD, we need actions that are
    10-100x beyond this range, not just a few sigma away.
    """
    B, T, A = real_actions.shape

    a_min = real_actions.min(dim=0).values.min(dim=0).values
    a_max = real_actions.max(dim=0).values.max(dim=0).values
    a_range = (a_max - a_min).abs().clamp(min=1e-6)

    abnormal = {}

    # Strategy 1: 10x the full action range (truly extreme)
    abnormal["random_10x_range"] = torch.empty_like(real_actions).uniform_(-1, 1) * a_range * 10

    # Strategy 2: 50x range — absurdly large actions
    abnormal["random_50x_range"] = torch.empty_like(real_actions).uniform_(-1, 1) * a_range * 50

    # Strategy 3: Constant saturated — all dims at +max*10
    extreme_val = a_max.abs().max().item() * 10
    abnormal["constant_+10x"] = torch.full_like(real_actions, extreme_val)

    # Strategy 4: Reversed actions — negate the real actions and scale up
    abnormal["reversed_5x"] = -real_actions * 5

    # Strategy 5: Random sign flipping + large magnitude
    signs = torch.sign(torch.randn_like(real_actions))
    abnormal["random_sign_20x"] = signs * a_range * 20

    # Strategy 6: Zero actions (control — may or may not be OOD)
    abnormal["zero_action"] = torch.zeros_like(real_actions)

    return abnormal


@torch.no_grad()
def ensemble_predict(
    models: list, z_0: torch.Tensor, action_seq: torch.Tensor, normalizer
) -> torch.Tensor:
    """Run forward pass through all ensemble members.

    Returns:
        predictions: (N_models, B, T_pred, C, H, W)
    """
    preds = []
    for model in models:
        model.set_normalizer(normalizer)
        # Normalize actions — ensure result is on the correct device
        action_norm = model.normalizer["action"].normalize(action_seq)
        action_norm = action_norm.to(z_0.device)
        z_pred = model.dynamics_forward(z_0, action_norm)  # (B, T_pred, C, H, W)
        preds.append(z_pred)

    return torch.stack(preds, dim=0)  # (N_models, B, T_pred, C, H, W)


def compute_variance(predictions: torch.Tensor) -> torch.Tensor:
    """Compute per-sample epistemic variance across ensemble members.

    Args:
        predictions: (N_models, B, T_pred, C, H, W)
    Returns:
        variance: (B,) — scalar variance per sample
    """
    if predictions.shape[0] < 2:
        # Single model: variance is 0 by definition
        return torch.zeros(predictions.shape[1], device=predictions.device)
    # Variance across ensemble dimension (dim=0), then mean over all other dims
    # This gives a scalar "disagreement score" per sample
    var = torch.var(predictions, dim=0)  # (B, T_pred, C, H, W)
    return var.mean(dim=(-1, -2, -3, -4))  # (B,) — mean over T, C, H, W


def main():
    parser = argparse.ArgumentParser(description="Validate Ensemble Epistemic Uncertainty")
    parser.add_argument("--ckpt_dir", required=True, help="Directory containing ensemble .ckpt files")
    parser.add_argument("--latent_dir", required=True, help="Pre-encoded latent dataset directory")
    parser.add_argument("--ae_ckpt", required=True, help="Original AE checkpoint (for config)")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of validation samples")
    parser.add_argument("--pred_horizon", type=int, default=4, help="Number of future steps to predict")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_plot", default=None, help="Path to save variance plot (e.g., variance.png)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Ensemble Epistemic Uncertainty Validation")
    print("=" * 70)

    # --- 1. Load Ensemble ---
    print(f"\n[1/5] Loading ensemble from {args.ckpt_dir} ...")
    models = load_ensemble(args.ckpt_dir, args.ae_ckpt, args.device)
    n_models = len(models)
    print(f"  Ensemble size: {n_models} models")
    if n_models < 2:
        print("  WARNING: Need ≥2 models for meaningful variance. Results with 1 model")
        print("           will show 0 variance (trivially). Interpret as smoke test only.")

    # --- 2. Sample Validation Data ---
    print(f"\n[2/5] Sampling {args.n_samples} validation trajectories ...")
    z_0, action_seq, normalizer, ds = sample_validation_data(
        args.latent_dir, args.n_samples, args.pred_horizon
    )
    z_0 = z_0.to(args.device)
    action_seq = action_seq.to(args.device)
    print(f"  z_0: {z_0.shape}")
    print(f"  action_seq: {action_seq.shape}")
    print(f"  Action range: [{action_seq.min():.3f}, {action_seq.max():.3f}]")

    # --- 3. Synthesize Abnormal Actions ---
    print(f"\n[3/5] Synthesizing abnormal (OOD) actions ...")
    abnormal_actions = synthesize_abnormal_actions(action_seq)
    for name, a in abnormal_actions.items():
        print(f"  {name}: range [{a.min():.3f}, {a.max():.3f}]")

    # --- 4. Forward Pass & Variance ---
    print(f"\n[4/5] Running ensemble forward passes ...")

    # Normal actions
    preds_normal = ensemble_predict(models, z_0, action_seq, normalizer)
    var_normal = compute_variance(preds_normal)

    # Abnormal actions
    results = {"normal_action": var_normal.cpu()}
    for name, abn_action in abnormal_actions.items():
        abn_action = abn_action.to(args.device)
        # Replace only the action, keep the same z_0
        preds_abn = ensemble_predict(models, z_0, abn_action, normalizer)
        var_abn = compute_variance(preds_abn)
        results[name] = var_abn.cpu()

    # --- 5. Report ---
    print(f"\n[5/5] Results")
    print("=" * 70)
    print(f"  Ensemble size: {n_models} models | Samples: {args.n_samples} | "
          f"Pred horizon: {args.pred_horizon} steps")
    print("=" * 70)
    print(f"  {'Action Type':<25s} {'Mean Var':>12s} {'Std Var':>12s} {'Max Var':>12s} {'Ratio':>8s}")
    print("-" * 70)

    baseline_mean = results["normal_action"].mean().item()
    for name, var in results.items():
        mean_v = var.mean().item()
        std_v = var.std().item()
        max_v = var.max().item()
        ratio = mean_v / baseline_mean if baseline_mean > 1e-12 else float("inf")
        marker = "  ←baseline" if name == "normal_action" else ""
        print(f"  {name:<25s} {mean_v:>12.6f} {std_v:>12.6f} {max_v:>12.6f} {ratio:>7.1f}x{marker}")

    print("-" * 70)

    if n_models >= 2:
        max_ratio = max(
            results[k].mean().item() / baseline_mean
            for k in results if k != "normal_action"
        ) if baseline_mean > 1e-12 else 0
        if max_ratio > 1.5:
            print("  ✓ PASS: OOD actions produce significantly higher variance")
            print(f"          Max ratio = {max_ratio:.1f}x (threshold: >1.5x)")
        else:
            print("  ✗ FAIL: Ensemble does NOT show higher variance for OOD actions")
            print(f"          Max ratio = {max_ratio:.1f}x (need >1.5x)")
            print("          Possible causes:")
            print("          - Insufficient training diversity (same bootstrap?)")
            print("          - Dynamics models have collapsed to similar solutions")
            print("          - Pred horizon too short to amplify disagreement")
    else:
        print("  ⚠ SKIP: Cannot compute meaningful variance with 1 model")
        print("          Variance is trivially 0. This run validates code paths only.")

    # --- Optional: Plot ---
    if args.save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Bar chart
            ax = axes[0]
            names = list(results.keys())
            means = [results[n].mean().item() for n in names]
            colors = ["#2ecc71" if n == "normal_action" else "#e74c3c" for n in names]
            bars = ax.bar(range(len(names)), means, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
            ax.set_ylabel("Mean Ensemble Variance")
            ax.set_title(f"Ensemble Variance by Action Type\n({n_models} models, {args.n_samples} samples)")
            ax.grid(axis="y", alpha=0.3)

            # Distribution plot
            ax = axes[1]
            for name, var in results.items():
                label = name.replace("_", " ")
                color = "#2ecc71" if name == "normal_action" else "#e74c3c"
                ax.hist(var.numpy(), bins=20, alpha=0.5, label=label, color=color, edgecolor="black", linewidth=0.3)
            ax.set_xlabel("Per-sample Ensemble Variance")
            ax.set_ylabel("Count")
            ax.set_title("Variance Distribution")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150, bbox_inches="tight")
            print(f"\n  Plot saved to: {args.save_plot}")
        except ImportError:
            print("\n  matplotlib not available, skipping plot.")

    print()


if __name__ == "__main__":
    main()
