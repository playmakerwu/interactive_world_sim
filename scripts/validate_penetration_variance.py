#!/usr/bin/env python3
"""Counterfactual Physical Violation Test: T-Block Penetration.

Test hypothesis: When two robot arms are commanded to aggressively converge
on the T-block center (a physical impossibility — rigid body penetration),
the ensemble's prediction variance should spike at the moment of impact.

PushT action space (bimanual_push, action_dim=4):
    [left_ee_x, left_ee_y, right_ee_x, right_ee_y]
    World-frame end-effector XY positions for left and right arms.

    Typical ranges from training data:
        dim 0 (left_x):  [0.01, 0.36]    — left arm X
        dim 1 (left_y):  [-0.02, 0.17]   — left arm Y
        dim 2 (right_x): [-0.05, 0.36]   — right arm X
        dim 3 (right_y): [-0.39, -0.05]  — right arm Y

    The T-block sits roughly in the center of the workspace.

Usage:
    python scripts/validate_penetration_variance.py \
        --ckpt_dir outputs/universe \
        --latent_dir data/mini/pusht_latent \
        --ae_ckpt outputs/pusht_cam1/checkpoints/best.ckpt
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))

from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel
from interactive_world_sim.datasets.latent_dynamics.latent_dataset import LatentDataset


def load_ensemble(ckpt_dir: str, ae_ckpt: str, device: str) -> list:
    ae_dir = os.path.dirname(os.path.dirname(ae_ckpt))
    base_cfg = OmegaConf.load(os.path.join(ae_dir, ".hydra", "config.yaml"))
    base_cfg.algorithm.training_stage = 2
    base_cfg.algorithm.load_ae = None
    base_cfg.algorithm.use_prebaked_latent = True

    ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    models = []
    for cp in ckpt_paths:
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


def synthesize_penetration_trajectory(H: int = 30):
    """Create adversarial action sequence: two arms converge to crush T-block.

    Phase 1 (t=0..9):   Normal — arms at typical starting positions
    Phase 2 (t=10..19):  Approach — arms move toward T-block center
    Phase 3 (t=20..H):   Penetration — arms overlap at the SAME point
                          (physically impossible with rigid T-block between them)

    Returns:
        actions: (H, 4) tensor of [left_x, left_y, right_x, right_y]
        impact_step: int, the step where penetration begins
    """
    actions = torch.zeros(H, 4)

    # Workspace geometry (from training data statistics)
    # Left arm operates in  x:[0.01, 0.36], y:[-0.02, 0.17]
    # Right arm operates in x:[-0.05, 0.36], y:[-0.39, -0.05]
    # T-block center is approximately at x=0.18, y=-0.10

    t_block_center_x = 0.18
    t_block_center_y_left = 0.05    # left arm's y-approach
    t_block_center_y_right = -0.15  # right arm's y-approach

    # Phase 1: Normal positions (arms far apart, typical starting config)
    left_start = torch.tensor([0.30, 0.15])     # left arm — top right
    right_start = torch.tensor([0.05, -0.35])    # right arm — bottom left

    # Phase 2: Approach target (arms moving toward the T-block)
    left_approach = torch.tensor([t_block_center_x, t_block_center_y_left])
    right_approach = torch.tensor([t_block_center_x, t_block_center_y_right])

    # Phase 3: Penetration target (BOTH arms at the SAME point — impossible!)
    # This simulates crushing through the T-block
    penetration_point = torch.tensor([t_block_center_x, -0.05])

    phase1_end = 10
    phase2_end = 20
    impact_step = phase2_end  # penetration begins here

    for t in range(H):
        if t < phase1_end:
            # Phase 1: Stationary at normal positions
            actions[t, 0:2] = left_start
            actions[t, 2:4] = right_start
        elif t < phase2_end:
            # Phase 2: Linear interpolation toward T-block
            alpha = (t - phase1_end) / (phase2_end - phase1_end)
            actions[t, 0:2] = left_start + alpha * (left_approach - left_start)
            actions[t, 2:4] = right_start + alpha * (right_approach - right_start)
        else:
            # Phase 3: Both arms converge to the SAME point (penetration!)
            alpha = min((t - phase2_end) / 5.0, 1.0)  # converge over 5 steps
            actions[t, 0:2] = left_approach + alpha * (penetration_point - left_approach)
            actions[t, 2:4] = right_approach + alpha * (penetration_point - right_approach)

    return actions, impact_step


def synthesize_normal_trajectory(H: int = 30):
    """Control trajectory: arms move gently without contacting the T-block.

    Both arms sweep across their respective workspace halves,
    never approaching the T-block center.
    """
    actions = torch.zeros(H, 4)
    for t in range(H):
        phase = t / H
        # Left arm: gentle sweep in upper workspace
        actions[t, 0] = 0.20 + 0.10 * np.sin(2 * np.pi * phase)
        actions[t, 1] = 0.12 + 0.03 * np.cos(2 * np.pi * phase)
        # Right arm: gentle sweep in lower workspace
        actions[t, 2] = 0.20 + 0.10 * np.cos(2 * np.pi * phase)
        actions[t, 3] = -0.30 + 0.05 * np.sin(2 * np.pi * phase)
    return actions


@torch.no_grad()
def autoregressive_rollout(
    models: list, z_0: torch.Tensor, action_seq: torch.Tensor, normalizer
) -> tuple:
    """Step-by-step autoregressive rollout through all ensemble members.

    At each step:
      1. Feed (z_t, a_t) to all models → get z_{t+1} predictions
      2. Compute per-step variance across models
      3. Use ensemble MEAN as next z_t (for stability)

    Returns:
        variances: list of per-step scalar variance
        all_preds: (H, N_models, B, C, H, W) — all predictions
    """
    device = z_0.device
    H = action_seq.shape[0] - 1  # action_seq includes t=0 history
    n_models = len(models)

    z_t = z_0.clone()  # (B, 1, C, H_lat, W_lat)
    variances = []

    for t in range(H):
        # Current action window: [t, t+1] for 1-step prediction
        # dynamics_forward expects action: (B, T_hist + T_pred, A)
        a_t = action_seq[t : t + 2].unsqueeze(0)  # (1, 2, A) — history + 1 pred

        preds_t = []
        for model in models:
            model.set_normalizer(normalizer)
            a_norm = model.normalizer["action"].normalize(a_t).to(device)
            z_next = model.dynamics_forward(z_t, a_norm)  # (B, 1, C, H, W)
            preds_t.append(z_next)

        preds_stack = torch.stack(preds_t, dim=0)  # (N_models, B, 1, C, H, W)

        # Variance across ensemble
        if n_models >= 2:
            var_t = torch.var(preds_stack, dim=0).mean().item()
        else:
            var_t = 0.0
        variances.append(var_t)

        # Use ensemble mean as next state
        z_t = preds_stack.mean(dim=0)  # (B, 1, C, H, W)

    return variances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--latent_dir", required=True)
    parser.add_argument("--ae_ckpt", required=True)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--n_trials", type=int, default=4,
                        help="Number of different z_0 starting states to average over")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_plot", default="outputs/universe/penetration_variance.png")
    args = parser.parse_args()

    print("=" * 70)
    print("  Counterfactual Physical Violation Test: T-Block Penetration")
    print("=" * 70)

    # 1. Load models
    print(f"\n[1] Loading ensemble from {args.ckpt_dir} ...")
    models = load_ensemble(args.ckpt_dir, args.ae_ckpt, args.device)
    print(f"  Ensemble size: {len(models)}")

    # 2. Load initial states
    print(f"\n[2] Loading initial states from {args.latent_dir} ...")
    ds_cfg = OmegaConf.create({
        "dataset_dir": args.latent_dir, "horizon": 2, "val_horizon": 2,
        "skip_frame": 1, "pad_before": 0, "pad_after": 0, "skip_idx": 1,
        "goal_sample": "intermediate", "action_mode": "single_ee",
        "bootstrap_seed": None, "debug": False,
    })
    ds = LatentDataset(ds_cfg)
    normalizer = ds.get_normalizer()

    # 3. Synthesize trajectories
    print(f"\n[3] Synthesizing adversarial trajectories (H={args.horizon}) ...")
    penetration_actions, impact_step = synthesize_penetration_trajectory(args.horizon)
    normal_actions = synthesize_normal_trajectory(args.horizon)
    print(f"  Impact/penetration begins at step {impact_step}")
    print(f"  Penetration actions sample (t={impact_step}): {penetration_actions[impact_step].tolist()}")
    print(f"  Penetration actions sample (t={args.horizon-1}): {penetration_actions[-1].tolist()}")
    print(f"  Normal actions sample (t=15): {normal_actions[15].tolist()}")

    # 4. Autoregressive rollouts (average over multiple z_0)
    print(f"\n[4] Running autoregressive rollouts ({args.n_trials} trials) ...")

    all_pen_var = []
    all_norm_var = []

    for trial in range(args.n_trials):
        idx = trial % len(ds)
        z_0 = ds[idx]["latent"][0:1].unsqueeze(0).to(args.device)  # (1, 1, C, H, W)

        pen_var = autoregressive_rollout(
            models, z_0, penetration_actions.to(args.device), normalizer
        )
        norm_var = autoregressive_rollout(
            models, z_0, normal_actions.to(args.device), normalizer
        )
        all_pen_var.append(pen_var)
        all_norm_var.append(norm_var)
        print(f"  Trial {trial+1}/{args.n_trials} done")

    # Average across trials
    pen_var_mean = np.mean(all_pen_var, axis=0)
    norm_var_mean = np.mean(all_norm_var, axis=0)
    pen_var_std = np.std(all_pen_var, axis=0)
    norm_var_std = np.std(all_norm_var, axis=0)

    # 5. Report
    print(f"\n[5] Results")
    print("=" * 70)
    print(f"  {'Step':>4s}  {'Normal Var':>12s}  {'Penetrate Var':>14s}  {'Ratio':>8s}  Phase")
    print("-" * 70)
    for t in range(len(pen_var_mean)):
        ratio = pen_var_mean[t] / norm_var_mean[t] if norm_var_mean[t] > 1e-12 else 0
        if t < 10:
            phase = "stationary"
        elif t < impact_step:
            phase = "approach"
        else:
            phase = "PENETRATION"
        marker = " <<<" if t == impact_step else ""
        print(f"  {t:4d}  {norm_var_mean[t]:12.6f}  {pen_var_mean[t]:14.6f}  {ratio:7.2f}x  {phase}{marker}")

    # Pre/post impact comparison
    pre_impact_pen = np.mean(pen_var_mean[:impact_step])
    post_impact_pen = np.mean(pen_var_mean[impact_step:])
    pre_impact_norm = np.mean(norm_var_mean[:impact_step])
    post_impact_norm = np.mean(norm_var_mean[impact_step:])
    print("-" * 70)
    print(f"  Penetration — Pre-impact avg: {pre_impact_pen:.6f}, Post-impact avg: {post_impact_pen:.6f}, "
          f"Spike: {post_impact_pen/pre_impact_pen:.2f}x" if pre_impact_pen > 1e-12 else "  N/A")
    print(f"  Normal      — Pre avg: {pre_impact_norm:.6f}, Post avg: {post_impact_norm:.6f}")

    # 6. Plot
    print(f"\n[6] Plotting ...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        steps = np.arange(len(pen_var_mean))

        # Left: Variance over time
        ax = axes[0]
        ax.plot(steps, pen_var_mean, "r-o", markersize=3, linewidth=2, label="Penetration (adversarial)")
        ax.fill_between(steps, pen_var_mean - pen_var_std, pen_var_mean + pen_var_std, color="red", alpha=0.15)
        ax.plot(steps, norm_var_mean, "g-s", markersize=3, linewidth=2, label="Normal (control)")
        ax.fill_between(steps, norm_var_mean - norm_var_std, norm_var_mean + norm_var_std, color="green", alpha=0.15)
        ax.axvline(x=impact_step, color="black", linestyle="--", linewidth=1.5, label=f"Impact (t={impact_step})")
        ax.axvspan(0, 10, alpha=0.05, color="blue", label="Phase 1: Stationary")
        ax.axvspan(10, impact_step, alpha=0.05, color="orange", label="Phase 2: Approach")
        ax.axvspan(impact_step, len(pen_var_mean), alpha=0.08, color="red", label="Phase 3: Penetration")
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Ensemble Variance", fontsize=12)
        ax.set_title(f"Ensemble Variance Over Time\n({len(models)} models, {args.n_trials} trials averaged)", fontsize=13)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

        # Right: Action trajectories (bird's eye)
        ax = axes[1]
        pa = penetration_actions.numpy()
        na = normal_actions.numpy()
        # Left arm trajectories
        ax.plot(pa[:, 0], pa[:, 1], "r-o", markersize=2, label="Penetrate: Left arm", linewidth=1.5)
        ax.plot(pa[:, 2], pa[:, 3], "r-^", markersize=2, label="Penetrate: Right arm", linewidth=1.5)
        ax.plot(na[:, 0], na[:, 1], "g-o", markersize=2, label="Normal: Left arm", alpha=0.6, linewidth=1)
        ax.plot(na[:, 2], na[:, 3], "g-^", markersize=2, label="Normal: Right arm", alpha=0.6, linewidth=1)
        # T-block center
        ax.scatter([0.18], [-0.05], s=200, c="blue", marker="*", zorder=10, label="T-block center")
        ax.set_xlabel("X (world frame)")
        ax.set_ylabel("Y (world frame)")
        ax.set_title("Action Trajectories (Bird's Eye View)")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
        plt.savefig(args.save_plot, dpi=150, bbox_inches="tight")
        print(f"  Saved: {args.save_plot}")
    except ImportError:
        print("  matplotlib not available.")

    print()


if __name__ == "__main__":
    main()
