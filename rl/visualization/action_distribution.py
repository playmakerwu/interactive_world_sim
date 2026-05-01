"""Per-plan-step action distribution diagnostic plots.

For each plan_step, render a 2x2 figure (one subplot per action dim)
showing the top-10 trajectory action sequences in raw physical units,
alongside the demo distribution's mean and ±1 std band for that dim
and the executed action's value (chosen by MPPI for this control_step).

Lets the user see at a glance whether MPPI's chosen actions are
near the demo distribution or off in some sub-region of action space
(the under-exploration signal documented in
``ACTION_NORMALIZATION_AUDIT.md``).

These figures are NOT inserted into the per-frame video — they're
standalone diagnostics dropped at ``<run_dir>/action_dist/plan_step_NNN.png``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl.visualization.demo_action_stats import (
    denormalize_action,
    load_or_compute_demo_action_stats,
)

DIM_LABELS = {
    0: "dim 0 (left x)",
    1: "dim 1 (left y)",
    2: "dim 2 (right x)",
    3: "dim 3 (right y)",
}


def _to_raw(actions_norm: torch.Tensor, normalizer: dict) -> torch.Tensor:
    """Denormalize MPPI normalized actions in [-1, +1] back to raw meters."""
    return denormalize_action(actions_norm, normalizer)


def render_plan_step_action_distribution(
    step_log: dict[str, Any],
    per_step_row: dict[str, Any],
    demo_stats: dict,
    out_path: Path,
    top_k: int = 10,
) -> Path | None:
    """Render one plan_step's action distribution figure.

    Returns the output path. Returns ``None`` (no file written) if the
    step_log lacks the per-sample action data needed (e.g., the planner
    that produced this run pre-dates the ``last_iter_full_actions``
    capture in commit eafa33d+).
    """
    full_actions = step_log.get("last_iter_full_actions")
    rewards_all = step_log.get("rewards_all")
    if full_actions is None or rewards_all is None:
        return None
    full_actions = full_actions.detach().cpu().float()  # (N, H, A)
    rewards_all = rewards_all.detach().cpu().float()    # (N,)

    # Pick top-K by reward (descending). Match the same ranking convention
    # the polyline renderer uses, so this figure tells the user "the lines
    # here are the same trajectories shown in the trajectory viz".
    K = min(int(top_k), int(rewards_all.numel()))
    top_idx = torch.argsort(rewards_all, descending=True)[:K]
    top_actions_norm = full_actions[top_idx]            # (K, H, A)

    # Denormalize for display in raw physical units
    normalizer = demo_stats["normalizer"]
    top_actions_raw = _to_raw(top_actions_norm, normalizer).numpy()  # (K, H, A)
    executed_norm = per_step_row.get("action")
    executed_raw = None
    if executed_norm is not None:
        executed_norm_t = torch.tensor(executed_norm, dtype=torch.float32)
        executed_raw = _to_raw(executed_norm_t, normalizer).tolist()

    demo_mean = demo_stats["mean_per_dim"].cpu().numpy()
    demo_std = demo_stats["std_per_dim"].cpu().numpy()

    H = top_actions_raw.shape[1]
    A = top_actions_raw.shape[2]
    horizon_steps = np.arange(H)

    # Color gradient: best (k=0) -> bright cyan, worst (k=K-1) -> dark blue
    cyan = np.array([0, 200, 255]) / 255.0
    dark_blue = np.array([20, 40, 130]) / 255.0
    if K == 1:
        line_colors = [cyan]
    else:
        line_colors = [
            (1 - i / (K - 1)) * cyan + (i / (K - 1)) * dark_blue
            for i in range(K)
        ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    flat_axes = axes.flatten()
    final_reward = per_step_row.get("reward")
    final_reward_str = "nan" if final_reward is None else f"{float(final_reward):+.3f}"
    t_label = per_step_row.get("t", -1)
    fig.suptitle(
        f"action dist | control_step {int(t_label):03d} | "
        f"exec_reward = {final_reward_str}",
        fontsize=13,
    )

    for d in range(min(A, 4)):
        ax = flat_axes[d]
        # Demo std band (light gray)
        ax.axhspan(
            demo_mean[d] - demo_std[d],
            demo_mean[d] + demo_std[d],
            color="lightgray", alpha=0.45, zorder=0,
            label="demo mean +/- 1 std",
        )
        # Demo mean (red dashed)
        ax.axhline(
            demo_mean[d], color="tab:red", linestyle="--", alpha=0.45,
            linewidth=1.0, zorder=1, label="demo mean",
        )
        # Top-K trajectories — draw worst -> best so best ends up on top
        for k in range(K - 1, -1, -1):
            ax.plot(
                horizon_steps, top_actions_raw[k, :, d],
                color=line_colors[k], linewidth=1.4, alpha=0.6,
                zorder=2 + (K - k),
            )
        # Executed action (solid black horizontal)
        if executed_raw is not None:
            ax.axhline(
                executed_raw[d], color="black", linewidth=2.0, alpha=0.9,
                zorder=K + 5, label="executed action",
            )
        ax.set_xlabel("horizon step")
        ax.set_ylabel("action (raw, m)")
        ax.set_title(DIM_LABELS.get(d, f"dim {d}"), fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xticks(horizon_steps)
        if d == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─── Public batch entry point ──────────────────────────────────────────

def render_action_distributions(
    iteration_logs: list[list[dict[str, Any]]],
    per_step: list[dict[str, Any]],
    out_dir: Path,
    demo_stats: dict | None = None,
    top_k: int = 10,
) -> list[Path]:
    """Render one PNG per plan_step under ``<out_dir>/action_dist/``.

    Skips silently if a step_log lacks ``last_iter_full_actions`` (e.g.,
    legacy runs from before the planner extension). Returns the list of
    written paths.
    """
    if demo_stats is None:
        demo_stats = load_or_compute_demo_action_stats()
    dist_dir = out_dir / "action_dist"
    written: list[Path] = []
    for step_idx, step_log in enumerate(iteration_logs):
        if not step_log:
            continue
        last_iter = step_log[-1]
        per_step_row = per_step[step_idx + 1] if step_idx + 1 < len(per_step) else {}
        out_path = dist_dir / f"plan_step_{step_idx:03d}.png"
        result = render_plan_step_action_distribution(
            last_iter, per_step_row, demo_stats, out_path, top_k=top_k,
        )
        if result is not None:
            written.append(result)
    return written
