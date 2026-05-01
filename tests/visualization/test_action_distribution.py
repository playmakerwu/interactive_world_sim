"""Tests for rl.visualization.action_distribution.

Synthesizes a step_log fixture with the new ``last_iter_full_actions``
field, runs the renderer in tmp_path, and asserts that the output PNG
exists and has the expected structural elements.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
import pytest
import torch

from rl.visualization.action_distribution import (
    render_action_distributions,
    render_plan_step_action_distribution,
)


def _fake_demo_stats() -> dict:
    return {
        "mean_per_dim": torch.tensor([0.15, 0.09, 0.16, -0.34], dtype=torch.float32),
        "std_per_dim": torch.tensor([0.12, 0.08, 0.10, 0.07], dtype=torch.float32),
        "max_magnitude_left": 0.4,
        "max_magnitude_right": 0.5,
        "std_magnitude_left": 0.15,
        "std_magnitude_right": 0.12,
        "normalizer": {
            "scale": torch.tensor([3.98, 4.54, 4.03, 4.80], dtype=torch.float32),
            "offset": torch.tensor([-0.45, +0.19, -0.45, +0.93], dtype=torch.float32),
        },
    }


def _make_step_log(N: int = 16, H: int = 5, A: int = 4) -> dict:
    """Mimic the LAST iteration's iteration_log entry from a real run."""
    rng = np.random.default_rng(0)
    rewards = torch.tensor(rng.uniform(-1.0, -0.05, size=N), dtype=torch.float32)
    weights = torch.softmax(rewards * 50.0, dim=0)
    full_actions = torch.tensor(rng.uniform(-0.3, 0.3, size=(N, H, A)), dtype=torch.float32)
    return {
        "iter": 9,
        "rewards_all": rewards,
        "weights_all": weights,
        "reward_max": float(rewards.max()),
        "reward_min": float(rewards.min()),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_softmax_weighted": float((weights * rewards).sum()),
        "sample_cx": None, "sample_cy": None,
        "sample_sin_theta": None, "sample_cos_theta": None,
        "sample_success": None,
        "top_k_intermediate_cx": None, "top_k_intermediate_cy": None,
        "top_k_intermediate_success": None,
        "top_k_indices": None, "top_k_rewards": None,
        "last_iter_full_actions": full_actions,
    }


def _per_step_row(executed_action: list[float] | None = None) -> dict:
    if executed_action is None:
        executed_action = [0.0, 0.1, -0.05, 0.0]
    return {
        "t": 7, "reward": -0.4, "cv_success": True,
        "cx": 60.0, "cy": 40.0, "theta_deg": 12.0,
        "action": executed_action,
    }


# ─── 1. dist plot writes a PNG file ────────────────────────────────────

def test_dist_plot_renders(tmp_path: Path):
    out = tmp_path / "plan_step_007.png"
    result = render_plan_step_action_distribution(
        _make_step_log(), _per_step_row(), _fake_demo_stats(), out,
    )
    assert result == out
    assert out.exists() and out.stat().st_size > 1000
    arr = mpimg.imread(str(out))
    # 2x2 grid + suptitle => figure should have non-trivial dimensions.
    assert arr.shape[0] >= 400 and arr.shape[1] >= 600


# ─── 2. demo reference lines/bands present (uses figure pixel inspection) ─

def test_demo_reference_lines_present(tmp_path: Path):
    """Demo std band is light gray; demo mean is red dashed; both must
    leave detectable color signatures in the rendered figure."""
    out = tmp_path / "plan_step_007.png"
    render_plan_step_action_distribution(
        _make_step_log(), _per_step_row(), _fake_demo_stats(), out,
    )
    arr = (mpimg.imread(str(out))[..., :3] * 255).astype(np.uint8).astype(np.int16)
    # The demo-mean dashed line is tab:red (~(214, 39, 40)) at alpha=0.45
    # over white -> roughly (247, 158, 158): a pinkish hue. Detect any
    # pixel whose R is at least 50 above max(G, B) -- that's the unambiguous
    # red-dominant signature regardless of exact alpha blending.
    red_dominant = (arr[..., 0] - np.maximum(arr[..., 1], arr[..., 2])) > 50
    assert int(red_dominant.sum()) > 50, (
        f"expected red-dominant demo-mean-line pixels in dist plot; "
        f"found {int(red_dominant.sum())}"
    )


# ─── 3. exactly K=10 (or N if N<10) trajectory lines drawn ─────────────

def test_top10_lines_count_via_sampling(tmp_path: Path):
    """Indirect verification: the figure output should be larger when
    K=10 lines are drawn than when K=1 (single-trajectory) is drawn,
    since more lines deposit more colored ink. This is a structural
    rather than per-pixel check."""
    out_k10 = tmp_path / "k10.png"
    render_plan_step_action_distribution(
        _make_step_log(N=10, H=5), _per_step_row(),
        _fake_demo_stats(), out_k10, top_k=10,
    )
    out_k1 = tmp_path / "k1.png"
    render_plan_step_action_distribution(
        _make_step_log(N=10, H=5), _per_step_row(),
        _fake_demo_stats(), out_k1, top_k=1,
    )
    a10 = (mpimg.imread(str(out_k10))[..., :3] * 255).astype(np.uint8)
    a1 = (mpimg.imread(str(out_k1))[..., :3] * 255).astype(np.uint8)
    # Count blue-ish pixels (cyan->dark-blue gradient) — should be more
    # at K=10 than K=1.
    def cyan_count(img):
        return int(((img[..., 0] < 100) & (img[..., 1] > 100) & (img[..., 2] > 100)).sum()
                   + ((img[..., 0] < 60) & (img[..., 1] < 100) & (img[..., 2] > 100)).sum())
    assert cyan_count(a10) > cyan_count(a1) + 100, (
        f"K=10 ({cyan_count(a10)}) should have more line pixels than K=1 ({cyan_count(a1)})"
    )


# ─── 4. executed action draws a horizontal black line ──────────────────

def test_executed_action_line_appears(tmp_path: Path):
    out = tmp_path / "plan_step.png"
    render_plan_step_action_distribution(
        _make_step_log(), _per_step_row([0.5, 0.5, 0.5, 0.5]),
        _fake_demo_stats(), out,
    )
    arr = (mpimg.imread(str(out))[..., :3] * 255).astype(np.uint8)
    # Detect very-dark pixels (exec action line is solid black at α=0.9)
    black_mask = (arr[..., 0] < 40) & (arr[..., 1] < 40) & (arr[..., 2] < 40)
    assert black_mask.sum() > 100, (
        f"expected the executed-action black line to show up; found {int(black_mask.sum())}"
    )


# ─── 5. graceful skip when last_iter_full_actions missing ──────────────

def test_render_returns_none_without_full_actions(tmp_path: Path):
    log = _make_step_log()
    log["last_iter_full_actions"] = None  # legacy run, no capture
    out = tmp_path / "should_not_exist.png"
    result = render_plan_step_action_distribution(
        log, _per_step_row(), _fake_demo_stats(), out,
    )
    assert result is None
    assert not out.exists()


# ─── 6. batch entry point produces one PNG per plan_step ───────────────

def test_render_action_distributions_writes_one_png_per_step(tmp_path: Path):
    iter_logs = [
        [_make_step_log(N=8, H=4)],
        [_make_step_log(N=8, H=4)],
        [_make_step_log(N=8, H=4)],
    ]
    per_step = [{"t": 0, "action": None, "reward": -1.5, "cv_success": True,
                 "cx": 50.0, "cy": 50.0, "theta_deg": 0.0}]
    for k in range(3):
        per_step.append(_per_step_row([0.1 * k] * 4))
    written = render_action_distributions(
        iter_logs, per_step, tmp_path, demo_stats=_fake_demo_stats(),
    )
    assert len(written) == 3
    for k in range(3):
        assert (tmp_path / "action_dist" / f"plan_step_{k:03d}.png").exists()
