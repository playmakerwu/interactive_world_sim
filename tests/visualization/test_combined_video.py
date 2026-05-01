"""Tests for rl.visualization.combined_video.

We don't need a real MPPI run for these — we synthesize a minimal run
artifact set (trajectory.mp4 + summary.json + iteration_log.pt) on the
fly in tmp_path and verify the renderer:
  * produces a video of the expected size
  * produces frames where the goal-arrow region contains red
  * produces frames where the current-arrow region contains green (frame 0)
  * does not crash when iteration_log records lack per-sample CV state
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from rl.visualization.combined_video import (
    CANVAS_PX,
    COMPOSITE_H,
    COMPOSITE_W,
    SIDE_PANEL_H,
    SIDE_PANEL_W,
    UPSAMPLE_FACTOR,
    _compose_frame,
    _compute_shared_y_lim,
    _render_gripper_arrow_panel,
    render_combined_video,
)


# ─── Demo stats fixture used by gripper-panel tests ────────────────────

def _make_fake_demo_stats() -> dict:
    """Synthetic demo_stats payload with WM-checkpoint-shaped normalizer."""
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

K_DEFAULT = 10
H_DEFAULT = 5  # so H+1 = 6 polyline points per top-K trajectory


def _empty_top_k_fields() -> dict:
    """Default-None top-K block for non-last iterations."""
    return {
        "top_k_intermediate_cx": None,
        "top_k_intermediate_cy": None,
        "top_k_intermediate_success": None,
        "top_k_indices": None,
        "top_k_rewards": None,
    }


def _make_top_k_polylines(
    K: int, Hp1: int,
    start_cx: float = 50.0, start_cy: float = 50.0,
    end_cx: float = 90.0, end_cy: float = 30.0,
    rewards: torch.Tensor | None = None,
) -> dict:
    """K straight polylines from (start_cx, start_cy) to slightly-jittered
    endpoints near (end_cx, end_cy). Rewards are descending."""
    rng = np.random.default_rng(7)
    cx = np.zeros((K, Hp1), dtype=np.float32)
    cy = np.zeros((K, Hp1), dtype=np.float32)
    for k in range(K):
        ex = end_cx + rng.uniform(-3, 3)
        ey = end_cy + rng.uniform(-3, 3)
        cx[k] = np.linspace(start_cx, ex, Hp1)
        cy[k] = np.linspace(start_cy, ey, Hp1)
    if rewards is None:
        rewards = torch.tensor(np.linspace(-0.05, -0.5, K), dtype=torch.float32)
    return {
        "top_k_intermediate_cx": torch.from_numpy(cx),
        "top_k_intermediate_cy": torch.from_numpy(cy),
        "top_k_intermediate_success": torch.ones(K, Hp1, dtype=torch.bool),
        "top_k_indices": torch.arange(K, dtype=torch.long),
        "top_k_rewards": rewards,
    }


def _make_iter_log_with_cv(
    n_iter: int = 3, n_sample: int = 8,
    cx_center: float = 64.0, cy_center: float = 64.0, jitter: float = 8.0,
    top_k_block: dict | None = None,
) -> list[dict]:
    """Synthesize an iteration log; only the LAST iter carries top-K data."""
    log: list[dict] = []
    rng = np.random.default_rng(0)
    for k in range(n_iter):
        rewards = torch.tensor(rng.uniform(-1.0, -0.05, size=n_sample), dtype=torch.float32)
        weights = torch.softmax(rewards * 50.0, dim=0)
        rec = {
            "iter": k,
            "rewards_all": rewards,
            "weights_all": weights,
            "reward_max": float(rewards.max()),
            "reward_min": float(rewards.min()),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_softmax_weighted": float((weights * rewards).sum()),
            "sample_cx": torch.tensor(
                rng.uniform(cx_center - jitter, cx_center + jitter, size=n_sample),
                dtype=torch.float32,
            ),
            "sample_cy": torch.tensor(
                rng.uniform(cy_center - jitter, cy_center + jitter, size=n_sample),
                dtype=torch.float32,
            ),
            "sample_sin_theta": torch.zeros(n_sample),
            "sample_cos_theta": torch.ones(n_sample),
            "sample_success": torch.ones(n_sample, dtype=torch.bool),
            **_empty_top_k_fields(),
        }
        log.append(rec)
    if top_k_block is None:
        top_k_block = _make_top_k_polylines(K=K_DEFAULT, Hp1=H_DEFAULT + 1)
    log[-1].update(top_k_block)
    return log


def _make_iter_log_no_cv(n_iter: int = 3, n_sample: int = 4) -> list[dict]:
    """MockEnv-style log: no per-sample CV, no top-K (regression coverage)."""
    log: list[dict] = []
    for k in range(n_iter):
        rewards = torch.linspace(-1.0, -0.1, n_sample)
        weights = torch.softmax(rewards * 10.0, dim=0)
        log.append({
            "iter": k,
            "rewards_all": rewards,
            "weights_all": weights,
            "reward_max": float(rewards.max()),
            "reward_min": float(rewards.min()),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_softmax_weighted": float((weights * rewards).sum()),
            "sample_cx": None,
            "sample_cy": None,
            "sample_sin_theta": None,
            "sample_cos_theta": None,
            "sample_success": None,
            **_empty_top_k_fields(),
        })
    return log


def _solid_frame(rgb_color=(40, 40, 40), size: int = 128) -> np.ndarray:
    return np.full((size, size, 3), rgb_color, dtype=np.uint8)


def _write_minimal_run_dir(
    tmp_path: Path,
    *,
    n_control_steps: int = 2,
    include_cv: bool = True,
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # trajectory.mp4 — n_control_steps + 1 solid frames
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(run_dir / "trajectory.mp4"), fourcc, 8, (128, 128))
    for _ in range(n_control_steps + 1):
        writer.write(cv2.cvtColor(_solid_frame(), cv2.COLOR_RGB2BGR))
    writer.release()

    # iteration_log.pt — one inner list per control_step
    iter_logs: list[list[dict]] = []
    for _ in range(n_control_steps):
        iter_logs.append(
            _make_iter_log_with_cv() if include_cv else _make_iter_log_no_cv()
        )
    torch.save(iter_logs, run_dir / "iteration_log.pt")

    # summary.json — minimal: per_step (length n_control_steps + 1) + goal_state
    per_step = [{
        "t": 0, "action": None, "reward": -1.5, "cv_success": True,
        "cx": 50.0, "cy": 60.0, "theta_deg": 30.0, "plan_wall_s": None,
    }]
    for k in range(n_control_steps):
        per_step.append({
            "t": k + 1, "action": [0.0] * 4,
            "reward": -0.5 - 0.1 * k,
            "cv_success": True,
            "cx": 55.0 + k, "cy": 58.0 - k, "theta_deg": 20.0 + k,
            "plan_wall_s": 1.23,
        })
    summary = {
        "config": {"n_sample": 8, "n_update_iter": 3},
        "config_deviation": {"changed": [], "reason": None,
                             "expected_impact_quantified": None},
        "per_step": per_step,
        "goal_state": {"cx": 64.0, "cy": 64.0, "theta_deg": 0.0},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir


# ─── 1. compose a single frame: goal draws red, current draws lime ─────

def test_compose_frame_draws_goal_and_current():
    rgb = _solid_frame()
    per_step_row = {
        "t": 1, "reward": -0.4, "cv_success": True,
        "cx": 80.0, "cy": 80.0, "theta_deg": 0.0,
        "action": [0.0, 0.0, 0.0, 0.0],
    }
    goal = {"cx": 30.0, "cy": 30.0, "theta_deg": 0.0}
    log = _make_iter_log_with_cv()

    composed = _compose_frame(rgb, per_step_row, goal, log, frame_idx=1,
                              demo_stats=_make_fake_demo_stats())

    assert composed.shape == (COMPOSITE_H, COMPOSITE_W, 3)
    assert composed.dtype == np.uint8

    # RGB canvas (after the left side panel) lives at columns [SIDE_PANEL_W,
    # SIDE_PANEL_W + CANVAS_PX).
    rgb_canvas = composed[:, SIDE_PANEL_W:SIDE_PANEL_W + CANVAS_PX]

    # Goal at (30, 30) * 4 = (120, 120) in canvas — should contain red
    goal_patch = rgb_canvas[120 - 8:120 + 8, 120 - 8:120 + 8]
    assert goal_patch[..., 0].max() > 150, "goal patch should contain red (R-channel)"

    # Current at (80, 80) * 4 = (320, 320) in canvas — should contain green
    cur_patch = rgb_canvas[320 - 8:320 + 8, 320 - 8:320 + 8]
    assert cur_patch[..., 1].max() > 150, "current patch should contain green (G-channel)"


# ─── 1b. compose draws top-K polylines from start to end region ────────

def test_compose_frame_draws_top_k_polylines():
    """Polylines from (50,50) to ~(90,30) should leave yellow-ish pixels
    (best trajectory) along the path between those two regions."""
    rgb = _solid_frame()
    per_step_row = {
        "t": 1, "reward": -0.05, "cv_success": True,
        "cx": 70.0, "cy": 40.0, "theta_deg": 0.0,
        "action": [0.0, 0.0, 0.0, 0.0],
    }
    goal = {"cx": 90.0, "cy": 30.0, "theta_deg": 0.0}
    # K=10 polylines from (50,50) to (90,30) area
    top_k = _make_top_k_polylines(K=10, Hp1=6,
                                  start_cx=50.0, start_cy=50.0,
                                  end_cx=90.0, end_cy=30.0)
    log = _make_iter_log_with_cv(top_k_block=top_k)

    composed = _compose_frame(rgb, per_step_row, goal, log, frame_idx=1,
                              demo_stats=_make_fake_demo_stats())
    # RGB canvas is between the side panels.
    left = composed[:, SIDE_PANEL_W:SIDE_PANEL_W + CANVAS_PX]

    # Best polyline pre-blend color is bright yellow (#FFD700 = R=255, G=215, B=0).
    # After the alpha=0.6 blend with gray (40, 40, 40) background:
    #   R ~ 169, G ~ 145, B ~ 16.
    # Halfway along the line from (50,50)->(90,30), in canvas (200,200)->(360,120),
    # midpoint is around (280, 160). Sample a 30x30 patch and look for any
    # post-blend yellow pixel (R medium-high, G medium, B low).
    patch = left[160 - 15:160 + 15, 280 - 15:280 + 15]
    yellow_mask = (
        (patch[..., 0] > 150) & (patch[..., 1] > 100) & (patch[..., 2] < 50)
    )
    assert yellow_mask.any(), (
        "expected at least one yellow polyline pixel along the top-K path"
    )


# ─── 2. compose handles missing per-sample CV gracefully ───────────────

def test_compose_frame_handles_missing_per_sample_cv():
    rgb = _solid_frame()
    per_step_row = {
        "t": 1, "reward": -0.4, "cv_success": True,
        "cx": 50.0, "cy": 50.0, "theta_deg": 10.0,
        "action": [0.0, 0.0, 0.0, 0.0],
    }
    goal = {"cx": 64.0, "cy": 64.0, "theta_deg": 0.0}
    log = _make_iter_log_no_cv()

    # Should NOT raise even though sample_cx/sample_cy are None.
    composed = _compose_frame(rgb, per_step_row, goal, log, frame_idx=1,
                              demo_stats=_make_fake_demo_stats())
    assert composed.shape == (COMPOSITE_H, COMPOSITE_W, 3)


# ─── 3. compose frame 0 (no plan executed yet) renders placeholder ─────

def test_compose_frame_zero_uses_placeholder_right_half():
    rgb = _solid_frame()
    per_step_row = {
        "t": 0, "reward": -1.5, "cv_success": True,
        "cx": 50.0, "cy": 60.0, "theta_deg": 30.0,
        "action": None,
    }
    goal = {"cx": 64.0, "cy": 64.0, "theta_deg": 0.0}

    composed = _compose_frame(rgb, per_step_row, goal, None, frame_idx=0,
                              demo_stats=_make_fake_demo_stats())
    assert composed.shape == (COMPOSITE_H, COMPOSITE_W, 3)
    # Reward plot (right plot region) lives at columns
    # [SIDE_PANEL_W + CANVAS_PX, SIDE_PANEL_W + 2*CANVAS_PX). Should be
    # mostly white (placeholder card).
    plot_region = composed[:, SIDE_PANEL_W + CANVAS_PX:SIDE_PANEL_W + 2 * CANVAS_PX]
    assert (plot_region > 200).mean() > 0.5, (
        "frame-0 reward-plot region should be mostly white (placeholder)"
    )


# ─── 4. end-to-end render produces an mp4 of the right shape ───────────

def test_render_combined_video_writes_mp4(tmp_path):
    run_dir = _write_minimal_run_dir(tmp_path, n_control_steps=2)
    out = render_combined_video(run_dir, fps=8, demo_stats=_make_fake_demo_stats())
    assert out == run_dir / "trajectory_combined.mp4"
    assert out.exists() and out.stat().st_size > 1000

    cap = cv2.VideoCapture(str(out))
    n_frames = 0
    last = None
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        last = frm
        n_frames += 1
    cap.release()
    assert n_frames == 3  # initial + 2 plan_steps
    assert last is not None
    H, W = last.shape[:2]
    assert (H, W) == (COMPOSITE_H, COMPOSITE_W)


# ─── 5. end-to-end works with iteration_log lacking per-sample CV ──────

def test_render_combined_video_works_without_per_sample_cv(tmp_path):
    run_dir = _write_minimal_run_dir(tmp_path, n_control_steps=2, include_cv=False)
    out = render_combined_video(run_dir, fps=8, demo_stats=_make_fake_demo_stats())
    assert out.exists()


# ─── 6. missing trajectory.mp4 surfaces clearly ────────────────────────

def test_render_combined_video_missing_inputs_raises(tmp_path):
    run_dir = tmp_path / "nope"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        render_combined_video(run_dir)


# ─── 7. shared y-axis covers the full reward range across plan_steps ───

def test_compute_shared_y_lim_covers_full_range():
    """Per-step logs with very different reward magnitudes should yield a
    single (y_min, y_max) that brackets all of them with 5% padding."""
    log_a = _make_iter_log_with_cv(n_iter=2)
    log_b = _make_iter_log_with_cv(n_iter=2)
    # spike one record in log_b far below
    log_b[0]["reward_min"] = -5.0
    log_b[0]["reward_mean"] = -2.5
    log_b[0]["reward_std"] = 1.0

    y_lim = _compute_shared_y_lim([log_a, log_b])
    assert y_lim is not None
    y_min, y_max = y_lim
    assert y_min < -3.0, "shared y_min must reach below the spike"
    assert y_max > -0.05, "shared y_max must include the highest reward in log_a"


def test_compute_shared_y_lim_empty_returns_none():
    assert _compute_shared_y_lim([]) is None
    assert _compute_shared_y_lim([[]]) is None


# ─── 8. gripper-arrow side panels ──────────────────────────────────────

def _count_pixels_close_to(img: np.ndarray, target_rgb, tol: int = 60) -> int:
    """Count pixels whose RGB is within tol of target on every channel."""
    diff = np.abs(img.astype(np.int16) - np.asarray(target_rgb, dtype=np.int16))
    return int(np.sum((diff < tol).all(axis=-1)))


def test_gripper_panel_dimensions():
    """Panel must be exactly SIDE_PANEL_H x SIDE_PANEL_W x 3 RGB uint8."""
    panel = _render_gripper_arrow_panel(
        executed_action_norm=[0.0, 0.0, 0.0, 0.0],
        gripper_label="Left",
        color=(0, 200, 220),
        demo_stats=_make_fake_demo_stats(),
    )
    assert panel.shape == (SIDE_PANEL_H, SIDE_PANEL_W, 3)
    assert panel.dtype == np.uint8


def test_composite_total_dimensions():
    """The full composite must be COMPOSITE_H x COMPOSITE_W = 512 x 1280."""
    rgb = _solid_frame()
    per_step_row = {
        "t": 1, "reward": -0.4, "cv_success": True,
        "cx": 64.0, "cy": 64.0, "theta_deg": 0.0,
        "action": [0.0, 0.0, 0.0, 0.0],
    }
    goal = {"cx": 64.0, "cy": 64.0, "theta_deg": 0.0}
    composed = _compose_frame(rgb, per_step_row, goal, _make_iter_log_with_cv(),
                              frame_idx=1, demo_stats=_make_fake_demo_stats())
    assert composed.shape == (COMPOSITE_H, COMPOSITE_W, 3)
    assert COMPOSITE_W == 2 * SIDE_PANEL_W + 2 * CANVAS_PX
    assert COMPOSITE_H == CANVAS_PX


def test_gripper_arrow_renders_when_action_nonzero():
    """A non-zero action must leave colored arrow pixels in the panel."""
    color = (0, 200, 220)
    panel = _render_gripper_arrow_panel(
        # Pick an action where dim 0 is far from the norm=0 -> raw=0.11
        # baseline so the resulting arrow is visibly long.
        executed_action_norm=[0.6, 0.6, 0.0, 0.0],
        gripper_label="Left",
        color=color,
        demo_stats=_make_fake_demo_stats(),
    )
    arrow_pixels = _count_pixels_close_to(panel, color)
    assert arrow_pixels > 30, (
        f"expected the cyan arrow to leave pixels in the panel; got {arrow_pixels}"
    )


def test_gripper_arrow_scales_with_magnitude():
    """A larger-magnitude action should produce a longer arrow (more
    colored pixels) than a small-magnitude one."""
    color = (0, 200, 220)
    stats = _make_fake_demo_stats()
    small = _render_gripper_arrow_panel(
        executed_action_norm=[0.0, 0.0, 0.0, 0.0],  # raw = (0.11, -0.04, 0.11, -0.19)
        gripper_label="Left", color=color, demo_stats=stats,
    )
    large = _render_gripper_arrow_panel(
        executed_action_norm=[1.0, 1.0, 0.0, 0.0],  # raw = (max, max, ...)
        gripper_label="Left", color=color, demo_stats=stats,
    )
    n_small = _count_pixels_close_to(small, color)
    n_large = _count_pixels_close_to(large, color)
    assert n_large > n_small, (
        f"large-magnitude arrow ({n_large} px) should exceed "
        f"small-magnitude arrow ({n_small} px)"
    )


def test_gripper_panel_handles_no_demo_stats_gracefully():
    """If demo_stats is None (e.g. cache file missing), panel still renders."""
    panel = _render_gripper_arrow_panel(
        executed_action_norm=[0.5, 0.5, 0.5, 0.5],
        gripper_label="Right", color=(255, 140, 0), demo_stats=None,
    )
    assert panel.shape == (SIDE_PANEL_H, SIDE_PANEL_W, 3)


def test_gripper_panel_handles_no_action_at_frame_zero():
    """Frame 0 (initial state, no plan executed yet) has executed_action_norm=None."""
    panel = _render_gripper_arrow_panel(
        executed_action_norm=None,
        gripper_label="Left", color=(0, 200, 220),
        demo_stats=_make_fake_demo_stats(),
    )
    assert panel.shape == (SIDE_PANEL_H, SIDE_PANEL_W, 3)
