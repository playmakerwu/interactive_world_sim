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
    UPSAMPLE_FACTOR,
    _compose_frame,
    _compute_shared_y_lim,
    render_combined_video,
)

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
    }
    goal = {"cx": 30.0, "cy": 30.0, "theta_deg": 0.0}
    log = _make_iter_log_with_cv()

    composed = _compose_frame(rgb, per_step_row, goal, log, frame_idx=1)

    expected_w = CANVAS_PX * 2
    expected_h = CANVAS_PX
    assert composed.shape == (expected_h, expected_w, 3)
    assert composed.dtype == np.uint8

    # Left half = first CANVAS_PX cols
    left = composed[:, :CANVAS_PX]

    # Goal at (30, 30) * 4 = (120, 120) in canvas — should contain red
    goal_patch = left[120 - 8:120 + 8, 120 - 8:120 + 8]
    assert goal_patch[..., 0].max() > 150, "goal patch should contain red (R-channel)"

    # Current at (80, 80) * 4 = (320, 320) in canvas — should contain green
    cur_patch = left[320 - 8:320 + 8, 320 - 8:320 + 8]
    assert cur_patch[..., 1].max() > 150, "current patch should contain green (G-channel)"


# ─── 1b. compose draws top-K polylines from start to end region ────────

def test_compose_frame_draws_top_k_polylines():
    """Polylines from (50,50) to ~(90,30) should leave yellow-ish pixels
    (best trajectory) along the path between those two regions."""
    rgb = _solid_frame()
    per_step_row = {
        "t": 1, "reward": -0.05, "cv_success": True,
        "cx": 70.0, "cy": 40.0, "theta_deg": 0.0,
    }
    goal = {"cx": 90.0, "cy": 30.0, "theta_deg": 0.0}
    # K=10 polylines from (50,50) to (90,30) area
    top_k = _make_top_k_polylines(K=10, Hp1=6,
                                  start_cx=50.0, start_cy=50.0,
                                  end_cx=90.0, end_cy=30.0)
    log = _make_iter_log_with_cv(top_k_block=top_k)

    composed = _compose_frame(rgb, per_step_row, goal, log, frame_idx=1)
    left = composed[:, :CANVAS_PX]

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
    }
    goal = {"cx": 64.0, "cy": 64.0, "theta_deg": 0.0}
    log = _make_iter_log_no_cv()

    # Should NOT raise even though sample_cx/sample_cy are None.
    composed = _compose_frame(rgb, per_step_row, goal, log, frame_idx=1)
    assert composed.shape == (CANVAS_PX, CANVAS_PX * 2, 3)


# ─── 3. compose frame 0 (no plan executed yet) renders placeholder ─────

def test_compose_frame_zero_uses_placeholder_right_half():
    rgb = _solid_frame()
    per_step_row = {
        "t": 0, "reward": -1.5, "cv_success": True,
        "cx": 50.0, "cy": 60.0, "theta_deg": 30.0,
    }
    goal = {"cx": 64.0, "cy": 64.0, "theta_deg": 0.0}

    composed = _compose_frame(rgb, per_step_row, goal, None, frame_idx=0)
    assert composed.shape == (CANVAS_PX, CANVAS_PX * 2, 3)
    # Right half should be mostly white (placeholder card)
    right = composed[:, CANVAS_PX:]
    assert (right > 200).mean() > 0.5, "frame-0 right half should be mostly white"


# ─── 4. end-to-end render produces an mp4 of the right shape ───────────

def test_render_combined_video_writes_mp4(tmp_path):
    run_dir = _write_minimal_run_dir(tmp_path, n_control_steps=2)
    out = render_combined_video(run_dir, fps=8)
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
    assert (H, W) == (CANVAS_PX, CANVAS_PX * 2)


# ─── 5. end-to-end works with iteration_log lacking per-sample CV ──────

def test_render_combined_video_works_without_per_sample_cv(tmp_path):
    run_dir = _write_minimal_run_dir(tmp_path, n_control_steps=2, include_cv=False)
    out = render_combined_video(run_dir, fps=8)
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
