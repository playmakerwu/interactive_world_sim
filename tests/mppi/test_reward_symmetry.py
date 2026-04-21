"""Step 5 follow-up tests for symmetry-aware reward.

Verifies the `symmetry_aware=True` branch of state_reward / batched_state_reward.
Original (non-sym) semantics stay verified by test_reward.py — these tests do
not overlap.

The synthetic-T helper is duplicated from test_reward.py rather than imported
to keep the two test modules independent (per §process rules: "Don't modify
tests in test_reward.py").
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from rl.labeling.cv_labeler import (
    CVLabeler,
    T_BLOCK_FILLED_CENTROID,
    T_BLOCK_SHAPE,
)
from rl.mppi.reward import DEFAULT_LARGE_PENALTY, state_reward

STATE_GOAL_PATH = Path("tests/goal_selection/state_goal.pt")
RES = 128
PINK_RGB = (220, 80, 145)


@pytest.fixture(scope="module")
def state_goal():
    g = torch.load(STATE_GOAL_PATH, map_location="cpu")
    return {
        "cx": g["cx"], "cy": g["cy"],
        "sin_theta": g["sin_theta"], "cos_theta": g["cos_theta"],
    }


@pytest.fixture(scope="module")
def labeler():
    return CVLabeler(preset="REAL", resolution=RES)


def _synth_t_image(cx: float, cy: float, theta_deg: float) -> np.ndarray:
    scale = RES / 512.0
    pts = T_BLOCK_SHAPE * scale
    centroid = T_BLOCK_FILLED_CENTROID * scale
    pts = pts - centroid
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = pts @ R.T
    pts = pts + np.array([cx, cy])
    img = np.zeros((RES, RES, 3), dtype=np.uint8)
    cv2.fillPoly(img, [pts.astype(np.int32)], color=PINK_RGB)
    return img


# -----------------------------------------------------------------------------
# Direct angle-term checks
# -----------------------------------------------------------------------------

def test_sym_reward_at_goal_angle_is_near_zero(state_goal, labeler):
    """θ = θ_goal => angle term = 0 under both reward variants."""
    cx_goal = float(state_goal["cx"])
    cy_goal = float(state_goal["cy"])
    theta_goal_deg = np.rad2deg(
        np.arctan2(state_goal["sin_theta"], state_goal["cos_theta"])
    )
    img = _synth_t_image(cx=cx_goal, cy=cy_goal, theta_deg=float(theta_goal_deg))
    r_sym, lbl = state_reward(img, state_goal, labeler=labeler, symmetry_aware=True)
    assert lbl.success
    assert abs(r_sym) < 0.05, f"sym reward at goal should be ~0, got {r_sym:.4f}"


def test_sym_reward_at_180_flip_is_near_zero(state_goal, labeler):
    """The whole point: θ = θ_goal + 180° must also score near zero under
    symmetry-aware reward. (Under the original variant it scored ≈ -2.)"""
    cx_goal = float(state_goal["cx"])
    cy_goal = float(state_goal["cy"])
    theta_goal_deg = np.rad2deg(
        np.arctan2(state_goal["sin_theta"], state_goal["cos_theta"])
    )
    img = _synth_t_image(
        cx=cx_goal, cy=cy_goal, theta_deg=float(theta_goal_deg) + 180.0
    )
    r_sym, lbl = state_reward(img, state_goal, labeler=labeler, symmetry_aware=True)
    assert lbl.success
    assert abs(r_sym) < 0.05, (
        f"sym reward at θ+180 should be ~0, got {r_sym:.4f}. "
        "This is the whole reason symmetry-aware exists."
    )

    # Cross-check: the *original* reward on the same frame should be ≈ -2.
    r_orig, _ = state_reward(img, state_goal, labeler=labeler, symmetry_aware=False)
    assert r_orig < -1.5, (
        f"original reward at θ+180 should be ≈ -2, got {r_orig:.4f} — "
        "if this trips, test_reward.py::test_reward_synthetic_180_flip is stale"
    )


def test_sym_reward_at_90_deg_is_near_minus_one(state_goal, labeler):
    """θ = θ_goal + 90° => cos(Δθ) = 0 => |cos| = 0 => angle term = -1 (both
    variants coincide at perpendicular). Position term ≈ 0 since we place the
    T at the goal. Total reward ≈ -1."""
    cx_goal = float(state_goal["cx"])
    cy_goal = float(state_goal["cy"])
    theta_goal_deg = np.rad2deg(
        np.arctan2(state_goal["sin_theta"], state_goal["cos_theta"])
    )
    img = _synth_t_image(
        cx=cx_goal, cy=cy_goal, theta_deg=float(theta_goal_deg) + 90.0
    )
    r_sym, lbl = state_reward(img, state_goal, labeler=labeler, symmetry_aware=True)
    assert lbl.success
    assert -1.1 < r_sym < -0.85, (
        f"sym reward at θ+90 should be ≈ -1, got {r_sym:.4f}"
    )


# -----------------------------------------------------------------------------
# Cross-comparisons
# -----------------------------------------------------------------------------

def test_sym_equals_original_when_aligned(state_goal, labeler):
    """When θ = θ_goal both reward variants should return identical values."""
    cx_goal = float(state_goal["cx"])
    cy_goal = float(state_goal["cy"])
    theta_goal_deg = np.rad2deg(
        np.arctan2(state_goal["sin_theta"], state_goal["cos_theta"])
    )
    img = _synth_t_image(cx=cx_goal, cy=cy_goal, theta_deg=float(theta_goal_deg))
    r_sym, _ = state_reward(img, state_goal, labeler=labeler, symmetry_aware=True)
    r_orig, _ = state_reward(img, state_goal, labeler=labeler, symmetry_aware=False)
    # Not exactly equal because pos_term is identical but the angle branches
    # differ in formula structure — should be within numerical noise.
    assert abs(r_sym - r_orig) < 1e-6, (
        f"sym vs orig at goal: {r_sym:.6f} vs {r_orig:.6f}"
    )


def test_sym_is_strictly_greater_at_180_flip(state_goal, labeler):
    """Sym >= Orig always (since |cos| <= |1 - cos| is not always true,
    the cleanest formulation is: both variants coincide when cos>=0;
    when cos<0, sym is strictly larger). Test the flip case."""
    cx_goal = float(state_goal["cx"])
    cy_goal = float(state_goal["cy"])
    theta_goal_deg = np.rad2deg(
        np.arctan2(state_goal["sin_theta"], state_goal["cos_theta"])
    )
    img = _synth_t_image(
        cx=cx_goal, cy=cy_goal, theta_deg=float(theta_goal_deg) + 180.0
    )
    r_sym, _ = state_reward(img, state_goal, labeler=labeler, symmetry_aware=True)
    r_orig, _ = state_reward(img, state_goal, labeler=labeler, symmetry_aware=False)
    # At 180-flip: sym should be ≈ 0, orig should be ≈ -2
    assert r_sym > r_orig + 1.0, (
        f"sym {r_sym:.4f} should be >> orig {r_orig:.4f} at 180° flip"
    )


def test_sym_cv_fail_returns_large_penalty(state_goal, labeler):
    """CV-fail path is independent of the symmetry flag — both still hit
    the -large_penalty branch at the top of state_reward."""
    blank = np.zeros((RES, RES, 3), dtype=np.uint8)
    r_sym, lbl = state_reward(
        blank, state_goal, labeler=labeler, symmetry_aware=True
    )
    assert not lbl.success
    assert r_sym == pytest.approx(-DEFAULT_LARGE_PENALTY)
