"""Step 2 tests for rl.mppi.reward.state_reward.

Four tests:
  1. self-reward on the decoded goal frame is ~0 (minor HSV noise is fine).
  2. a frame with the T far from the goal produces a strongly negative reward.
  3. a synthetic 180-flipped T at the goal position produces the expected
     angle penalty (|1 - cos(180)| = 2).
  4. a blank black image (no T visible) triggers the CV-fail branch and
     returns exactly -large_penalty.
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
from rl.mppi.reward import DEFAULT_LARGE_PENALTY, IMAGE_DIAGONAL_128, state_reward

CKPT_PATH = Path("outputs/pusht_cam1/checkpoints/best.ckpt")
Z_GOAL_PATH = Path("tests/goal_selection/z_goal.pt")
STATE_GOAL_PATH = Path("tests/goal_selection/state_goal.pt")
RES = 128
# RGB value (pink) that lands inside the REAL HSV mask at ~H=170.
PINK_RGB = (220, 80, 145)


def _have_gpu_and_ckpt() -> bool:
    return torch.cuda.is_available() and CKPT_PATH.exists()


@pytest.fixture(scope="module")
def state_goal():
    g = torch.load(STATE_GOAL_PATH, map_location="cpu")
    return {
        "cx": g["cx"],
        "cy": g["cy"],
        "sin_theta": g["sin_theta"],
        "cos_theta": g["cos_theta"],
    }


@pytest.fixture(scope="module")
def labeler():
    return CVLabeler(preset="REAL", resolution=RES)


@pytest.fixture(scope="module")
def goal_rgb():
    """Decoded RGB of z_goal via the pretrained WM. This is exactly what
    MPPI rewards will see at runtime (a WM-decoded latent, not a raw frame)."""
    if not _have_gpu_and_ckpt():
        pytest.skip("CUDA + WM checkpoint required for decoded goal frame")
    from rl.models.world_model import DifferentiableDynamics
    wm = DifferentiableDynamics(str(CKPT_PATH), device="cuda:0")
    z = torch.load(Z_GOAL_PATH, map_location="cuda:0")
    with torch.no_grad():
        rgb = wm.decode(z, resolution=RES)  # (1, 3, 128, 128) in [0,1]
    arr = (rgb.clamp(0, 1).cpu().numpy()[0] * 255).astype(np.uint8)
    return arr.transpose(1, 2, 0)  # (128, 128, 3) uint8 RGB


def _synthetic_t_image(cx: float, cy: float, theta_deg: float) -> np.ndarray:
    """Draw a filled pink T-block on a black 128x128 RGB canvas.

    The T_BLOCK_SHAPE constant is defined at 512-px canvas scale, so we
    rescale to RES=128 before rotating/translating. The centroid offset
    (T_BLOCK_FILLED_CENTROID) places the block so its geometric center
    lands exactly at (cx, cy) after translation.
    """
    scale = RES / 512.0
    pts = T_BLOCK_SHAPE * scale
    centroid = T_BLOCK_FILLED_CENTROID * scale

    # center on origin
    pts = pts - centroid
    # rotate (image-space: y grows downward, so use the standard matrix
    # that matches the labeler's own convention)
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = pts @ R.T
    # translate
    pts = pts + np.array([cx, cy])

    img = np.zeros((RES, RES, 3), dtype=np.uint8)
    cv2.fillPoly(img, [pts.astype(np.int32)], color=PINK_RGB)
    return img


# -----------------------------------------------------------------------------


def test_reward_goal_frame_against_self(goal_rgb, state_goal, labeler):
    reward, label = state_reward(goal_rgb, state_goal, labeler=labeler)
    assert label.success, "CV should succeed on the decoded goal frame"
    # Small non-zero reward is expected because:
    # 1. WM decode → CV label does not reproduce state_goal exactly (the
    #    stored state_goal was measured from a different decode, and the
    #    decoder is stochastic).
    # 2. HSV mask edges jitter by a pixel or two per frame.
    # Tolerance: within 0.1 of zero combined; position + angle terms both tiny.
    assert reward > -0.1, f"self-reward {reward:.4f} too negative"
    print(f"[self-reward] reward={reward:.4f} (cx={label.cx:.2f}, cy={label.cy:.2f}, "
          f"theta_deg={label.theta_deg:.2f})")


def test_reward_far_from_goal_is_negative(state_goal, labeler):
    """Synthetic T placed far from the goal. Goal is (~55, ~62). Putting
    the T at (100, 30) is ~50 px away in each axis -> ~70 px, ~38% of the
    diagonal -> position term ≈ -0.39 alone, before any angle contribution."""
    far_img = _synthetic_t_image(cx=100.0, cy=30.0, theta_deg=0.0)
    reward, label = state_reward(far_img, state_goal, labeler=labeler)
    assert label.success, "CV should detect the synthetic pink T"
    assert reward < -0.3, f"far-from-goal reward {reward:.4f} not negative enough"
    print(f"[far] reward={reward:.4f}, cx={label.cx:.2f}, cy={label.cy:.2f}, "
          f"theta_deg={label.theta_deg:.2f}")


def test_reward_synthetic_180_flip(state_goal, labeler):
    """Synthetic T at the goal position but rotated by 180 from goal.

    The angle term is |1 - cos(delta_theta)|. At delta = 180°,
    cos(180°) = -1, so the angle term = |1 - (-1)| = 2. Position term
    ≈ 0 (we put the T *at* the goal). Total reward ≈ -2.
    """
    cx_goal = float(state_goal["cx"])
    cy_goal = float(state_goal["cy"])
    # goal theta_deg ≈ 0.75 per state_goal metadata; add 180 to flip.
    flipped = _synthetic_t_image(cx=cx_goal, cy=cy_goal, theta_deg=180.75)
    reward, label = state_reward(flipped, state_goal, labeler=labeler)
    assert label.success, "CV should detect the flipped pink T"
    # Angle term dominates; tolerance for CV sub-pixel position error.
    assert reward < -1.5, f"180-flip reward {reward:.4f} should be near -2"
    assert reward > -2.2, f"180-flip reward {reward:.4f} too negative"
    print(
        f"[180-flip] reward={reward:.4f}, cx={label.cx:.2f}, cy={label.cy:.2f}, "
        f"theta_deg={label.theta_deg:.2f}"
    )


def test_reward_cv_fail_returns_penalty(state_goal, labeler):
    """Blank black image — no pink T anywhere — must trigger CV fail."""
    blank = np.zeros((RES, RES, 3), dtype=np.uint8)
    reward, label = state_reward(blank, state_goal, labeler=labeler)
    assert not label.success
    assert reward == pytest.approx(-DEFAULT_LARGE_PENALTY)
    print(f"[cv-fail] reward={reward:.4f} (expected -{DEFAULT_LARGE_PENALTY})")


def test_image_diagonal_constant():
    """Defensive: if someone edits IMAGE_DIAGONAL_128, don't silently rescale
    all rewards. Lock the value to match sqrt(128^2 + 128^2)."""
    assert abs(IMAGE_DIAGONAL_128 - 181.019335984) < 1e-6
