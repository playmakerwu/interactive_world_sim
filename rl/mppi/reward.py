"""CV-based state reward for MPPI scoring.

Scores a single RGB image against a T-block state goal using the classical CV
labeler (rl.labeling.cv_labeler). Reward is:

    R = -||pos - pos_goal|| / image_diagonal  -  |1 - cos(theta - theta_goal)|

with a large-magnitude penalty when CV fails, so off-distribution rollouts are
pushed away without being confused with merely "far from goal".

The angle term uses the sin-cos dot product identity:
    sin(a)*sin(b) + cos(a)*cos(b) = cos(a - b)
so |1 - cos(delta_theta)| is computed directly from the label's sin_theta /
cos_theta — we never call atan2.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch

from rl.labeling.cv_labeler import CVLabeler, CVLabelResult

IMAGE_DIAGONAL_128 = math.sqrt(128.0**2 + 128.0**2)  # ≈ 181.02
DEFAULT_LARGE_PENALTY = 10.0


def state_reward(
    rgb: np.ndarray | torch.Tensor,
    state_goal: Mapping[str, float],
    *,
    labeler: CVLabeler | None = None,
    image_diagonal: float = IMAGE_DIAGONAL_128,
    large_penalty: float = DEFAULT_LARGE_PENALTY,
) -> tuple[float, CVLabelResult]:
    """Reward one RGB against a state goal.

    Args:
        rgb: (H, W, 3) uint8 RGB, float32 [0,1], or torch.Tensor. Must
            match the labeler's resolution (128x128 by default).
        state_goal: mapping with keys `cx`, `cy`, `sin_theta`, `cos_theta`
            (scalars). Angles in sin/cos form — no atan2 round-trip.
        labeler: a reusable CVLabeler. Callers planning to score many
            frames should build one and pass it in — construction
            pre-computes the T-block template contour.
        image_diagonal: divisor for the position term. Defaults to the
            128x128 canvas diagonal.
        large_penalty: returned when CV detection fails. Should be much
            more negative than any typical reward so failures visibly
            dominate the softmax.

    Returns:
        (reward: float, label: CVLabelResult). The label is exposed so
        callers can log CV fail counts, residuals, etc.
    """
    if labeler is None:
        resolution = rgb.shape[0]
        labeler = CVLabeler(preset="REAL", resolution=resolution)

    label = labeler.label(rgb)
    if not label.success:
        return -large_penalty, label

    dx = label.cx - float(state_goal["cx"])
    dy = label.cy - float(state_goal["cy"])
    pos_dist = math.sqrt(dx * dx + dy * dy)
    pos_term = -pos_dist / image_diagonal

    cos_delta = (
        label.sin_theta * float(state_goal["sin_theta"])
        + label.cos_theta * float(state_goal["cos_theta"])
    )
    # clamp to [-1, 1] before the subtract to guard against tiny FP > 1.
    cos_delta = max(-1.0, min(1.0, cos_delta))
    ang_term = -abs(1.0 - cos_delta)

    reward = pos_term + ang_term
    return reward, label
