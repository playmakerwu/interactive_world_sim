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

# CV is CPU-bound and takes ~20-25 ms per 128x128 frame (Step 2 measurement).
# For N=128 that is ~3 s sequentially. We keep the Python-loop form for v0 and
# revisit if it dominates per-step wall time.

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


def batched_state_reward(
    rgb_batch: np.ndarray | torch.Tensor,
    state_goal: Mapping[str, float],
    *,
    labeler: CVLabeler | None = None,
    image_diagonal: float = IMAGE_DIAGONAL_128,
    large_penalty: float = DEFAULT_LARGE_PENALTY,
) -> tuple[np.ndarray, list[CVLabelResult]]:
    """Reward for a batch of RGB frames against a state goal.

    Args:
        rgb_batch: one of
            - (N, H, W, 3) uint8 numpy array,
            - (N, H, W, 3) float32 numpy array in [0, 1],
            - (N, 3, H, W) torch tensor in [0, 1] (what wm.decode returns).
        state_goal: see `state_reward`.
        labeler: CVLabeler. Pass a pre-built instance to amortize
            template-contour setup across the batch.
        image_diagonal: divisor for the position term.
        large_penalty: returned for frames where CV fails.

    Returns:
        rewards: (N,) np.ndarray of float64, one reward per frame.
        labels:  list[CVLabelResult], aligned with rewards.
    """
    if isinstance(rgb_batch, torch.Tensor):
        # decoder output: (N, 3, H, W) float [0, 1] -> (N, H, W, 3) uint8
        arr = rgb_batch.detach().cpu().float().permute(0, 2, 3, 1).numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = rgb_batch
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(
            f"rgb_batch must be (N, H, W, 3), got shape {arr.shape}"
        )

    if labeler is None:
        labeler = CVLabeler(preset="REAL", resolution=arr.shape[1])

    N = arr.shape[0]
    rewards = np.empty(N, dtype=np.float64)
    labels: list[CVLabelResult] = []
    for i in range(N):
        r, label = state_reward(
            arr[i],
            state_goal,
            labeler=labeler,
            image_diagonal=image_diagonal,
            large_penalty=large_penalty,
        )
        rewards[i] = r
        labels.append(label)
    return rewards, labels


def score_latents(
    z_batch: torch.Tensor,
    state_goal: Mapping[str, float],
    wm,
    *,
    labeler: CVLabeler | None = None,
    resolution: int = 128,
    image_diagonal: float = IMAGE_DIAGONAL_128,
    large_penalty: float = DEFAULT_LARGE_PENALTY,
) -> tuple[np.ndarray, list[CVLabelResult]]:
    """Decode a batch of latents through the WM, then score each RGB.

    Args:
        z_batch: (N, C, H_lat, W_lat) latents to score.
        state_goal, labeler, image_diagonal, large_penalty: see
            `batched_state_reward`.
        wm: any object exposing `.decode(z: (N, C, H, W), resolution=int)
            -> (N, 3, res, res) in [0, 1]`.
        resolution: output resolution for decoded RGB (must match
            `labeler.resolution`, which defaults to 128).

    Returns:
        (rewards, labels) from batched_state_reward.
    """
    if labeler is None:
        labeler = CVLabeler(preset="REAL", resolution=resolution)
    if labeler.resolution != resolution:
        raise ValueError(
            f"resolution mismatch: labeler={labeler.resolution}, arg={resolution}"
        )
    with torch.no_grad():
        rgb = wm.decode(z_batch, resolution=resolution)
    return batched_state_reward(
        rgb,
        state_goal,
        labeler=labeler,
        image_diagonal=image_diagonal,
        large_penalty=large_penalty,
    )
