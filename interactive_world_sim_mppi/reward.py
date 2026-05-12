"""PushT-specific reward functions for the MPPI planner.

Two parallel surfaces with the same signature:

  - `pusht_terminal_reward(latents, rgbs, actions, config)` — bare
    function, sequential CV detection. Stateless; safe to use across
    plan() calls without lifecycle management.

  - `PushTTerminalReward(num_workers=...)` — callable class wrapping a
    persistent DetectorPool from `interactive_world_sim_cv`. Faster
    when plan() is called repeatedly. Must be closed via
    `.shutdown()` (or auto-closed by `MPPIPlanner.close()`).

This is the ONLY file in `interactive_world_sim_mppi` that imports
`interactive_world_sim_cv`. The planner core (api.py, _mppi_core.py)
knows nothing about pose detection — reward is supplied as a callable.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from interactive_world_sim_cv import DetectorPool, TPose, detect, detect_batch

from .config import Config


def _terminal_rgbs_hwc(rgbs: np.ndarray) -> np.ndarray:
    """Extract terminal-frame RGB per rollout, transpose to HWC.

    Parameters
    ----------
    rgbs : np.ndarray
        Shape (K, H, 3, H_img, W_img), uint8, channel-first per
        BatchedObservation convention.

    Returns
    -------
    out : np.ndarray
        Shape (K, H_img, W_img, 3), uint8, contiguous. Ready for
        detect() / detect_batch().
    """
    assert rgbs.ndim == 5 and rgbs.shape[2] == 3, f"unexpected rgbs shape {rgbs.shape}"
    terminal = rgbs[:, -1]  # (K, 3, H_img, W_img)
    # Transpose CHW → HWC, then make contiguous to be safe.
    hwc = np.ascontiguousarray(terminal.transpose(0, 2, 3, 1))
    return hwc


def _reward_from_pose(
    pose: Optional[TPose],
    goal_x: float,
    goal_y: float,
    goal_sin: float,
    goal_cos: float,
    config: Config,
) -> float:
    """Compute the scalar reward for one pose against a fixed goal.

    Per Phase 2 design §3 (reward function design):
      - None detection → config.detection_failure_penalty.
      - Else → -(pos_weight * pos_dist + angle_weight * angle_dist_rad)
        where angle_dist_rad = arccos(clip(sin·sin_g + cos·cos_g, -1, 1)).
    """
    if pose is None:
        return float(config.detection_failure_penalty)
    dx = pose.x - goal_x
    dy = pose.y - goal_y
    pos_dist = math.sqrt(dx * dx + dy * dy)
    cos_diff = pose.sin * goal_sin + pose.cos * goal_cos
    cos_diff = max(-1.0, min(1.0, cos_diff))
    angle_dist_rad = math.acos(cos_diff)
    return -float(
        config.pos_weight * pos_dist + config.angle_weight * angle_dist_rad
    )


def _resolve_goal_trig(config: Config) -> tuple[float, float, float, float]:
    """Read goal (x, y, sin, cos) from config.goal. Raises if unset."""
    if config.goal is None:
        raise ValueError(
            "config.goal must be set before calling the reward function. "
            "Pass goal=GoalPose(...) to MPPIPlanner or set Config(goal=...)."
        )
    g_rad = math.radians(config.goal.angle_deg)
    return config.goal.x, config.goal.y, math.sin(g_rad), math.cos(g_rad)


def pusht_terminal_reward(
    latents: torch.Tensor,
    rgbs: np.ndarray,
    actions: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """Terminal reward for PushT: detect T-pose on the last decoded frame.

    For each rollout k ∈ [0, K):
      1. Transpose rgbs[k, -1] from CHW to HWC.
      2. Call detect(rgb, mode='wm',
                     processing_resolution=config.cv_processing_resolution).
      3. None → reward[k] = config.detection_failure_penalty.
         Else → reward[k] = -(pos_weight * pos_dist
                            + angle_weight * angle_dist_rad).

    This function uses SEQUENTIAL detect() calls — stateless, ~4 s per
    K=100 call. Use PushTTerminalReward for pool-backed parallel
    detection (~0.7 s after warmup).

    Pre-conditions
    --------------
    - latents.shape == (K, H, C_latent, H_lat, W_lat), float32.
    - rgbs.shape == (K, H, 3, H_img, W_img), uint8 (channel-first).
    - actions.shape == (K, H, A); not used here.
    - config.goal is non-None.

    Returns
    -------
    rewards : torch.Tensor
        Shape (K,), float32, on `latents.device`. Higher is better.
        The MPPI softmax expects this sign convention.
    """
    goal_x, goal_y, goal_sin, goal_cos = _resolve_goal_trig(config)

    hwc = _terminal_rgbs_hwc(rgbs)  # (K, H_img, W_img, 3) uint8
    K = hwc.shape[0]

    rewards_py: list[float] = [0.0] * K
    for k in range(K):
        pose = detect(
            hwc[k],
            mode="wm",
            processing_resolution=config.cv_processing_resolution,
        )
        rewards_py[k] = _reward_from_pose(
            pose, goal_x, goal_y, goal_sin, goal_cos, config
        )
    return torch.tensor(rewards_py, dtype=torch.float32, device=latents.device)


class PushTTerminalReward:
    """Pool-backed PushTTerminalReward for fast K-parallel detection.

    Same semantics as pusht_terminal_reward; just faster because CV
    detection on K frames runs across `num_workers` processes.

    Usage with MPPIPlanner:
        reward = PushTTerminalReward(num_workers=8)
        planner = MPPIPlanner(env, cfg, reward_fn=reward)
        try:
            ...
        finally:
            planner.close()   # auto-calls reward.shutdown()

    Standalone usage:
        with PushTTerminalReward(num_workers=8) as reward:
            rewards = reward(latents, rgbs, actions, config)

    Not thread-safe. The internal DetectorPool is not thread-safe.
    """

    def __init__(self, num_workers: Optional[int] = None) -> None:
        """num_workers=None defers to config.detector_num_workers at first
        call. Explicit override for standalone usage with a known size.
        """
        self._num_workers = num_workers
        self._pool: Optional[DetectorPool] = None

    def _ensure_pool(self, config: Config) -> DetectorPool:
        if self._pool is None:
            workers = self._num_workers
            if workers is None:
                workers = config.detector_num_workers
            if workers is None or workers < 1:
                workers = 1
            self._pool = DetectorPool(num_workers=workers)
            self._pool.start()
        return self._pool

    def __call__(
        self,
        latents: torch.Tensor,
        rgbs: np.ndarray,
        actions: torch.Tensor,
        config: Config,
    ) -> torch.Tensor:
        goal_x, goal_y, goal_sin, goal_cos = _resolve_goal_trig(config)
        hwc = _terminal_rgbs_hwc(rgbs)  # (K, H_img, W_img, 3) uint8
        pool = self._ensure_pool(config)
        poses: list[Optional[TPose]] = pool.detect_batch(
            hwc,
            mode="wm",
            processing_resolution=config.cv_processing_resolution,
        )
        rewards_py = [
            _reward_from_pose(pose, goal_x, goal_y, goal_sin, goal_cos, config)
            for pose in poses
        ]
        return torch.tensor(rewards_py, dtype=torch.float32, device=latents.device)

    def shutdown(self) -> None:
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=True)
            finally:
                self._pool = None

    def __enter__(self) -> "PushTTerminalReward":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


def detect_goal_pose_from_episode(
    episode_path: str,
    t: int,
    *,
    cam_key: str = "camera_1_color",
    processing_resolution: int = 512,
):
    """Convenience: detect the pose of the T-block on a given episode frame.

    Loads obs/images/<cam_key>[t] from the HDF5, center-crops 640×480 →
    480×480, then calls `interactive_world_sim_cv.detect(..., mode='real')`.
    Returns a GoalPose constructed from the detected TPose, or raises if
    detection fails.

    This is the recommended way to compute the goal for the first
    experiment (Phase 2 design §2: "Goal pose = the pose detected on the
    t=150 frame of data/mini/pusht/val/episode_0.hdf5").
    """
    import h5py

    from .config import GoalPose

    with h5py.File(episode_path, "r") as f:
        raw = np.asarray(f[f"obs/images/{cam_key}"][t])  # (480, 640, 3) uint8 RGB
    h, w = raw.shape[:2]
    if (h, w) != (480, 640):
        raise ValueError(f"expected (480, 640) frame; got {(h, w)}")
    cropped = raw[:, 80:560]  # → (480, 480, 3)
    tpose = detect(cropped, mode="real", processing_resolution=processing_resolution)
    if tpose is None:
        raise RuntimeError(
            f"Detection failed on {episode_path} t={t} cam={cam_key}. "
            "Try a different t, or use mode='wm' if the frame is from a WM render."
        )
    return GoalPose(x=tpose.x, y=tpose.y, angle_deg=tpose.angle_deg)
