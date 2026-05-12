"""Expert action extraction from episode HDF5, ported from yiru branch.

Returns the normalized action the world model was trained on. Uses the
4mm closed-form approximation discovered on yiru:

    raw_action[0:2] = (world_t_robot_base[0] @ [ee_pos[0:3], 1])[:2]
    raw_action[2:4] = (world_t_robot_base[0] @ [ee_pos[7:10], 1])[:2]

Both arms transformed through the LEFT robot's pose because
``obs/ee_pos[7:10]`` is the right EE expressed in the left robot's
frame, not the right robot's own. The recipe skips the workspace clip
performed in ``joint_pos_to_action_primitive``; residual error is
~2-4 mm in world-frame meters, ~0.018 in normalized space.

Ported from:
  yiru:interactive_world_sim_env/helpers/expert_action.py
  yiru:interactive_world_sim_env/helpers/projection.py (ee_pos_to_world_xy)

Yiru source SHA: 90d6f025d43e8acc86497424f5c5831b2207b912
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch

if TYPE_CHECKING:
    from env.pusht_wm_env import PushTWMEnv


def ee_pos_to_world_xy(
    ee_pos: np.ndarray,
    world_t_robot_base: np.ndarray,
) -> np.ndarray:
    """Compute 4D action (left_xy, right_xy) in world frame from ee_pos.

    Args:
        ee_pos: (14,) float64 — left ee_pos at [0:7], right ee_pos at [7:14].
        world_t_robot_base: (2, 4, 4) float64 — base poses for the two
            robots. Both arms use the LEFT robot's frame (index 0)
            because ``obs/ee_pos[7:10]`` is the right EE also in the
            LEFT robot's frame.

    Returns:
        (4,) float32 — [left_x, left_y, right_x, right_y] in world frame.
    """
    base = world_t_robot_base[0]  # (4, 4)
    left_local = np.concatenate([ee_pos[0:3], [1.0]])
    right_local = np.concatenate([ee_pos[7:10], [1.0]])
    left_world = base @ left_local   # (4,)
    right_world = base @ right_local  # (4,)
    return np.array(
        [left_world[0], left_world[1], right_world[0], right_world[1]],
        dtype=np.float32,
    )


def expert_action_from_episode(
    env: "PushTWMEnv",
    episode_path: str,
    t: int,
) -> np.ndarray:
    """Return the expert's normalized action at step t from an episode HDF5.

    Args:
        env: a constructed PushTWMEnv. Used for env.device and the
            normalizer at env._wm.normalizer["action"].
        episode_path: path to an episode HDF5 file.
        t: step index within the episode.

    Returns:
        (4,) float32 normalized action in roughly [-1, 1] (small
        overshoot possible because the closed-form approximation
        skips a workspace clip; see module docstring).
    """
    with h5py.File(episode_path, "r") as f:
        ee_pos = np.asarray(f["obs/ee_pos"][t], dtype=np.float64)
        base = np.asarray(f["obs/world_t_robot_base"][t], dtype=np.float64)

    raw_world_xy = ee_pos_to_world_xy(ee_pos, base)  # (4,) float32 meters

    # Access path: PushTWMEnv exposes self._wm (DifferentiableDynamics),
    # which holds the normalizer at self.normalizer (a LinearNormalizer
    # mapping). See rl/models/world_model.py:93.
    normalizer = env._wm.normalizer["action"]
    raw_t = torch.from_numpy(raw_world_xy).to(env.device).float()
    normalized_t = normalizer.normalize(raw_t)
    return normalized_t.detach().cpu().numpy().astype(np.float32)
