"""interactive_world_sim_mppi — MPPI planner over WorldModelEnv.

Public API:
    Config                          — frozen dataclass with all hyperparameters
    GoalPose                        — legacy target-pose dataclass
    MPPIPlanner                     — high-level planner over snapshots
    Planner                         — production-faithful low-level planner
    detect_goal_state_from_episode  — production-style dict goal_state from HDF5

Legacy backwards-compat surface:
    pusht_terminal_reward, PushTTerminalReward, detect_goal_pose_from_episode
    — used by the older smoke scripts. The new MPPIPlanner does not consume
    them; the production-faithful reward is computed inside ``_planner.py``.

The algorithmic core lives in ``_planner.py`` (production-faithful port).
The verbatim diffusion-forcing source remains in ``_mppi_core.py`` and
``_splines.py`` for the bitwise-equivalence smoke test; it is not active
in the public API.
"""

from ._planner import Planner, detect_goal_state_from_episode
from .api import MPPIPlanner
from .config import Config, GoalPose
from .reward import (
    PushTTerminalReward,
    detect_goal_pose_from_episode,
    pusht_terminal_reward,
)

__all__ = [
    "Config",
    "GoalPose",
    "MPPIPlanner",
    "Planner",
    "PushTTerminalReward",
    "detect_goal_pose_from_episode",
    "detect_goal_state_from_episode",
    "pusht_terminal_reward",
]
