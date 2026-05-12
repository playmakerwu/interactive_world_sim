"""interactive_world_sim_mppi — MPPI planner over WorldModelEnv.

Public API:
    Config                    — frozen dataclass with all hyperparameters
    GoalPose                  — target pose in detector pixel space
    MPPIPlanner               — the planner class
    pusht_terminal_reward     — sequential PushT reward (stateless)
    PushTTerminalReward       — pool-backed PushT reward (stateful)
    detect_goal_pose_from_episode — helper to detect a GoalPose from HDF5

The algorithmic core lives in _mppi_core.py and _splines.py (verbatim
copies from diffusion-forcing SHA 180a2639a01c593c1a73275abe42b3acf4afc162;
see _mppi_core.py header). Production code does not import from
diffusion-forcing at runtime.
"""

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
    "PushTTerminalReward",
    "detect_goal_pose_from_episode",
    "pusht_terminal_reward",
]
