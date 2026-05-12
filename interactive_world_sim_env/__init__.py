"""Public surface of the WorldModelEnv wrapper.

MPPI and other downstream code should import only from this module.
"""

from .env import WorldModelEnv
from .obs import BatchedObservation, Observation
from .registry import TASKS, RegistryError, TaskSpec, get_task_spec
from .state import EnvState

__all__ = [
    "WorldModelEnv",
    "Observation",
    "BatchedObservation",
    "EnvState",
    "TaskSpec",
    "RegistryError",
    "TASKS",
    "get_task_spec",
]
