"""Task registry for WorldModelEnv.

A flat dict mapping task names to TaskSpec entries. The registry is a hint
about what each shipped checkpoint should look like; the ground truth lives
in the checkpoint's sibling .hydra/config.yaml. The loader cross-checks
action_dim, obs_keys, and resolution against the config and raises
RegistryError on mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass


class RegistryError(RuntimeError):
    """Raised when a task is unknown or its registry entry disagrees with the checkpoint config."""


@dataclass(frozen=True)
class TaskSpec:
    ckpt_path: str
    obs_keys: tuple[str, ...]
    action_dim: int
    resolution: int
    default_episode_path: str
    ctrl_mode: str


TASKS: dict[str, TaskSpec] = {
    "pusht_cam1": TaskSpec(
        ckpt_path="outputs/pusht_cam1/checkpoints/best.ckpt",
        obs_keys=("camera_1_color",),
        action_dim=4,
        resolution=128,
        default_episode_path="data/mini/pusht/val/episode_0.hdf5",
        ctrl_mode="bimanual_push",
    ),
    "single_grasp_cam0": TaskSpec(
        ckpt_path="outputs/single_grasp_cam0/checkpoints/best.ckpt",
        obs_keys=("camera_0_color",),
        action_dim=4,
        resolution=128,
        default_episode_path="data/mini/single_grasp/val/episode_0.hdf5",
        ctrl_mode="single_grasp",
    ),
    "single_grasp_cam1": TaskSpec(
        ckpt_path="outputs/single_grasp_cam1/checkpoints/best.ckpt",
        obs_keys=("camera_1_color",),
        action_dim=4,
        resolution=128,
        default_episode_path="data/mini/single_grasp/val/episode_0.hdf5",
        ctrl_mode="single_grasp",
    ),
    "bimanual_sweep_cam0": TaskSpec(
        ckpt_path="outputs/bimanual_sweep_cam0/checkpoints/best.ckpt",
        obs_keys=("camera_0_color",),
        action_dim=4,
        resolution=128,
        default_episode_path="data/mini/bimanual_sweep/val/episode_0.hdf5",
        ctrl_mode="bimanual_sweep",
    ),
    "bimanual_sweep_cam1": TaskSpec(
        ckpt_path="outputs/bimanual_sweep_cam1/checkpoints/best.ckpt",
        obs_keys=("camera_1_color",),
        action_dim=4,
        resolution=128,
        default_episode_path="data/mini/bimanual_sweep/val/episode_0.hdf5",
        ctrl_mode="bimanual_sweep",
    ),
    "bimanual_rope_cam0": TaskSpec(
        ckpt_path="outputs/bimanual_rope_cam0/checkpoints/best.ckpt",
        obs_keys=("camera_0_color",),
        action_dim=8,
        resolution=128,
        default_episode_path="data/mini/bimanual_rope/val/episode_0.hdf5",
        ctrl_mode="bimanual_rope",
    ),
    "bimanual_rope_cam1": TaskSpec(
        ckpt_path="outputs/bimanual_rope_cam1/checkpoints/best.ckpt",
        obs_keys=("camera_1_color",),
        action_dim=8,
        resolution=128,
        default_episode_path="data/mini/bimanual_rope/val/episode_0.hdf5",
        ctrl_mode="bimanual_rope",
    ),
}


def get_task_spec(task: str) -> TaskSpec:
    """Look up a task; raise RegistryError if unknown."""
    if task not in TASKS:
        raise RegistryError(
            f"Unknown task {task!r}. Known tasks: {sorted(TASKS.keys())}"
        )
    return TASKS[task]
