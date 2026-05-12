"""Visual reference: decode 21 rollout frames into a single grid PNG.

Constructs a WorldModelEnv("pusht_cam1"), resets from the default episode
(data/mini/pusht/val/episode_0.hdf5, frame 0), rolls out 10 steps with
zero action then (after a fresh reset) 10 steps with a constant
action[0] = 0.3, decodes every frame, and stitches them into a 2-row
grid PNG so the two rollouts can be compared at a glance.

Run from the repo root, with the iws conda env active:

    python interactive_world_sim_env/scripts/show_camera.py
    # or
    python interactive_world_sim_env/scripts/show_camera.py /tmp/out.png
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling `interactive_world_sim_env` package importable when this
# script is invoked directly (e.g. `python interactive_world_sim_env/scripts/show_camera.py`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import imageio.v3 as iio
import numpy as np

from interactive_world_sim_env import WorldModelEnv

DEFAULT_OUT = "/tmp/pusht_cam1_preview.png"
ZERO_STEPS = 10
ACTION_STEPS = 10
NONTRIVIAL_ACTION_VAL = 0.3  # applied to action[0]


def _rollout(env: WorldModelEnv, action: np.ndarray, n_steps: int) -> list[np.ndarray]:
    """Reset env, return [init_rgb, *n_steps decoded frames after applying `action` each step]."""
    env.reset()
    frames = [env.render()]
    for _ in range(n_steps):
        env.step(action)
        frames.append(env.render())
    return frames


def main(out_path: str = DEFAULT_OUT) -> tuple[str, tuple[int, int, int]]:
    env = WorldModelEnv("pusht_cam1")
    print(
        f"constructed env: task={env.task} device={env.device} "
        f"action_space={env.action_space.shape}"
    )

    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    zero_frames = _rollout(env, zero_action, ZERO_STEPS)
    print(f"zero-action rollout: {len(zero_frames)} frames, each {zero_frames[0].shape}")

    nontrivial = np.zeros(env.action_space.shape, dtype=np.float32)
    nontrivial[0] = NONTRIVIAL_ACTION_VAL
    action_frames = _rollout(env, nontrivial, ACTION_STEPS)
    print(
        f"non-trivial-action rollout (action[0]={NONTRIVIAL_ACTION_VAL}): "
        f"{len(action_frames)} frames"
    )

    # Both rows share an initial frame produced from the same dataset
    # frame, but each is independently decoded (the decoder is stochastic),
    # so the leftmost cells of the two rows may differ slightly. That is
    # informative, not a bug.
    row0 = np.concatenate(zero_frames, axis=1)
    row1 = np.concatenate(action_frames, axis=1)
    grid = np.concatenate([row0, row1], axis=0)

    iio.imwrite(out_path, grid)
    env.close()

    print(f"wrote {out_path}: shape={grid.shape} dtype={grid.dtype}")
    return out_path, grid.shape


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    main(out)
