"""Expert-action extraction helper — 4 mm-approximation path.

Reads an expert episode from HDF5 and returns the normalized action
the world model was trained on. Uses the closed-form recipe
discovered in Phase 1 (see `IMPLEMENTATION_REPORT.md` and
`/tmp/test_ee_pos_shortcut.py`):

  raw_action[0:2] = (world_t_robot_base[0] @ [ee_pos[0:3], 1])[:2]
  raw_action[2:4] = (world_t_robot_base[0] @ [ee_pos[7:10], 1])[:2]

Both arms are transformed through the LEFT robot's pose because
`obs/ee_pos[7:10]` is the right EE *also* expressed in the left
robot's frame, not in the right robot's own frame. The math lives in
`projection.ee_pos_to_world_xy`; this module just applies the
model's `LinearNormalizer["action"]` on top.

The recipe skips the workspace clip
(`rob_t_eef[0,3]` to `[0.25, 1.0]`, `rob_t_eef[1,3]` to
`[-0.25, 0.25]`) that `joint_pos_to_action_primitive` performs in
the full dataset pipeline. The residual error is ~2-4 mm in
world-frame meters at the three timesteps we checked (0, 50, 100);
the ``if __name__`` block below also reports the same delta after
normalization so it can be judged against an MPPI tolerance.

ISOLATION. This module is in `helpers/` because it intentionally
reaches into `env._loaded.model.normalizer["action"]` — that
private path is the agreed exception for this one helper. The
wrapper's public API still has no notion of this helper; nobody
imports from it via the package's top-level `__init__.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch

from .projection import ee_pos_to_world_xy

if TYPE_CHECKING:
    from interactive_world_sim_env.env import WorldModelEnv


_SUPPORTED_TASKS: frozenset[str] = frozenset({"pusht_cam1"})


def expert_action_from_episode(
    env: "WorldModelEnv",
    episode_path: str,
    t: int,
) -> np.ndarray:
    """Return the expert's normalized action at step ``t``.

    Parameters
    ----------
    env: a constructed `WorldModelEnv`. Used for `env.device`, `env.task`,
        and the model's `LinearNormalizer["action"]`.
    episode_path: path to an episode HDF5 file produced by the ALOHA
        data-collection pipeline.
    t: step index within the episode.

    Returns
    -------
    `(action_dim,)` float32 in roughly `[-1, 1]` (small overshoot
    possible because the closed-form approximation skips a workspace
    clip; see module docstring).

    Raises
    ------
    NotImplementedError: when ``env.task`` is not in the verified set.
        The recipe was only checked for ``pusht_cam1``; other tasks
        need their own per-task verification before being added.
    """
    if env.task not in _SUPPORTED_TASKS:
        raise NotImplementedError(
            "expert_action_from_episode is only verified for "
            f"{sorted(_SUPPORTED_TASKS)}; got task={env.task!r}. "
            "Verify the ee_pos -> action closed-form for the new task "
            "before extending."
        )

    with h5py.File(episode_path, "r") as f:
        ee_pos = np.asarray(f["obs/ee_pos"][t], dtype=np.float64)
        base = np.asarray(f["obs/world_t_robot_base"][t], dtype=np.float64)

    raw_world_xy = ee_pos_to_world_xy(ee_pos, base)  # (4,) float32 meters

    # Documented privacy violation, scoped to this helper.
    normalizer = env._loaded.model.normalizer["action"]
    raw_t = torch.from_numpy(raw_world_xy).to(env.device).float()
    normalized_t = normalizer.normalize(raw_t)
    return normalized_t.detach().cpu().numpy().astype(np.float32)


if __name__ == "__main__":  # pragma: no cover - smoke
    import sys
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from yixuan_utilities.kinematics_helper import KinHelper  # noqa: E402

    from interactive_world_sim.utils.action_utils import (  # noqa: E402
        joint_pos_to_action_primitive,
    )
    from interactive_world_sim.utils.aloha_conts import (  # noqa: E402
        MASTER_GRIPPER_JOINT_UNNORMALIZE_FN,
        PUPPET_GRIPPER_JOINT_NORMALIZE_FN,
    )
    from interactive_world_sim_env import WorldModelEnv  # noqa: E402

    EPISODE = str(
        _REPO_ROOT / "data" / "mini" / "pusht" / "val" / "episode_0.hdf5"
    )

    env = WorldModelEnv("pusht_cam1")
    kin = KinHelper("trossen_vx300s")
    normalizer = env._loaded.model.normalizer["action"]

    # ---------- primary smoke at t=0 ----------
    action_norm = expert_action_from_episode(env, EPISODE, 0)
    print(f"t=0  normalized action: {np.round(action_norm, 5)}  dtype={action_norm.dtype}")
    print(f"     min={action_norm.min():+.5f}  max={action_norm.max():+.5f}")
    for i, a in enumerate(action_norm):
        assert -1.05 <= a <= 1.05, f"action[{i}]={a} outside [-1.05, 1.05]"
    print("     all components in [-1.05, 1.05] ✓")

    with h5py.File(EPISODE, "r") as f:
        ee0 = np.asarray(f["obs/ee_pos"][0], dtype=np.float64)
        base0 = np.asarray(f["obs/world_t_robot_base"][0], dtype=np.float64)
    raw0 = ee_pos_to_world_xy(ee0, base0)
    print(f"     underlying world XY (raw, meters): {np.round(raw0, 5)}")

    # ---------- delta vs full pipeline at t = 0, 50, 100 ----------
    print()
    print("4 mm-approx vs full-pipeline delta after normalization:")
    print(f"{'t':>4s}  {'truth_norm':>50s}  {'approx_norm':>50s}  {'max|err|':>10s}")
    for t in (0, 50, 100):
        with h5py.File(EPISODE, "r") as f:
            jp = np.asarray(f["obs/joint_pos"][t], dtype=np.float64).copy()
            base = np.asarray(f["obs/world_t_robot_base"][t], dtype=np.float64)
            ee_t = np.asarray(f["obs/ee_pos"][t], dtype=np.float64)

        # Full pipeline: gripper-joint normalize composition,
        # then joint_pos_to_action_primitive (which does FK + clipping + world transform).
        num_rob = jp.shape[0] // 7
        for r_i in range(num_rob):
            jp[r_i * 7 + 6] = MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(
                PUPPET_GRIPPER_JOINT_NORMALIZE_FN(jp[r_i * 7 + 6])
            )
        truth_raw = np.asarray(
            joint_pos_to_action_primitive(
                joint_pos=jp,
                ctrl_mode="bimanual_push",
                base_pose_in_world=base,
                kin_helper=kin,
            ),
            dtype=np.float32,
        ).reshape(-1)

        # 4 mm-approx (no FK, no clipping).
        approx_raw = ee_pos_to_world_xy(ee_t, base)

        truth_n = normalizer.normalize(
            torch.from_numpy(truth_raw).to(env.device).float()
        ).detach().cpu().numpy()
        approx_n = normalizer.normalize(
            torch.from_numpy(approx_raw).to(env.device).float()
        ).detach().cpu().numpy()
        err = float(np.max(np.abs(truth_n - approx_n)))
        print(
            f"{t:>4d}  {str(np.round(truth_n, 5)):>50s}  "
            f"{str(np.round(approx_n, 5)):>50s}  {err:>10.6f}"
        )

    env.close()
    print()
    print("OK")
