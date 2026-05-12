"""Pure-function projection helpers for verifying the expert-action recipe.

These functions live in the isolated `helpers/` subpackage. They do not
depend on the WorldModelEnv, the model, or torch — only numpy + h5py.

All three functions assume the dataset's HDF5 conventions inferred from
`data/mini/pusht/val/episode_0.hdf5`:

* `obs/ee_pos[t]` is a 14-dim float32 vector with the left EE's
  position in the LEFT robot's frame at indices 0..2, and the right
  EE's position ALSO in the LEFT robot's frame at indices 7..9.
* `obs/world_t_robot_base[t]` is `(2, 4, 4)`; entry 0 is the left
  robot's pose in world frame, entry 1 is the right robot's.
* `obs/images/{cam_key}_intrinsics[t]` is the standard `(3, 3)` pinhole
  K matrix for the corresponding RGB stream.
* `obs/images/{cam_key}_extrinsics[t]` is `world_t_cam` — i.e., camera
  pose in the world frame, OpenCV camera convention (X right, Y down,
  Z forward into the scene). Inverting it gives the world→cam
  transform required for projection.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np


def ee_pos_to_world_xy(
    ee_pos: np.ndarray,
    world_t_robot_base: np.ndarray,
) -> np.ndarray:
    """Approximate the dataset's expert 4-dim action target from `obs/ee_pos`.

    Per the Phase 1 finding: apply the LEFT robot's pose to both arms'
    positions stored in `ee_pos` (because both arms are encoded in the
    left robot's frame in the HDF5), then drop Z. Reproduces the
    dataset's `joint_pos_to_action_primitive(ctrl_mode="bimanual_push")`
    output to within ~4 mm — the residual is the workspace clipping
    that this approximation skips.

    Parameters
    ----------
    ee_pos: shape `(14,)`, float — usually the value at `obs/ee_pos[t]`.
    world_t_robot_base: shape `(2, 4, 4)`, float — usually the value at
        `obs/world_t_robot_base[t]`. Only entry 0 is used.

    Returns
    -------
    shape `(4,)`, float32: `[left_x, left_y, right_x, right_y]` in
    world frame, meters.
    """
    if ee_pos.shape != (14,):
        raise ValueError(f"ee_pos must have shape (14,); got {ee_pos.shape}")
    if world_t_robot_base.shape != (2, 4, 4):
        raise ValueError(
            f"world_t_robot_base must have shape (2, 4, 4); got {world_t_robot_base.shape}"
        )

    w_t_r0 = world_t_robot_base[0].astype(np.float64)
    left_h = np.array([ee_pos[0], ee_pos[1], ee_pos[2], 1.0], dtype=np.float64)
    right_h = np.array([ee_pos[7], ee_pos[8], ee_pos[9], 1.0], dtype=np.float64)
    left_world = (w_t_r0 @ left_h)[:3]
    right_world = (w_t_r0 @ right_h)[:3]
    return np.array(
        [left_world[0], left_world[1], right_world[0], right_world[1]],
        dtype=np.float32,
    )


def world_xy_to_pixel(
    world_xy_per_arm: np.ndarray,
    cam_intrinsics: np.ndarray,
    cam_extrinsics: np.ndarray,
) -> np.ndarray:
    """Project two 3D world points to pixel coordinates via a pinhole camera.

    Despite the parameter name, this function consumes the full 3D
    position (X, Y, Z) for each arm — the "world_xy" in the name refers
    to which subset of the action vector we ultimately care about. Z is
    required to project.

    Parameters
    ----------
    world_xy_per_arm: shape `(2, 3)`, float — `[left_xyz, right_xyz]`
        in world frame, meters.
    cam_intrinsics: shape `(3, 3)`, float — standard pinhole K matrix.
    cam_extrinsics: shape `(4, 4)`, float — `world_t_cam` in OpenCV
        camera convention.

    Returns
    -------
    shape `(2, 2)`, float64: `[[u_left, v_left], [u_right, v_right]]`.
    Pixel coordinates are returned as floats; the caller rounds to int
    for drawing.
    """
    if world_xy_per_arm.shape != (2, 3):
        raise ValueError(
            f"world_xy_per_arm must have shape (2, 3); got {world_xy_per_arm.shape}"
        )
    if cam_intrinsics.shape != (3, 3):
        raise ValueError(
            f"cam_intrinsics must have shape (3, 3); got {cam_intrinsics.shape}"
        )
    if cam_extrinsics.shape != (4, 4):
        raise ValueError(
            f"cam_extrinsics must have shape (4, 4); got {cam_extrinsics.shape}"
        )

    cam_t_world = np.linalg.inv(cam_extrinsics.astype(np.float64))
    K = cam_intrinsics.astype(np.float64)

    pixels = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        p_world = np.array(
            [
                world_xy_per_arm[i, 0],
                world_xy_per_arm[i, 1],
                world_xy_per_arm[i, 2],
                1.0,
            ],
            dtype=np.float64,
        )
        p_cam = cam_t_world @ p_world  # (4,)
        X, Y, Z = p_cam[0], p_cam[1], p_cam[2]
        if Z <= 0:
            # Point is behind the camera; the OpenCV pinhole formula
            # would give a nonsense pixel. Return NaN so the caller
            # can decide what to do.
            pixels[i] = [np.nan, np.nan]
            continue
        u = K[0, 0] * X / Z + K[0, 2]
        v = K[1, 1] * Y / Z + K[1, 2]
        pixels[i] = [u, v]
    return pixels


def project_expert_grippers(
    hdf5_path: str,
    t: int,
    cam_key: str = "camera_1",
) -> dict[str, Any]:
    """Open `hdf5_path`, project the two gripper positions at frame `t`.

    Returns
    -------
    `{"pixels": (2, 2) float64, "world_xy_4d": (4,) float32,
      "world_xyz_per_arm": (2, 3) float64}`.

    `pixels` is the per-arm (u, v) for `cam_key`'s image. `world_xy_4d`
    is the 4-dim approximation of the dataset's expert action target.
    `world_xyz_per_arm` is the full 3D position used for projection.
    """
    with h5py.File(hdf5_path, "r") as f:
        ee = np.asarray(f["obs/ee_pos"][t], dtype=np.float64)
        base = np.asarray(f["obs/world_t_robot_base"][t], dtype=np.float64)
        K = np.asarray(f[f"obs/images/{cam_key}_intrinsics"][t], dtype=np.float64)
        ext = np.asarray(f[f"obs/images/{cam_key}_extrinsics"][t], dtype=np.float64)

    w_t_r0 = base[0]
    left_world = (w_t_r0 @ np.array([ee[0], ee[1], ee[2], 1.0]))[:3]
    right_world = (w_t_r0 @ np.array([ee[7], ee[8], ee[9], 1.0]))[:3]
    world_xyz_per_arm = np.stack([left_world, right_world])

    pixels = world_xy_to_pixel(world_xyz_per_arm, K, ext)
    world_xy_4d = ee_pos_to_world_xy(ee.astype(np.float32), base.astype(np.float32))

    return {
        "pixels": pixels,
        "world_xy_4d": world_xy_4d,
        "world_xyz_per_arm": world_xyz_per_arm,
    }
