"""Visual check: does the projected expert-action pixel land on the
gripper that the DECODER paints, not just on the real camera image?

Approach (b) from the spec: for each sample t we do a fresh 1-step
prediction. We do NOT autoregressively roll out to t.

1. env.reset(init_episode_path=episode, init_episode_index=t-1)
   -> latent_window = encode(real RGB at frame t-1)
2. action_norm = expert_action_from_episode(env, episode, t-1)
   -> normalized world-frame XY that the expert was at at frame t-1
3. env.step(action_norm)
   -> dynamics produces the latent the model predicts for frame t
4. decoded_rgb = env.render()  # (128, 128, 3) uint8
5. Compute the SAME action's projection pixel:
   - un-normalize the action through the model's normalizer
   - take Z from obs/ee_pos[t] (per the spec)
   - project via helpers.projection.world_xy_to_pixel
6. Rescale 640x480 -> 128x128 using the same center-crop + resize
   pipeline env uses internally (see helpers/projection.py and
   env.py:_rgb_to_chw_float01). The crop discovered in
   yixuan_utilities.draw_utils.center_crop is aspect-ratio-driven:
   for a 640x480 input and a 1:1 target, it returns the center
   480x480, dropping 80 px on each side. The resize is uniform.
   Pixel mapping:
       u_decoded = (u_real - 80) * (128 / 480)
       v_decoded = v_real * (128 / 480)
7. Draw red (left arm) and blue (right arm) circles on the decoded
   RGB at the rescaled pixels, radius=3 thickness=2 (image is small).
8. Build a side-by-side grid:
     Row 0: decoded 128x128 -> NN-upscale to 384x384, circles drawn
            in upscaled coords (radius=10 for visibility)
     Row 1: real 640x480 -> downscale to 384x288 (preserves 4:3),
            circles drawn in downscaled coords (radius=8)
   Final grid is (672, 1920, 3) uint8.

CAVEATS to keep in mind reading the output:
- The decoder is stochastic. Re-running produces slightly different
  decoded frames; the circle/gripper alignment should be judged
  modulo that jitter.
- This is a 1-step prediction at each t, not a multi-step
  autoregressive rollout. Long-horizon prediction quality is
  measured separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import h5py
import imageio.v3 as iio
import numpy as np
import torch

from interactive_world_sim_env import WorldModelEnv
from interactive_world_sim_env.helpers.expert_action import (
    expert_action_from_episode,
)
from interactive_world_sim_env.helpers.projection import world_xy_to_pixel

EPISODE = str(
    _REPO_ROOT / "data" / "mini" / "pusht" / "val" / "episode_0.hdf5"
)
SAMPLE_TS = [1, 25, 50, 100, 150]
CAM_KEY = "camera_1"
RES = 128
DECODED_DISPLAY = 384
REAL_DISPLAY_W = 384
REAL_DISPLAY_H = 288

LEFT_COLOR_RGB = (255, 0, 0)
RIGHT_COLOR_RGB = (0, 0, 255)
TEXT_COLOR_RGB = (0, 255, 0)


def _real_to_decoded_pixel(u_real: float, v_real: float) -> tuple[float, float]:
    """Map a (u, v) in 640x480 to (u, v) in 128x128 after env's center-crop+resize."""
    # center_crop with aspect 1:1 keeps the center 480x480 (drops 80 px left/right)
    u_cropped = u_real - 80.0
    v_cropped = v_real
    scale = RES / 480.0
    return u_cropped * scale, v_cropped * scale


def _project_action_pixels(env: WorldModelEnv, action_norm: np.ndarray, t: int) -> np.ndarray:
    """Un-normalize the action, take Z from obs/ee_pos[t], project to camera_1 pixels.

    Returns (2, 2): [[u_left, v_left], [u_right, v_right]] in 640x480 pixels.
    """
    normalizer = env._loaded.model.normalizer["action"]
    action_unnorm = (
        normalizer.unnormalize(torch.from_numpy(action_norm).to(env.device).float())
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )  # (4,) world-frame meters

    with h5py.File(EPISODE, "r") as f:
        ee_t = np.asarray(f["obs/ee_pos"][t], dtype=np.float64)
        base_t = np.asarray(f["obs/world_t_robot_base"][t], dtype=np.float64)
        K = np.asarray(f[f"obs/images/{CAM_KEY}_intrinsics"][t], dtype=np.float64)
        ext = np.asarray(f[f"obs/images/{CAM_KEY}_extrinsics"][t], dtype=np.float64)

    # Z for each arm comes from ee_pos[t] transformed through the LEFT robot's pose.
    left_world = (base_t[0] @ np.array([ee_t[0], ee_t[1], ee_t[2], 1.0]))[:3]
    right_world = (base_t[0] @ np.array([ee_t[7], ee_t[8], ee_t[9], 1.0]))[:3]
    left_xyz = np.array([action_unnorm[0], action_unnorm[1], left_world[2]])
    right_xyz = np.array([action_unnorm[2], action_unnorm[3], right_world[2]])
    world_xyz = np.stack([left_xyz, right_xyz])

    return world_xy_to_pixel(world_xyz, K, ext)


def _draw_circles(
    img: np.ndarray,
    left_uv: tuple[float, float],
    right_uv: tuple[float, float],
    radius: int,
    thickness: int,
) -> None:
    """Draw red/blue circles in place. Out-of-frame coords are clipped by cv2."""
    if np.isfinite(left_uv[0]) and np.isfinite(left_uv[1]):
        cv2.circle(
            img,
            (int(round(left_uv[0])), int(round(left_uv[1]))),
            radius,
            LEFT_COLOR_RGB,
            thickness,
        )
    if np.isfinite(right_uv[0]) and np.isfinite(right_uv[1]):
        cv2.circle(
            img,
            (int(round(right_uv[0])), int(round(right_uv[1]))),
            radius,
            RIGHT_COLOR_RGB,
            thickness,
        )


def _annotate_label(img: np.ndarray, label: str) -> None:
    cv2.putText(
        img, label, (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR_RGB, 2,
    )


def main() -> int:
    env = WorldModelEnv("pusht_cam1")
    print(f"task={env.task} device={env.device}")

    decoded_cells: list[np.ndarray] = []
    real_cells: list[np.ndarray] = []

    for t in SAMPLE_TS:
        # Reset to frame (t-1), step once with the expert action that drove
        # the demonstrator from t-1 -> t, render.
        env.reset(init_episode_path=EPISODE, init_episode_index=t - 1)
        action_norm = expert_action_from_episode(env, EPISODE, t - 1)
        env.step(action_norm)
        decoded = env.render()  # (128, 128, 3) uint8

        # Project the same action into camera_1's image plane.
        pixels_real = _project_action_pixels(env, action_norm, t)
        (u_l, v_l), (u_r, v_r) = pixels_real
        u_ld, v_ld = _real_to_decoded_pixel(u_l, v_l)
        u_rd, v_rd = _real_to_decoded_pixel(u_r, v_r)

        # ---- decoded cell: upscale 128 -> 384, then draw circles ----
        decoded_up = cv2.resize(
            decoded, (DECODED_DISPLAY, DECODED_DISPLAY), interpolation=cv2.INTER_NEAREST
        )
        up_scale = DECODED_DISPLAY / RES
        _draw_circles(
            decoded_up,
            (u_ld * up_scale, v_ld * up_scale),
            (u_rd * up_scale, v_rd * up_scale),
            radius=10, thickness=2,
        )
        _annotate_label(
            decoded_up,
            f"decoded t={t}  L=({int(round(u_ld))},{int(round(v_ld))}) "
            f"R=({int(round(u_rd))},{int(round(v_rd))})",
        )
        decoded_cells.append(decoded_up)

        # ---- real cell: pull from HDF5, downscale to 384x288, draw circles ----
        with h5py.File(EPISODE, "r") as f:
            real = np.asarray(f[f"obs/images/{CAM_KEY}_color"][t])
        real_down = cv2.resize(
            real, (REAL_DISPLAY_W, REAL_DISPLAY_H), interpolation=cv2.INTER_AREA
        )
        sx, sy = REAL_DISPLAY_W / 640.0, REAL_DISPLAY_H / 480.0
        _draw_circles(
            real_down,
            (u_l * sx, v_l * sy),
            (u_r * sx, v_r * sy),
            radius=8, thickness=2,
        )
        _annotate_label(
            real_down,
            f"real t={t}  L=({int(round(u_l))},{int(round(v_l))}) "
            f"R=({int(round(u_r))},{int(round(v_r))})",
        )
        real_cells.append(real_down)

        print(
            f"  t={t:>3d}  pixels_real L({u_l:6.1f},{v_l:6.1f}) "
            f"R({u_r:6.1f},{v_r:6.1f})  -> decoded "
            f"L({u_ld:5.1f},{v_ld:5.1f}) R({u_rd:5.1f},{v_rd:5.1f})"
        )

    row0 = np.concatenate(decoded_cells, axis=1)  # (384, 384*5, 3)
    row1 = np.concatenate(real_cells, axis=1)     # (288, 384*5, 3)
    grid = np.concatenate([row0, row1], axis=0)   # (672, 1920, 3)
    out_path = "/tmp/projection_on_decoded_grid.png"
    iio.imwrite(out_path, grid)
    env.close()
    print(f"wrote {out_path}  shape={grid.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
