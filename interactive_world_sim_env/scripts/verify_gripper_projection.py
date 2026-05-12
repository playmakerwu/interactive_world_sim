"""Visual verification: do projected gripper positions land on the actual grippers?

Uses the 4 mm-error closed-form recipe from
`interactive_world_sim_env.helpers.projection` plus the camera
intrinsics/extrinsics stored in the HDF5 to project both gripper
positions to pixel coordinates, then draws red (left) and blue
(right) circles on the real RGB frames at t = 0, 25, 50, 100, 150
from `data/mini/pusht/val/episode_0.hdf5`.

No model, no decoder, no env — only HDF5 + numpy + cv2 + imageio.

Run from the repo root:

    python interactive_world_sim_env/scripts/verify_gripper_projection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling `interactive_world_sim_env` package importable when
# this script is invoked directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import h5py
import imageio.v3 as iio
import numpy as np

from interactive_world_sim_env.helpers.projection import project_expert_grippers

HDF5_PATH = str(
    _REPO_ROOT / "data" / "mini" / "pusht" / "val" / "episode_0.hdf5"
)
TS = [0, 25, 50, 100, 150]
CAM_KEY = "camera_1"

# Channels are RGB (we use imageio to save).
LEFT_COLOR_RGB = (255, 0, 0)   # red
RIGHT_COLOR_RGB = (0, 0, 255)  # blue
TEXT_COLOR_RGB = (0, 255, 0)   # green


def _annotate(image_rgb: np.ndarray, pixels: np.ndarray, t: int) -> np.ndarray:
    """Draw left/right circles + a small label. Returns a new image."""
    img = image_rgb.copy()
    (u_l, v_l), (u_r, v_r) = pixels
    if np.isfinite(u_l) and np.isfinite(v_l):
        cv2.circle(
            img,
            (int(round(float(u_l))), int(round(float(v_l)))),
            radius=8,
            color=LEFT_COLOR_RGB,
            thickness=2,
        )
    if np.isfinite(u_r) and np.isfinite(v_r):
        cv2.circle(
            img,
            (int(round(float(u_r))), int(round(float(v_r)))),
            radius=8,
            color=RIGHT_COLOR_RGB,
            thickness=2,
        )
    label = (
        f"t={t} "
        f"L=({int(round(float(u_l)))},{int(round(float(v_l)))}) "
        f"R=({int(round(float(u_r)))},{int(round(float(v_r)))})"
    )
    cv2.putText(
        img,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR_RGB,
        2,
    )
    return img


def main() -> int:
    with h5py.File(HDF5_PATH, "r") as f:
        n, H, W, C = f[f"obs/images/{CAM_KEY}_color"].shape
    print(f"episode length: {n}  image: {H}x{W}x{C}")

    annotated: list[np.ndarray] = []
    for t in TS:
        result = project_expert_grippers(HDF5_PATH, t, cam_key=CAM_KEY)
        pixels = result["pixels"]
        wxy = result["world_xy_4d"]
        with h5py.File(HDF5_PATH, "r") as f:
            rgb = np.asarray(f[f"obs/images/{CAM_KEY}_color"][t])
        ann = _annotate(rgb, pixels, t)
        out_path = f"/tmp/projection_check_t{t}.png"
        iio.imwrite(out_path, ann)
        print(
            f"  t={t:>3d}  pixels=L({pixels[0,0]:6.1f},{pixels[0,1]:6.1f}) "
            f"R({pixels[1,0]:6.1f},{pixels[1,1]:6.1f})  "
            f"world_xy_4d={wxy.tolist()}  -> {out_path}"
        )
        annotated.append(ann)

    grid = np.concatenate(annotated, axis=1)  # 480 x (640*5) x 3
    iio.imwrite("/tmp/projection_check_grid.png", grid)
    print(f"wrote /tmp/projection_check_grid.png  shape={grid.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
