"""Extract the final frame from each PushT episode for goal-state review.

Saves PNGs to tests/goal_selection/frames/episode_{i}_final.png.
Uses val (5 episodes) + train (5 episodes) to reach 10 total.

Usage (from repo root):
    conda run -n iws python tests/goal_selection/extract_final_frames.py
"""

from pathlib import Path

import cv2
import numpy as np
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

DATASET_DIRS = [
    "data/mini/pusht/val",
    "data/mini/pusht/train",
]
OBS_KEY = "camera_1_color"
RESOLUTION = 128
OUTPUT_DIR = Path("tests/goal_selection/frames")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    for dataset_dir in DATASET_DIRS:
        ddir = Path(dataset_dir)
        split = ddir.name
        episode_files = sorted(ddir.glob("episode_*.hdf5"))
        for ep_path in episode_files:
            if saved >= 10:
                break
            epi_data, _ = load_dict_from_hdf5(str(ep_path))
            images = epi_data["obs"]["images"][OBS_KEY][()]  # (T, H, W, 3) uint8
            T = images.shape[0]

            # take the last frame
            last_frame = images[-1]  # (H, W, 3) uint8

            # apply same preprocessing as inference: center crop + resize
            last_frame = center_crop(last_frame, (RESOLUTION, RESOLUTION))
            last_frame = cv2.resize(
                last_frame, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA
            )

            # save as PNG (convert RGB → BGR for cv2)
            out_name = f"episode_{saved}_final.png"
            out_path = OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR))

            print(
                f"[{saved:2d}] {split}/{ep_path.name}  "
                f"length={T:4d}  → {out_path}"
            )
            saved += 1

        if saved >= 10:
            break

    print(f"\nSaved {saved} final-frame PNGs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
