"""Visualize the CV pipeline on a diverse batch of decoded frames.

For 12 frames spanning 4 dataset episodes (early/mid/late):
  HDF5 raw RGB (480x640) -> center crop 480x480 -> resize 128x128
  -> encoder_forward -> decoder (render_img_cm) -> CV pose estimator
  -> overlay (cx, cy, theta) on the decoded RGB.

Outputs an individual annotated PNG per frame and a grid combining all 12.

Run from repo root in the iws conda env:
    conda run -n iws python tests/state_estimator/visualize_examples.py

Outputs land in `tests/state_estimator/sanity_outputs/examples/`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.models.world_model import DifferentiableDynamics  # noqa: E402
from rl.visualization.state_viz import render_state_on_image  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
DATASET_DIR = REPO_ROOT / "data" / "mini" / "pusht" / "train"
OUTPUT_DIR = REPO_ROOT / "tests" / "state_estimator" / "sanity_outputs" / "examples"
SUPERVISOR_ANALYZE = (
    Path.home() / "Documents" / "aloha" / "aloha" / "world_model" / "eval" / "analyze.py"
)

OBS_KEY = "camera_1_color"
RESOLUTION = 128
DEVICE = "cuda:0"

EPISODES = [0, 1, 2, 3]
FRAC_INDICES = [0.05, 0.50, 0.95]  # early, mid, late

UPSCALE = 4  # for visibility of overlays in the grid; per-tile upscaling
COLOR_MARKER = (0, 220, 0)  # green for the only preset we use here


def _load_supervisor_module():
    spec = importlib.util.spec_from_file_location("supervisor_analyze", SUPERVISOR_ANALYZE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    sh = (h - s) // 2
    sw = (w - s) // 2
    return img[sh:sh + s, sw:sw + s]


def preprocess(raw_rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> (1, 3, 128, 128) float32 in [0, 1] (pre-normalizer)."""
    cropped = center_crop_square(raw_rgb)
    resized = cv2.resize(cropped, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def decode_and_estimate(wm, sup, raw_rgb: np.ndarray, hsv_lower, hsv_upper):
    """Returns (decoded_rgb_uint8, mask_u8, record_dict)."""
    img_pre = preprocess(raw_rgb).to(DEVICE)

    with torch.no_grad():
        z = wm.encode(img_pre)        # (1, C, H, W)
        rgb = wm.decode(z, RESOLUTION) # (1, 3, H, W) in [0, 1]

    rgb_np = rgb[0].permute(1, 2, 0).detach().cpu().float().numpy()
    rgb_uint8 = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    mask = sup.detect_t_block_mask(bgr, hsv_lower, hsv_upper)

    t_scale = RESOLUTION / 512.0
    template_contour = sup.get_template_contour(sup.T_BLOCK_SHAPE, t_scale)
    center, angle_deg, error = sup.estimate_current_pose(
        bgr, template_contour, t_scale, hsv_lower, hsv_upper
    )

    record = {
        "z_norm": float(z.float().flatten().norm().item()),
        "z_shape": list(z.shape),
        "mask_pixels": int((mask > 0).sum()),
        "success": center is not None,
    }
    if center is not None:
        record["center_px"] = [float(center[0]), float(center[1])]
        record["angle_deg"] = float(angle_deg)
        record["icp_residual"] = float(error)
    return rgb_uint8, mask, record


def annotate(rgb_uint8: np.ndarray, record: dict, caption: str) -> np.ndarray:
    """Annotate the decoded image with marker + arrow + small text caption."""
    out = rgb_uint8.copy()
    if record["success"]:
        cx, cy = record["center_px"]
        theta = np.deg2rad(record["angle_deg"])
        out = render_state_on_image(
            out, cx, cy, np.sin(theta), np.cos(theta), color=COLOR_MARKER
        )
    out_big = cv2.resize(
        out, (out.shape[1] * UPSCALE, out.shape[0] * UPSCALE),
        interpolation=cv2.INTER_NEAREST,
    )

    h_pad = 28
    canvas = np.full((out_big.shape[0] + h_pad, out_big.shape[1], 3), 255, dtype=np.uint8)
    canvas[h_pad:, :] = out_big
    cv2.putText(
        canvas, caption, (4, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )
    return canvas


def make_grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    rows = (len(tiles) + cols - 1) // cols
    h, w = tiles[0].shape[:2]
    blank = np.full((h, w, 3), 255, dtype=np.uint8)
    pad = blank.copy()
    grid_rows = []
    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            idx = r * cols + c
            row_tiles.append(tiles[idx] if idx < len(tiles) else pad)
        grid_rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(grid_rows, axis=0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sup = _load_supervisor_module()
    hsv_lower, hsv_upper = sup.HSV_LOWER_REAL, sup.HSV_UPPER_REAL  # equivalent to WM here

    print(f"loading WM from {CKPT_PATH}")
    free_pre, _ = torch.cuda.mem_get_info(0)
    print(f"free GPU mem before load: {free_pre / 1024**2:.0f} MiB")
    t0 = time.time()
    wm = DifferentiableDynamics(str(CKPT_PATH), device=DEVICE)
    print(f"WM loaded in {time.time() - t0:.1f}s")

    tiles, all_records = [], []
    for ep_id in EPISODES:
        ep_path = DATASET_DIR / f"episode_{ep_id}.hdf5"
        if not ep_path.exists():
            print(f"skipping missing {ep_path}")
            continue
        with h5py.File(ep_path, "r") as f:
            n_frames = f[f"obs/images/{OBS_KEY}"].shape[0]
            for frac in FRAC_INDICES:
                t_idx = int(frac * (n_frames - 1))
                raw_rgb = f[f"obs/images/{OBS_KEY}"][t_idx]  # (480, 640, 3) uint8 RGB

                rgb_dec, mask, rec = decode_and_estimate(
                    wm, sup, raw_rgb, hsv_lower, hsv_upper
                )
                rec.update({"episode": ep_id, "frame": t_idx})
                all_records.append(rec)

                tag = f"ep{ep_id} t={t_idx}/{n_frames - 1}"
                if rec["success"]:
                    cx, cy = rec["center_px"]
                    info = (
                        f"  cxy=({cx:.1f},{cy:.1f}) "
                        f"th={rec['angle_deg']:+.1f}deg "
                        f"area={rec['mask_pixels']}px"
                    )
                else:
                    info = "  CV FAIL"
                caption = tag + info

                tile = annotate(rgb_dec, rec, caption)
                fname = f"ep{ep_id}_t{t_idx:04d}.png"
                cv2.imwrite(str(OUTPUT_DIR / fname), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                tiles.append(tile)
                print(f"  {caption}")

    if tiles:
        grid = make_grid(tiles, cols=3)
        grid_path = OUTPUT_DIR / "grid.png"
        cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"grid saved: {grid_path}")

    with (OUTPUT_DIR / "results.json").open("w") as f:
        json.dump(all_records, f, indent=2)

    n_ok = sum(1 for r in all_records if r["success"])
    print(f"\ndone: {n_ok}/{len(all_records)} CV detections succeeded")


if __name__ == "__main__":
    main()
