"""Phase-1 sanity check: decode z_goal, run supervisor's CV pose estimator
with both preset HSV ranges, save masks + annotated overlays.

What this proves before we touch anything else:
1. Our WM checkpoint loads and decode(z_goal) produces a 128x128 RGB.
2. The supervisor's `estimate_current_pose` runs without crashing on our
   decoder outputs.
3. Whether either preset HSV range (REAL or WM) actually finds the T-block
   in our decoder outputs — this is the open question Section 3.1 of the
   task brief flagged.

Run from repo root in the iws conda env:
    conda run -n iws python tests/state_estimator/sanity_check.py

Outputs land in `tests/state_estimator/sanity_outputs/`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl.models.world_model import DifferentiableDynamics  # noqa: E402
from rl.visualization.state_viz import render_state_on_image  # noqa: E402

CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
Z_GOAL_PATH = REPO_ROOT / "tests" / "goal_selection" / "z_goal.pt"
OUTPUT_DIR = REPO_ROOT / "tests" / "state_estimator" / "sanity_outputs"
SUPERVISOR_ANALYZE = Path.home() / "Documents" / "aloha" / "aloha" / "world_model" / "eval" / "analyze.py"

RESOLUTION = 128
PROBE_COLOR_REAL = (0, 200, 0)   # green for REAL preset
PROBE_COLOR_WM = (0, 120, 255)   # orange for WM preset
DEVICE = "cuda:0"


def _load_supervisor_module():
    """Load supervisor's analyze.py directly, bypassing the aloha package
    __init__.py which pulls in ROS-only imports."""
    spec = importlib.util.spec_from_file_location("supervisor_analyze", SUPERVISOR_ANALYZE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {SUPERVISOR_ANALYZE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_goal(ckpt_path: Path, z_goal_path: Path) -> tuple[np.ndarray, dict]:
    """Returns (rgb_uint8, info). rgb_uint8 is (H, W, 3) RGB."""
    print(f"loading WM from {ckpt_path}")
    free_before, _ = torch.cuda.mem_get_info(0)
    print(f"free GPU mem before load: {free_before / 1024**2:.0f} MiB")

    t0 = time.time()
    wm = DifferentiableDynamics(str(ckpt_path), device=DEVICE)
    load_s = time.time() - t0

    free_after_load, total = torch.cuda.mem_get_info(0)
    print(f"WM loaded in {load_s:.1f}s; free GPU mem: {free_after_load / 1024**2:.0f} / {total / 1024**2:.0f} MiB")

    z_goal = torch.load(z_goal_path, map_location=DEVICE, weights_only=False)
    if z_goal.ndim == 4:
        z_in = z_goal
    else:
        raise ValueError(f"unexpected z_goal shape {tuple(z_goal.shape)}")
    print(f"z_goal shape: {tuple(z_in.shape)}, norm: {z_in.float().flatten().norm().item():.2f}")

    t0 = time.time()
    rgb = wm.decode(z_in.to(DEVICE), resolution=RESOLUTION)  # (1, 3, H, W) in [0, 1]
    decode_s = time.time() - t0
    free_after_decode, _ = torch.cuda.mem_get_info(0)
    print(f"decode in {decode_s:.2f}s; free GPU mem: {free_after_decode / 1024**2:.0f} MiB")

    rgb_np = rgb[0].permute(1, 2, 0).detach().cpu().float().numpy()
    rgb_uint8 = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)

    info = {
        "ckpt": str(ckpt_path),
        "z_goal": str(z_goal_path),
        "z_shape": list(z_in.shape),
        "z_norm": float(z_in.float().flatten().norm().item()),
        "wm_load_s": load_s,
        "decode_s": decode_s,
        "free_mib_after_load": free_after_load / 1024**2,
        "free_mib_after_decode": free_after_decode / 1024**2,
        "total_mib": total / 1024**2,
    }
    return rgb_uint8, info


def run_one_preset(rgb_uint8: np.ndarray, sup, hsv_lower, hsv_upper, name: str):
    """Returns dict with keys: name, mask, center, angle_deg, error, area, contour_count."""
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    mask = sup.detect_t_block_mask(bgr, hsv_lower, hsv_upper)
    mask_pixels = int((mask > 0).sum())

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_area = 0.0
    if contours:
        largest_area = float(cv2.contourArea(max(contours, key=cv2.contourArea)))

    t_scale = RESOLUTION / 512.0
    template_contour = sup.get_template_contour(sup.T_BLOCK_SHAPE, t_scale)

    t0 = time.time()
    center, angle_deg, error = sup.estimate_current_pose(
        bgr, template_contour, t_scale, hsv_lower, hsv_upper
    )
    icp_s = time.time() - t0

    success = center is not None
    record = {
        "name": name,
        "hsv_lower": [int(v) for v in hsv_lower],
        "hsv_upper": [int(v) for v in hsv_upper],
        "mask_pixels": mask_pixels,
        "contour_count": len(contours),
        "largest_contour_area_px": largest_area,
        "icp_s": icp_s,
        "success": success,
    }
    if success:
        record["center_px"] = [float(center[0]), float(center[1])]
        record["angle_deg"] = float(angle_deg)
        record["icp_residual"] = float(error) if error is not None else None
    return record, mask


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"WM ckpt missing: {CKPT_PATH}")
    if not Z_GOAL_PATH.exists():
        raise FileNotFoundError(f"z_goal missing: {Z_GOAL_PATH}")
    if not SUPERVISOR_ANALYZE.exists():
        raise FileNotFoundError(f"supervisor analyze.py missing: {SUPERVISOR_ANALYZE}")

    sup = _load_supervisor_module()
    print(f"loaded supervisor module: HSV_REAL={sup.HSV_LOWER_REAL.tolist()}-{sup.HSV_UPPER_REAL.tolist()}")
    print(f"                          HSV_WM  ={sup.HSV_LOWER_WM.tolist()}-{sup.HSV_UPPER_WM.tolist()}")

    rgb_uint8, decode_info = decode_goal(CKPT_PATH, Z_GOAL_PATH)
    cv2.imwrite(str(OUTPUT_DIR / "00_decoded_rgb.png"), cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))

    presets = [
        ("real", sup.HSV_LOWER_REAL, sup.HSV_UPPER_REAL, PROBE_COLOR_REAL),
        ("wm",   sup.HSV_LOWER_WM,   sup.HSV_UPPER_WM,   PROBE_COLOR_WM),
    ]

    results = {"decode": decode_info, "presets": []}
    annotated_combined = rgb_uint8.copy()

    for preset_name, lo, hi, color in presets:
        rec, mask = run_one_preset(rgb_uint8, sup, lo, hi, preset_name)
        results["presets"].append(rec)
        cv2.imwrite(str(OUTPUT_DIR / f"01_mask_{preset_name}.png"), mask)

        annotated = rgb_uint8.copy()
        if rec["success"]:
            cx, cy = rec["center_px"]
            theta = np.deg2rad(rec["angle_deg"])
            annotated = render_state_on_image(
                annotated, cx, cy, np.sin(theta), np.cos(theta),
                color=color, label=preset_name,
            )
            annotated_combined = render_state_on_image(
                annotated_combined, cx, cy, np.sin(theta), np.cos(theta),
                color=color, label=preset_name,
            )
        cv2.imwrite(
            str(OUTPUT_DIR / f"02_annotated_{preset_name}.png"),
            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
        )

    cv2.imwrite(
        str(OUTPUT_DIR / "03_annotated_both.png"),
        cv2.cvtColor(annotated_combined, cv2.COLOR_RGB2BGR),
    )

    with (OUTPUT_DIR / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print("SANITY CHECK SUMMARY")
    print("=" * 70)
    for rec in results["presets"]:
        flag = "OK " if rec["success"] else "FAIL"
        print(
            f"[{flag}] preset={rec['name']:5s} "
            f"mask_px={rec['mask_pixels']:6d} "
            f"largest_area={rec['largest_contour_area_px']:7.1f} "
            f"icp_s={rec['icp_s']:.2f}"
        )
        if rec["success"]:
            cx, cy = rec["center_px"]
            print(
                f"        center=({cx:6.2f}, {cy:6.2f}) px  "
                f"angle={rec['angle_deg']:+7.2f} deg  "
                f"icp_residual={rec['icp_residual']:.3f}"
            )
    print(f"\noutputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
