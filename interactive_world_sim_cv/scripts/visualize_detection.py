"""Visualize T-pose detection on real and decoded frames.

Produces two PNGs in /tmp/:

  cv_overlay_real.png     — 10 real episode frames (t = 0, 20, ..., 180)
                            with detection at mode='real'.
  cv_overlay_decoded.png  — 10 decoded frames from a fresh 100-step
                            imagination rollout, sampled at t = 10, 20,
                            ..., 100, with detection at mode='wm'.

Each row in both PNGs has three 512×512 columns:
  - SOURCE: the native-resolution frame (128² or 480²), upscaled to
    512 with INTER_NEAREST so source pixelation is visible.
  - UPSCALED (512²): cv2.INTER_CUBIC upscale to 512, matching the
    actual frame the detector ingests internally.
  - DETECTED: UPSCALED with marker drawn — filled green circle at
    (x, y), radius 8, plus a thickness-3 green line of length 30 px
    in direction (cos, sin). Empty + red "DETECTION FAILED" text if
    detect() returned None.

Run from the repo root:

    python interactive_world_sim_cv/scripts/visualize_detection.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: E402
import h5py  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402

from interactive_world_sim_cv import TPose, detect  # noqa: E402


EPISODE = str(_REPO_ROOT / "data" / "mini" / "pusht" / "val" / "episode_0.hdf5")
PROCESSING_RES = 512
CELL = 512
PAD = 20
ROW_LABEL_H = 30
HEADER_H = 36
BG = 32  # dark gray
FONT = cv2.FONT_HERSHEY_SIMPLEX

OUT_REAL = "/tmp/cv_overlay_real.png"
OUT_DECODED = "/tmp/cv_overlay_decoded.png"

REAL_TS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
DECODED_TS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


# --------------------------------------------------------------- frame fetch


def _center_crop_640x480_to_480x480(rgb: np.ndarray) -> np.ndarray:
    """Drop 80 px from each side of width — matches env's preprocessing."""
    h, w = rgb.shape[:2]
    if (h, w) != (480, 640):
        raise ValueError(f"expected (480, 640) frame; got {(h, w)}")
    return rgb[:, 80:560]


def _load_real_frames() -> list[np.ndarray]:
    """Load 10 frames from camera_1_color at REAL_TS, center-cropped to 480²."""
    frames = []
    with h5py.File(EPISODE, "r") as f:
        ds = f["obs/images/camera_1_color"]
        for t in REAL_TS:
            raw = np.asarray(ds[t])  # (480, 640, 3) uint8 RGB
            frames.append(_center_crop_640x480_to_480x480(raw))
    return frames


def _generate_decoded_frames() -> list[np.ndarray]:
    """Warmup + 91-step imagination rollout, capture decoded RGB at DECODED_TS."""
    from interactive_world_sim_env import WorldModelEnv
    from interactive_world_sim_env.helpers.expert_action import (
        expert_action_from_episode,
    )

    target_set = set(DECODED_TS)
    horizon = max(DECODED_TS) - 10 + 1  # = 91 steps to reach t=100 from t=10

    env = WorldModelEnv("pusht_cam1")
    env.reset(
        init_episode_path=EPISODE,
        init_episode_index=9,
        init_window_size=10,
    )
    captured: list[tuple[int, np.ndarray]] = []
    for offset in range(horizon):
        t = 10 + offset
        action_t = expert_action_from_episode(env, EPISODE, t - 1)
        env.step(action_t)
        if t in target_set:
            captured.append((t, env.render()))
    env.close()
    # Sort by t (DECODED_TS is already sorted; this is belt-and-suspenders).
    captured.sort(key=lambda x: x[0])
    if [t for t, _ in captured] != DECODED_TS:
        raise RuntimeError(
            f"captured ts mismatch: got {[t for t, _ in captured]}, want {DECODED_TS}"
        )
    return [frame for _, frame in captured]


# --------------------------------------------------------------- compose cells


def _to_512_nearest(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (CELL, CELL), interpolation=cv2.INTER_NEAREST)


def _to_512_cubic(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (CELL, CELL), interpolation=cv2.INTER_CUBIC)


def _draw_marker(rgb_512: np.ndarray, pose: TPose) -> np.ndarray:
    """Filled green circle + line in (cos, sin) direction. Green is (0,255,0)
    in both RGB and BGR (G is symmetric), so we can draw on RGB directly.
    """
    out = rgb_512.copy()
    x, y = int(round(pose.x)), int(round(pose.y))
    x2 = int(round(pose.x + 30.0 * pose.cos))
    y2 = int(round(pose.y + 30.0 * pose.sin))
    cv2.line(out, (x, y), (x2, y2), color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.circle(out, (x, y), radius=8, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
    return out


def _draw_failed(rgb_512: np.ndarray) -> np.ndarray:
    """Centered 'DETECTION FAILED' in red (RGB 255, 0, 0)."""
    out = rgb_512.copy()
    text = "DETECTION FAILED"
    scale = 1.2
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (CELL - tw) // 2
    y = (CELL + th) // 2
    cv2.putText(out, text, (x, y), FONT, scale, color=(255, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)
    return out


# --------------------------------------------------------------- overlay


def _make_header(source_label: str) -> np.ndarray:
    """Top column-header band, full width."""
    W = 4 * PAD + 3 * CELL
    img = np.full((HEADER_H, W, 3), BG, dtype=np.uint8)
    centers = [
        PAD + CELL // 2,
        2 * PAD + CELL + CELL // 2,
        3 * PAD + 2 * CELL + CELL // 2,
    ]
    labels = [source_label, "UPSCALED (512²)", "DETECTED"]
    for c, label in zip(centers, labels):
        (tw, _th), _ = cv2.getTextSize(label, FONT, 0.7, 2)
        cv2.putText(
            img, label,
            (c - tw // 2, HEADER_H - 10),
            FONT, 0.7, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA,
        )
    return img


def _make_row(label_text: str, col1: np.ndarray, col2: np.ndarray, col3: np.ndarray) -> np.ndarray:
    """One 542×W row: label band + three 512² cells side by side."""
    W = 4 * PAD + 3 * CELL
    img = np.full((ROW_LABEL_H + CELL, W, 3), BG, dtype=np.uint8)
    cv2.putText(
        img, label_text,
        (PAD, ROW_LABEL_H - 8),
        FONT, 0.7, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA,
    )
    img[ROW_LABEL_H : ROW_LABEL_H + CELL, PAD : PAD + CELL] = col1
    img[ROW_LABEL_H : ROW_LABEL_H + CELL, 2 * PAD + CELL : 2 * PAD + 2 * CELL] = col2
    img[ROW_LABEL_H : ROW_LABEL_H + CELL, 3 * PAD + 2 * CELL : 3 * PAD + 3 * CELL] = col3
    return img


def _build_overlay(
    frames: list[np.ndarray],
    ts: list[int],
    mode: str,
    source_label: str,
    out_path: str,
) -> dict:
    """Compose the 10-row overlay PNG, return per-frame detection stats."""
    if len(frames) != len(ts):
        raise ValueError(f"len(frames)={len(frames)} != len(ts)={len(ts)}")

    stats = {
        "ts": list(ts),
        "poses": [],
        "errors": [],
        "x_vals": [],
        "y_vals": [],
        "succeed": [],
    }

    rows = []
    for t, src in zip(ts, frames):
        col1 = _to_512_nearest(src)
        col2 = _to_512_cubic(src)
        pose = detect(src, mode=mode, processing_resolution=PROCESSING_RES)
        if pose is None:
            col3 = _draw_failed(col2)
            label = f"t={t:>3d}   None"
            stats["poses"].append(None)
            stats["succeed"].append(False)
        else:
            col3 = _draw_marker(col2, pose)
            label = f"t={t:>3d}   err={pose.error:.3f}"
            stats["poses"].append(pose)
            stats["errors"].append(pose.error)
            stats["x_vals"].append(pose.x)
            stats["y_vals"].append(pose.y)
            stats["succeed"].append(True)
        rows.append(_make_row(label, col1, col2, col3))

    header = _make_header(source_label)
    img = np.concatenate([header] + rows, axis=0)
    iio.imwrite(out_path, img)
    print(f"  wrote {out_path}  shape={img.shape}")
    return stats


# --------------------------------------------------------------- report


def _report(name: str, stats: dict) -> None:
    print(f"\n--- {name} ---")
    n_total = len(stats["ts"])
    n_succ = sum(stats["succeed"])
    print(f"  success: {n_succ}/{n_total}")
    if n_succ:
        errs = np.asarray(stats["errors"])
        xs = np.asarray(stats["x_vals"])
        ys = np.asarray(stats["y_vals"])
        print(f"  error  min={errs.min():.4f}  median={float(np.median(errs)):.4f}  "
              f"mean={errs.mean():.4f}  max={errs.max():.4f}  (pixels in 512² processing space)")
        print(f"  x range: [{xs.min():.2f}, {xs.max():.2f}]   (must be in [0, 512))")
        print(f"  y range: [{ys.min():.2f}, {ys.max():.2f}]   (must be in [0, 512))")
        print("  per-frame:")
        for t, succ, pose in zip(stats["ts"], stats["succeed"], stats["poses"]):
            if succ:
                print(
                    f"    t={t:>3d}  x={pose.x:7.2f}  y={pose.y:7.2f}  "
                    f"angle={pose.angle_deg:7.2f}°  err={pose.error:.4f}"
                )
            else:
                print(f"    t={t:>3d}  DETECTION FAILED")


def _trend_observation(stats: dict) -> str:
    """One-line qualitative observation of error vs t (for decoded overlay)."""
    if sum(stats["succeed"]) < 3:
        return "too few successful detections to assess error trend"
    pairs = [(t, e) for t, succ, e in zip(stats["ts"], stats["succeed"], [None] + [None] * len(stats["ts"]))]
    # Build trend from actual values
    ts = []
    errs = []
    for t, succ, pose in zip(stats["ts"], stats["succeed"], stats["poses"]):
        if succ:
            ts.append(t)
            errs.append(pose.error)
    if len(ts) < 3:
        return "too few successful detections to assess error trend"
    slope = np.polyfit(ts, errs, 1)[0]
    if slope > 0.005:
        return f"error trends UPWARD with t (slope ≈ {slope:.4f}/frame) — later imagination steps detect worse"
    if slope < -0.005:
        return f"error trends DOWNWARD with t (slope ≈ {slope:.4f}/frame) — later imagination steps detect better"
    return f"error stays roughly flat with t (slope ≈ {slope:.4f}/frame) — no monotonic degradation"


# --------------------------------------------------------------- main


def main() -> int:
    print("=" * 72)
    print("interactive_world_sim_cv — Phase 4 detection visualization")
    print("=" * 72)
    print(f"  episode      : {EPISODE}")
    print(f"  real ts      : {REAL_TS}")
    print(f"  decoded ts   : {DECODED_TS}")
    print(f"  processing   : {PROCESSING_RES}²")

    # ---- Real frames ----
    print("\n[1/2] loading real frames + detecting (mode='real')...")
    t0 = time.perf_counter()
    real_frames = _load_real_frames()
    real_stats = _build_overlay(
        real_frames, REAL_TS, mode="real",
        source_label="SOURCE (480²)", out_path=OUT_REAL,
    )
    print(f"  took {time.perf_counter() - t0:.2f}s")

    # ---- Decoded frames ----
    print("\n[2/2] generating decoded frames + detecting (mode='wm')...")
    t0 = time.perf_counter()
    decoded_frames = _generate_decoded_frames()
    decoded_stats = _build_overlay(
        decoded_frames, DECODED_TS, mode="wm",
        source_label="SOURCE (128²)", out_path=OUT_DECODED,
    )
    print(f"  took {time.perf_counter() - t0:.2f}s")

    # ---- Numeric reports ----
    print("\n" + "=" * 72)
    print("Numeric report")
    print("=" * 72)
    _report("real overlay (mode='real')", real_stats)
    _report("decoded overlay (mode='wm')", decoded_stats)
    print(f"\n  decoded error trend: {_trend_observation(decoded_stats)}")

    print("\n" + "=" * 72)
    print(f"  written: {OUT_REAL}")
    print(f"  written: {OUT_DECODED}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
