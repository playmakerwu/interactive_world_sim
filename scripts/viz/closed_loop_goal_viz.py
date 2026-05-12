"""Closed-loop world-model-only video with goal + current pose markers.

Produces 50 PNG frames (one per MPPI control step) and an mp4 stitching
them at 5 fps. Layout:

  ┌─────────────────────────────────────┐
  │  decoded RGB from world model       │
  │  (128² upscaled to 512² INTER_CUBIC)│
  │                                     │
  │    ● ─→   (red, fixed goal pose)    │
  │       ● ─→  (green, current pose)   │
  │                                     │
  │  step k/50  reward: ...  ...        │
  └─────────────────────────────────────┘

Data sources (existing A.3 outputs):
  - overnight/.../A3_run/03_closed_loop.mp4: middle row holds the
    50 closed-loop WM-decoded frames. Extract via ffmpeg crop.
  - overnight/.../A3_run/03_run_log.csv: per-step CV pose + reward.
  - overnight/.../A1_candidates/candidates.json: easy candidate's
    end_pose (the goal).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]

DEFAULT_AUDIT = REPO / "overnight/two_fixes_2026_05_12_1438/A3_run/audit_log.json"
DEFAULT_VIDEO = REPO / "overnight/two_fixes_2026_05_12_1438/A3_run/03_closed_loop.mp4"
DEFAULT_CSV   = REPO / "overnight/two_fixes_2026_05_12_1438/A3_run/03_run_log.csv"
DEFAULT_CAND  = REPO / "overnight/two_fixes_2026_05_12_1438/A1_candidates/candidates.json"
DEFAULT_OUT   = REPO / "overnight/two_fixes_2026_05_12_1438/viz_closed_loop_goal"

# A.3 mp4 layout (verified via ffprobe):
#   total 128 wide × 456 tall × 60 frames @ 8 fps
#   each of the 3 rows = 24-px black title strip + 128-px image = 152 px tall
#   so MIDDLE row image strip (closed-loop MPPI) = y in [176, 304)
A3_ROW_TOP = 176
A3_ROW_BOT = 304   # exclusive
A3_FRAME_W = 128
A3_FRAME_H = 128
A3_WARMUP = 10     # first 10 mp4 frames are decoded warmup; MPPI steps start at frame 10
N_STEPS = 50


def extract_mppi_frames(video_path: Path, out_dir: Path) -> list[np.ndarray]:
    """Extract the 50 MPPI-step frames (mp4 indices 10..59), middle row.

    Returns a list of (128, 128, 3) uint8 RGB arrays.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total != A3_WARMUP + N_STEPS:
        print(f"[viz] WARNING: expected {A3_WARMUP + N_STEPS} frames, got {total}")
    frames: list[np.ndarray] = []
    for i in range(total):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if i < A3_WARMUP:
            continue   # skip warmup
        # Crop middle row's image strip
        row = frame_bgr[A3_ROW_TOP:A3_ROW_BOT, :A3_FRAME_W]
        frame_rgb = cv2.cvtColor(row, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def draw_pose_marker(
    canvas: np.ndarray,         # (H, W, 3) uint8 RGB; modified in-place
    cx_px: float, cy_px: float, theta_deg: float,
    color_rgb: tuple[int, int, int],
    dot_radius: int = 8,
    arrow_length: int = 30,
    thickness: int = 3,
) -> None:
    """Draw a pose marker: filled dot + arrow indicating angle.

    `canvas` is assumed to be in RGB order (the final RGB→BGR swap
    happens at cv2.imwrite time). cv2 draw functions write the literal
    color tuple into the array, so we pass color_rgb directly without
    a BGR swap — the swap would be wrong in this color space.
    """
    cv2.circle(canvas, (int(round(cx_px)), int(round(cy_px))),
               dot_radius, color_rgb, thickness=-1)
    rad = math.radians(theta_deg)
    dx = arrow_length * math.cos(rad)
    dy = arrow_length * math.sin(rad)
    cv2.arrowedLine(
        canvas,
        (int(round(cx_px)), int(round(cy_px))),
        (int(round(cx_px + dx)), int(round(cy_px + dy))),
        color_rgb, thickness=thickness, tipLength=0.32,
    )


def render_caption_strip(
    width: int, height: int,
    step: int, total_steps: int,
    reward: float, pos_dist: float, angle_err: float,
) -> np.ndarray:
    """Caption strip (dark background, white text). Returns (height, width, 3) RGB."""
    strip = np.full((height, width, 3), 30, dtype=np.uint8)
    txt = (
        f"step {step}/{total_steps}    "
        f"reward: {reward:+.4f}    "
        f"pos_dist: {pos_dist:.2f} px    "
        f"angle_err: {angle_err:+.2f} deg"
    )
    cv2.putText(strip, txt, (12, int(height * 0.66)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return strip


def angle_diff_deg(a: float, b: float) -> float:
    """Shortest signed angle (b - a) in [-180, 180]."""
    return ((b - a + 180) % 360) - 180


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--csv",   default=str(DEFAULT_CSV))
    ap.add_argument("--candidates", default=str(DEFAULT_CAND))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--display_size", type=int, default=512)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys_dir = out_dir / "keyframes"
    keys_dir.mkdir(parents=True, exist_ok=True)

    # ── load goal pose (easy candidate end_pose) ──
    with open(args.candidates) as f:
        easy = json.load(f)["picks"]["easy"]
    goal_cx_128 = float(easy["end_cx"])
    goal_cy_128 = float(easy["end_cy"])
    goal_theta = float(easy["end_theta"])
    print(f"[viz] goal (128² space): cx={goal_cx_128:.2f} cy={goal_cy_128:.2f} "
          f"theta={goal_theta:.2f}°")

    # ── load per-step CV pose + reward ──
    rows: list[dict[str, str]] = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    # CSV t=0 is warmup-init; t=1..50 are the 50 MPPI control steps.
    mppi_rows = [r for r in rows if int(r["t"]) >= 1]
    if len(mppi_rows) < N_STEPS:
        print(f"[viz] WARNING: only {len(mppi_rows)} MPPI rows; expected {N_STEPS}")
    mppi_rows = mppi_rows[:N_STEPS]

    # ── extract WM decoded frames from A.3 video ──
    print(f"[viz] extracting MPPI rows from {args.video}")
    frames_128 = extract_mppi_frames(Path(args.video), out_dir)
    print(f"[viz] extracted {len(frames_128)} frames (128x128)")
    assert len(frames_128) == N_STEPS, (
        f"expected {N_STEPS} mppi frames, got {len(frames_128)}"
    )

    # ── compute display-space goal pixel (scale 128² → display_size) ──
    target = int(args.display_size)
    scale = target / 128.0
    goal_cx_disp = goal_cx_128 * scale
    goal_cy_disp = goal_cy_128 * scale

    # ── render frames ──
    caption_h = 36
    total_h = target + caption_h
    individual_paths: list[Path] = []
    for k, (rgb_128, row) in enumerate(zip(frames_128, mppi_rows), start=1):
        # Upscale with INTER_CUBIC (the marker positions are in detector
        # 128² space, and bilinear/cubic resizing is what the WM produced).
        rgb_disp = cv2.resize(rgb_128, (target, target),
                              interpolation=cv2.INTER_CUBIC).copy()

        # Goal marker — red, fixed across all frames
        draw_pose_marker(rgb_disp, goal_cx_disp, goal_cy_disp, goal_theta,
                         color_rgb=(255, 0, 0))

        # Current pose — green, from per-row CSV
        cv_success = row["cv_success"] == "True"
        if cv_success:
            cx128 = float(row["cx"])
            cy128 = float(row["cy"])
            th = float(row["theta_deg"])
            draw_pose_marker(
                rgb_disp, cx128 * scale, cy128 * scale, th,
                color_rgb=(0, 255, 0),
            )
            dx = cx128 - goal_cx_128
            dy = cy128 - goal_cy_128
            pos_dist = math.sqrt(dx * dx + dy * dy)
            angle_err = angle_diff_deg(goal_theta, th)
        else:
            pos_dist = float("nan")
            angle_err = float("nan")

        # Caption strip
        cap_strip = render_caption_strip(
            target, caption_h, step=k, total_steps=N_STEPS,
            reward=float(row["reward"]),
            pos_dist=pos_dist if not math.isnan(pos_dist) else 0.0,
            angle_err=angle_err if not math.isnan(angle_err) else 0.0,
        )
        composite = np.concatenate([rgb_disp, cap_strip], axis=0)

        # Save (RGB → BGR for cv2.imwrite)
        out_path = out_dir / f"frame_{k - 1}.png"
        cv2.imwrite(str(out_path),
                    cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        individual_paths.append(out_path)

    print(f"[viz] wrote {len(individual_paths)} PNG frames")

    # ── keyframe copies ──
    for tag, idx in [("frame_0", 0), ("frame_25", 25), ("frame_49", 49)]:
        src = out_dir / f"{tag}.png"
        if src.exists():
            dst = keys_dir / f"{tag}.png"
            dst.write_bytes(src.read_bytes())
            print(f"[viz] keyframe: {dst}")

    # ── ffmpeg into mp4 ──
    mp4_path = out_dir / "closed_loop_with_goal.mp4"
    cmd = [
        "ffmpeg", "-y", "-framerate", "5",
        "-i", str(out_dir / "frame_%d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(mp4_path),
    ]
    print(f"[viz] running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[viz] ffmpeg stderr:\n{res.stderr[-800:]}")
        raise SystemExit(f"ffmpeg failed with code {res.returncode}")
    print(f"[viz] wrote {mp4_path}")


if __name__ == "__main__":
    main()
