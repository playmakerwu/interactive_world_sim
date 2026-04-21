"""Build a polished showcase MP4 from an existing MPPI run.

Reads `trajectory_overlay.mp4` (128x128) and `summary.json` from a run
directory, then writes a new `showcase.mp4` that is:

  - Upscaled 4x (512x512 per panel)
  - Paired side-by-side with a reward curve that grows frame-by-frame
  - Lower fps (3) so each control step is readable
  - Labeled with step counter, reward, and final-success flags in a banner

No WM re-runs needed — this is pure post-processing on saved artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PANEL_SIZE = 512           # each side is 512x512 in the showcase
FPS = 3                    # slow enough to read
BANNER_H = 70


def _load_frames(mp4_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _render_reward_curve(rewards_so_far: list[float], total_steps: int,
                         size_px: int) -> np.ndarray:
    """Render a matplotlib plot of rewards[0..t] to a (size, size, 3) RGB array."""
    fig, ax = plt.subplots(figsize=(size_px / 100, size_px / 100), dpi=100)
    xs = list(range(len(rewards_so_far)))
    ax.plot(xs, rewards_so_far, marker="o", color="tab:blue", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlim(-0.5, total_steps - 0.5)
    # Y range covers both sym-aware ([-1, 0]) and original ([-10, 0]) scales.
    ymin = min(-1.1, min(rewards_so_far) - 0.1)
    ax.set_ylim(ymin, 0.15)
    ax.set_xlabel("control step t", fontsize=11)
    ax.set_ylabel("reward(z_t, state_goal)", fontsize=11)
    ax.set_title(f"reward so far  (t={len(rewards_so_far) - 1})", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return arr


def _make_banner(summary: dict, W: int, H: int) -> np.ndarray:
    banner = np.full((H, W, 3), 240, dtype=np.uint8)
    cfg = summary["config"]
    sym = "symmetry-aware" if cfg.get("symmetry_aware") else "original"
    line1 = (
        f"{summary['run_name']}   "
        f"reward={sym}   N={cfg['N']}  H={cfg['H']}  "
        f"sigma={cfg['sigma']}   init={cfg['initial_state']}"
    )
    gx = summary["goal_state"]["cx"]
    gy = summary["goal_state"]["cy"]
    gth = summary["goal_state"]["theta_deg"]
    line2 = (
        f"goal=({gx:.1f}, {gy:.1f}, {gth:+.1f}deg)   "
        f"final_pos={summary['final_pos_distance_px']:.2f} px   "
        f"final_|dTheta|={abs(summary['final_angle_error_deg']):.1f} deg   "
        f"success={summary['success_strict']}"
    )
    cv2.putText(banner, line1, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(banner, line2, (12, 54), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (40, 40, 40), 1, cv2.LINE_AA)
    return banner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        default="outputs/mppi/step5_v2_symaware_axisaligned",
        help="MPPI run directory (contains trajectory_overlay.mp4 and summary.json)",
    )
    ap.add_argument(
        "--out_name", default="showcase.mp4",
        help="Output filename written inside run_dir",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    overlay_path = run_dir / "trajectory_overlay.mp4"
    summary_path = run_dir / "summary.json"
    assert overlay_path.exists(), overlay_path
    assert summary_path.exists(), summary_path

    frames = _load_frames(overlay_path)
    summary = json.loads(summary_path.read_text())
    per_step = summary["per_step"]
    assert len(frames) == len(per_step), (
        f"{len(frames)} frames vs {len(per_step)} recorded steps"
    )
    total = len(frames)
    rewards = [row["reward"] for row in per_step]

    W = PANEL_SIZE * 2
    H = PANEL_SIZE + BANNER_H
    banner = _make_banner(summary, W, BANNER_H)

    out_path = run_dir / args.out_name
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, FPS, (W, H))

    print(f"Rendering {total} frames -> {out_path}")
    try:
        for i, frame in enumerate(frames):
            left = cv2.resize(
                frame, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST
            )
            right = _render_reward_curve(rewards[: i + 1], total, PANEL_SIZE)

            canvas = np.full((H, W, 3), 240, dtype=np.uint8)
            canvas[:BANNER_H] = banner
            canvas[BANNER_H : BANNER_H + PANEL_SIZE, :PANEL_SIZE] = left
            canvas[BANNER_H : BANNER_H + PANEL_SIZE, PANEL_SIZE:] = right

            # Per-frame step/reward caption over the trajectory panel.
            txt = f"t={per_step[i]['t']:02d}   r={per_step[i]['reward']:+.3f}"
            cv2.putText(
                canvas, txt, (16, BANNER_H + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA,
            )

            vw.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            if i % 10 == 0:
                print(f"  frame {i + 1}/{total}")
    finally:
        vw.release()
    print(f"wrote: {out_path}   ({total} frames at {FPS} fps = "
          f"{total / FPS:.1f} s)")


if __name__ == "__main__":
    main()
