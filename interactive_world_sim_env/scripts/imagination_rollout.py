"""Imagination rollout test.

Warmup with 10 real frames, then run H pure-imagination steps driven
by the expert's actions only — no further real RGB is ever fed back
into the env after the warmup. Compare each predicted frame to the
real episode's frame at the same time index, and save:

- /tmp/imagination_rollout.mp4 (or .gif if ffmpeg is unavailable)
- /tmp/imagination_rollout_grid.png    (6 snapshots, predicted vs GT)
- /tmp/imagination_rollout_drift.png   (normalized L2 vs time)

Run from the repo root:

    python interactive_world_sim_env/scripts/imagination_rollout.py
    python interactive_world_sim_env/scripts/imagination_rollout.py \
        --horizon 50 --no-circles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import h5py
import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")  # noqa: E402  must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from interactive_world_sim_env import WorldModelEnv  # noqa: E402
from interactive_world_sim_env.helpers.expert_action import (  # noqa: E402
    expert_action_from_episode,
)
from interactive_world_sim_env.helpers.projection import (  # noqa: E402
    project_expert_grippers,
)

EPISODE = str(_REPO_ROOT / "data" / "mini" / "pusht" / "val" / "episode_0.hdf5")
RES = 128
DISPLAY = 384
CAM_KEY = "camera_1"
WARMUP_W = 10
WARMUP_END_FRAME = 9  # last frame in the warmup window

LEFT_COLOR_RGB = (255, 0, 0)
RIGHT_COLOR_RGB = (0, 0, 255)
TEXT_COLOR_RGB = (0, 255, 0)
DRIFT_THRESHOLD = 0.1

VIDEO_PATH_MP4 = "/tmp/imagination_rollout.mp4"
VIDEO_PATH_GIF = "/tmp/imagination_rollout.gif"
GRID_PATH = "/tmp/imagination_rollout_grid.png"
DRIFT_PATH = "/tmp/imagination_rollout_drift.png"


# ----------------------------------------------------------------- helpers


def _real_to_decoded_pixel(u_real: float, v_real: float) -> tuple[float, float]:
    """640x480 → 128x128 after env's center-crop (480x480) + uniform resize."""
    scale = RES / 480.0
    return (u_real - 80.0) * scale, v_real * scale


def _draw_circles_128(img128: np.ndarray, pixels_640: np.ndarray) -> np.ndarray:
    img = img128.copy()
    (u_l, v_l), (u_r, v_r) = pixels_640
    u_ld, v_ld = _real_to_decoded_pixel(u_l, v_l)
    u_rd, v_rd = _real_to_decoded_pixel(u_r, v_r)
    if np.isfinite(u_ld) and np.isfinite(v_ld):
        cv2.circle(
            img,
            (int(round(u_ld)), int(round(v_ld))),
            radius=3,
            color=LEFT_COLOR_RGB,
            thickness=1,
        )
    if np.isfinite(u_rd) and np.isfinite(v_rd):
        cv2.circle(
            img,
            (int(round(u_rd)), int(round(v_rd))),
            radius=3,
            color=RIGHT_COLOR_RGB,
            thickness=1,
        )
    return img


def _upscale_nn(img: np.ndarray, target: int) -> np.ndarray:
    return cv2.resize(img, (target, target), interpolation=cv2.INTER_NEAREST)


def _normalized_l2(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    """RMSE on [0, 1] images. 0 = identical, 1.0 = fully saturated."""
    pred = pred_u8.astype(np.float32) / 255.0
    gt = gt_u8.astype(np.float32) / 255.0
    return float(np.sqrt(((pred - gt) ** 2).mean()))


def _compose_video_frame(
    pred_128: np.ndarray,
    gt_128: np.ndarray,
    t: int,
    l2: float,
) -> np.ndarray:
    """Side-by-side (PREDICTED | GROUND TRUTH) at 384x384 each, with labels."""
    pred_up = _upscale_nn(pred_128, DISPLAY)
    gt_up = _upscale_nn(gt_128, DISPLAY)
    title_h = 30
    label_h = 30
    H = title_h + DISPLAY + label_h
    W = DISPLAY * 2
    frame = np.full((H, W, 3), 32, dtype=np.uint8)
    frame[title_h : title_h + DISPLAY, 0:DISPLAY] = pred_up
    frame[title_h : title_h + DISPLAY, DISPLAY : 2 * DISPLAY] = gt_up
    cv2.putText(
        frame, f"PREDICTED (t={t})", (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR_RGB, 2,
    )
    cv2.putText(
        frame, f"GROUND TRUTH (t={t})", (DISPLAY + 10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR_RGB, 2,
    )
    cv2.putText(
        frame, f"L2 (norm RMSE) = {l2:.4f}", (10, title_h + DISPLAY + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR_RGB, 2,
    )
    return frame


def _save_video(frames: list[np.ndarray]) -> tuple[str, str]:
    """Try MP4 with libx264; fall back to GIF at the same effective rate."""
    arr = np.stack(frames)
    try:
        iio.imwrite(VIDEO_PATH_MP4, arr, fps=10, codec="libx264")
        return VIDEO_PATH_MP4, "mp4"
    except Exception as e:
        print(f"  MP4 write failed ({e!r}); falling back to GIF.")
        iio.imwrite(VIDEO_PATH_GIF, arr, duration=100, loop=0)
        return VIDEO_PATH_GIF, "gif"


def _save_drift(l2_history: list[float], start_t: int) -> None:
    """Line plot of normalized L2 over time. Log y if >2 orders of range."""
    xs = list(range(start_t, start_t + len(l2_history)))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, l2_history, marker="o", markersize=2, linewidth=1.2)
    ax.axhline(DRIFT_THRESHOLD, color="orange", linestyle="--", alpha=0.5,
               label=f"threshold {DRIFT_THRESHOLD}")
    ax.set_xlabel("episode frame index t")
    ax.set_ylabel("L2 distance (normalized RMSE)")
    ax.set_title(f"Imagination rollout drift — pusht_cam1, horizon={len(l2_history)}")
    lo = max(min(l2_history), 1e-6)
    hi = max(l2_history)
    if hi / lo > 100:
        ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(DRIFT_PATH, dpi=100)
    plt.close(fig)


def _grid_snapshot_ts(horizon: int) -> list[int]:
    if horizon == 100:
        return [10, 20, 40, 60, 80, 100]
    return [
        int(x) for x in np.linspace(WARMUP_END_FRAME + 1, WARMUP_END_FRAME + horizon, 6)
    ]


# ----------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=100, help="Number of imagination steps.")
    parser.add_argument(
        "--circles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw expert-action projection circles on both rows.",
    )
    args = parser.parse_args()
    horizon = int(args.horizon)
    draw_circles = bool(args.circles)
    if horizon < 1:
        parser.error("--horizon must be >= 1")

    print(
        f"horizon={horizon}  circles={draw_circles}  episode={EPISODE}  "
        f"warmup window={WARMUP_W}"
    )

    env = WorldModelEnv("pusht_cam1")
    env.reset(
        init_episode_path=EPISODE,
        init_episode_index=WARMUP_END_FRAME,
        init_window_size=WARMUP_W,
    )

    grid_target_ts = _grid_snapshot_ts(horizon)
    grid_pred: dict[int, np.ndarray] = {}
    grid_gt: dict[int, np.ndarray] = {}

    frames: list[np.ndarray] = []
    l2_history: list[float] = []

    with h5py.File(EPISODE, "r") as f:
        for offset in range(horizon):
            t = WARMUP_END_FRAME + 1 + offset  # 10, 11, ...

            # Expert action that drove the demonstrator from frame t-1 to t.
            action_t = expert_action_from_episode(env, EPISODE, t - 1)
            env.step(action_t)
            pred_128 = env.render()  # (128, 128, 3) uint8

            # Ground truth: real frame at index t, preprocessed the same way
            # the env preprocesses inputs internally.
            raw_gt = np.asarray(f[f"obs/images/{CAM_KEY}_color"][t])
            gt_chw01 = env.goal_preprocess(raw_gt)  # (3, 128, 128) float32 in [0,1]
            gt_128 = (np.transpose(gt_chw01, (1, 2, 0)) * 255.0).astype(np.uint8)

            l2 = _normalized_l2(pred_128, gt_128)
            l2_history.append(l2)

            if draw_circles:
                proj = project_expert_grippers(EPISODE, t, cam_key=CAM_KEY)
                pixels_640 = proj["pixels"]
                pred_disp = _draw_circles_128(pred_128, pixels_640)
                gt_disp = _draw_circles_128(gt_128, pixels_640)
            else:
                pred_disp = pred_128
                gt_disp = gt_128

            frames.append(_compose_video_frame(pred_disp, gt_disp, t, l2))

            if t in grid_target_ts:
                grid_pred[t] = _upscale_nn(pred_disp, DISPLAY)
                grid_gt[t] = _upscale_nn(gt_disp, DISPLAY)

            if (offset + 1) % 10 == 0 or offset == horizon - 1:
                print(
                    f"  step {offset+1:>3d}/{horizon}  t={t:>3d}  l2={l2:.4f}"
                )

    env.close()

    # ---------- write outputs ----------
    print("\nwriting outputs...")
    video_path, fmt = _save_video(frames)
    print(f"  {fmt.upper():<3s} : {video_path}  ({len(frames)} frames @ 10 fps)")

    chosen = [t for t in grid_target_ts if t in grid_pred]
    if len(chosen) == 6:
        row_pred = np.concatenate([grid_pred[t] for t in chosen], axis=1)
        row_gt = np.concatenate([grid_gt[t] for t in chosen], axis=1)
        # Add a label band on top with each timestep
        label_h = 30
        n_cols = len(chosen)
        labels = np.full((label_h, n_cols * DISPLAY, 3), 16, dtype=np.uint8)
        for i, t in enumerate(chosen):
            cv2.putText(
                labels, f"t={t}",
                (i * DISPLAY + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR_RGB, 2,
            )
        # also a label between rows ("PREDICTED" / "GROUND TRUTH")
        gap = np.full((label_h, n_cols * DISPLAY, 3), 16, dtype=np.uint8)
        cv2.putText(gap, "GROUND TRUTH", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR_RGB, 2)
        pred_label = np.full((label_h, n_cols * DISPLAY, 3), 16, dtype=np.uint8)
        cv2.putText(pred_label, "PREDICTED", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR_RGB, 2)
        grid = np.concatenate([labels, pred_label, row_pred, gap, row_gt], axis=0)
        iio.imwrite(GRID_PATH, grid)
        print(f"  PNG : {GRID_PATH}  shape={grid.shape}  (snapshots at t={chosen})")
    else:
        print(
            f"  PNG : skipped — only {len(chosen)} of 6 requested snapshot frames "
            f"happened to land in the rollout (requested {grid_target_ts})"
        )

    _save_drift(l2_history, start_t=WARMUP_END_FRAME + 1)
    print(f"  PNG : {DRIFT_PATH}")

    # ---------- numeric report ----------
    print("\n========== summary ==========")
    arr = np.asarray(l2_history)
    print(f"  mean L2:               {arr.mean():.4f}")
    print(f"  median L2:             {float(np.median(arr)):.4f}")
    print(f"  min L2:                {arr.min():.4f}  (at t={WARMUP_END_FRAME + 1 + int(arr.argmin())})")
    print(f"  max L2:                {arr.max():.4f}  (at t={WARMUP_END_FRAME + 1 + int(arr.argmax())})")
    print()
    sample_ts = [10, 25, 50, 100]
    for tt in sample_ts:
        offset_idx = tt - (WARMUP_END_FRAME + 1)
        if 0 <= offset_idx < len(l2_history):
            print(f"  L2 at t={tt:>3d}:           {l2_history[offset_idx]:.4f}")
    print()
    first_over = next(
        (i for i, x in enumerate(l2_history) if x > DRIFT_THRESHOLD), None
    )
    if first_over is None:
        print(
            f"  L2 never exceeded {DRIFT_THRESHOLD} threshold over {horizon} frames."
        )
    else:
        first_t = WARMUP_END_FRAME + 1 + first_over
        print(
            f"  L2 first exceeded {DRIFT_THRESHOLD} at t={first_t} "
            f"(offset {first_over}, value {l2_history[first_over]:.4f})."
        )
    print()
    print("  Visual collapse: not auto-detected. Inspect the grid PNG and the")
    print("  drift curve to judge; large sustained jumps in L2 are the signal.")
    print()
    print("  CAVEATS:")
    print("    1. Decoder is stochastic — every render() draws fresh noise.")
    print("       L2 numbers are from a SINGLE rollout, not averaged.")
    print("    2. Expert action via expert_action_from_episode is the 4 mm-")
    print("       approx (no FK / no workspace clip). After normalization the")
    print("       residual was 0.01-0.02; small contribution to early drift.")
    print(f"    3. After step {WARMUP_W} the latent_window holds entirely")
    print("       model-predicted latents (no real frames left). This is the")
    print("       pure-imagination regime that MPPI scoring would see.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
