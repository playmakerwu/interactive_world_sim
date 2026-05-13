"""Closed-loop world-model-only video with goal + current pose markers
and (optional) per-step top-2 future action trajectories overlaid.

Produces N PNG frames (one per MPPI control step) and an mp4 stitching
them at 5 fps. Layout:

  ┌─────────────────────────────────────┐
  │  decoded RGB from world model       │
  │  (128² upscaled to 512² INTER_CUBIC)│
  │                                     │
  │    ● ─→   (red, fixed goal pose)    │
  │       ● ─→  (green, current pose)   │
  │    ╲      (yellow,  top-1 sample)   │
  │     ╲     (orange,  top-2 sample)   │
  │  step k/N  reward: ...  ...         │
  └─────────────────────────────────────┘

Data sources:
  - <video>: 3-row driver-generated mp4 (middle row = WM-decoded closed-loop).
  - <csv>:   per-step CV pose + reward.
  - <candidates>: A.1 candidates.json — pick goal_pose from picks[<goal_pick>].
  - <run_dir>/debug/plan_<t:03d>/niter_<final>.npz:
      per-step (samples, sample_rewards) for trajectory overlay.
      Pass --run_dir to enable trajectory overlay; omit it for the original
      goal+current-only viz.
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
sys.path.insert(0, str(REPO))

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

# Top-K trajectory overlay: all N_TOP candidates drawn uniformly as thin
# anti-aliased red polylines with a dim-red → bright-red gradient along time,
# forming an "exploration cloud" rather than a highlighted-best + backdrop.
N_TOP = 8
UNIFORM_GRADIENT_START_RGB = (102, 0, 0)     # 40% red, t=0 (dim)
UNIFORM_GRADIENT_END_RGB   = (255, 0, 0)     # pure red, t=H-1 (vivid)
UNIFORM_OVERLAY_ALPHA      = 0.7             # blend weight of trajectory overlay
UNIFORM_LINE_WIDTH         = 1               # thinnest cv2 supports; rely on LINE_AA
UNIFORM_POINT_RADIUS       = 2


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


def world_xyz_to_pixel(
    world_xyz: np.ndarray,    # (M, 3) float64
    K: np.ndarray,            # (3, 3)
    cam_extrinsics: np.ndarray,  # (4, 4) world_t_cam, OpenCV convention
) -> np.ndarray:
    """Pinhole projection world_xyz -> (u, v) in original-camera-image space.

    Mirrors scripts/viz/action_sampling_viz.py::world_xyz_to_pixel.
    Returns (M, 2) float64; entries with Zc <= 0 are np.nan.
    """
    cam_t_world = np.linalg.inv(cam_extrinsics.astype(np.float64))
    K = K.astype(np.float64)
    out = np.full((world_xyz.shape[0], 2), np.nan, dtype=np.float64)
    for i, (X, Y, Z) in enumerate(world_xyz):
        p_world = np.array([X, Y, Z, 1.0], dtype=np.float64)
        p_cam = cam_t_world @ p_world
        Xc, Yc, Zc = p_cam[0], p_cam[1], p_cam[2]
        if Zc <= 0:
            continue
        out[i, 0] = K[0, 0] * Xc / Zc + K[0, 2]
        out[i, 1] = K[1, 1] * Yc / Zc + K[1, 2]
    return out


def camera_to_canvas_pixel(
    uv: np.ndarray,             # (M, 2) in original camera 480x640 space
    raw_h: int, raw_w: int,     # 480, 640
    target_size: int,           # 128 (decoder native) or 512 (display upscale)
) -> np.ndarray:
    """Apply the same center-crop + resize the WM preprocess pipeline does:
    480x640 -> center-crop 480x480 -> resize to (target_size, target_size).
    """
    s = min(raw_h, raw_w)         # 480
    crop_x = (raw_w - s) // 2     # 80
    crop_y = (raw_h - s) // 2     # 0
    u = (uv[:, 0] - crop_x) * (target_size / s)
    v = (uv[:, 1] - crop_y) * (target_size / s)
    return np.stack([u, v], axis=-1)


def draw_trajectory(
    canvas: np.ndarray,       # (canvas_h, canvas_w, 3) uint8 RGB
    pixels: np.ndarray,       # (H, 2) in canvas-pixel space; may contain NaN
    start_color_rgb: tuple[int, int, int],
    end_color_rgb: tuple[int, int, int],
    line_width: int = 2,
    point_radius: int = 4,
) -> None:
    """Draw a polyline with a linear color gradient from start → end along time.

    Point at t=0 uses `start_color_rgb`; point at t=H-1 uses `end_color_rgb`;
    intermediate points are linearly interpolated. NaN waypoints (off-frame
    after pinhole projection) are skipped, and the line to the previous
    point is also skipped at that boundary.
    """
    H = pixels.shape[0]
    prev_valid = False
    prev_xy: tuple[int, int] | None = None
    for t in range(H):
        x, y = pixels[t, 0], pixels[t, 1]
        if not (np.isfinite(x) and np.isfinite(y)):
            prev_valid = False
            prev_xy = None
            continue
        u = t / max(H - 1, 1)
        color = tuple(
            int(round(start_color_rgb[c]
                      + u * (end_color_rgb[c] - start_color_rgb[c])))
            for c in range(3)
        )
        ix, iy = int(round(x)), int(round(y))
        if prev_valid and prev_xy is not None:
            cv2.line(canvas, prev_xy, (ix, iy), color,
                     thickness=line_width, lineType=cv2.LINE_AA)
        prev_xy = (ix, iy)
        prev_valid = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--csv",   default=str(DEFAULT_CSV))
    ap.add_argument("--candidates", default=str(DEFAULT_CAND))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--display_size", type=int, default=512)
    # New: trajectory overlay
    ap.add_argument(
        "--run_dir",
        default="",
        help="A3_run directory with debug/plan_*/niter_*.npz. "
             "If empty, the trajectory overlay is skipped.",
    )
    ap.add_argument(
        "--goal_pick",
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Which entry under picks{} to use for the goal pose.",
    )
    ap.add_argument(
        "--episode",
        default="",
        help="HDF5 episode used as the warmup start frame for action "
             "projection. Defaults to candidates.json picks[<goal_pick>].episode.",
    )
    ap.add_argument(
        "--start_frame",
        type=int,
        default=-1,
        help="Frame index in --episode at which the warmup begins. "
             "Defaults to candidates.json picks[<goal_pick>].N.",
    )
    ap.add_argument(
        "--wm_ckpt",
        default=str(REPO / "outputs/pusht_cam1/checkpoints/best.ckpt"),
        help="World-model checkpoint (only its action normalizer is used).",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys_dir = out_dir / "keyframes"
    keys_dir.mkdir(parents=True, exist_ok=True)

    # ── load goal pose ──
    with open(args.candidates) as f:
        cand = json.load(f)
    pick = cand["picks"][args.goal_pick]
    goal_cx_128 = float(pick["end_cx"])
    goal_cy_128 = float(pick["end_cy"])
    goal_theta = float(pick["end_theta"])
    print(f"[viz] goal_pick={args.goal_pick}: "
          f"cx={goal_cx_128:.2f} cy={goal_cy_128:.2f} theta={goal_theta:.2f}°")

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

    # ── optional: prepare trajectory-overlay projection metadata ──
    overlay_enabled = bool(args.run_dir)
    plans_by_step: dict[int, dict[str, np.ndarray]] = {}
    K_cam = ext = None
    Z_LEFT = 0.0
    Z_RIGHT = 0.0
    raw_h = raw_w = 0
    action_norm = None
    if overlay_enabled:
        run_dir = Path(args.run_dir)
        debug_dir = run_dir / "debug"
        if not debug_dir.is_dir():
            raise SystemExit(f"--run_dir/debug does not exist: {debug_dir}")

        # ── load camera meta + ee_pos at the warmup start frame ──
        import h5py  # noqa: E402  (lazy: only needed when overlay is on)
        import torch  # noqa: E402

        episode_path = args.episode or str(REPO / pick["episode"])
        start_frame = args.start_frame if args.start_frame >= 0 else int(pick["N"])
        print(f"[viz] episode={episode_path} start_frame={start_frame}")
        with h5py.File(episode_path, "r") as f:
            K_cam = np.asarray(
                f["obs/images/camera_1_intrinsics"][start_frame], dtype=np.float64,
            )
            ext = np.asarray(
                f["obs/images/camera_1_extrinsics"][start_frame], dtype=np.float64,
            )
            ee_pos = np.asarray(f["obs/ee_pos"][start_frame], dtype=np.float64)
            base = np.asarray(
                f["obs/world_t_robot_base"][start_frame], dtype=np.float64,
            )
            raw_h, raw_w = 480, 640
        base_left = base[0]   # (4, 4)
        # Both arms transform through base[0]: ee_pos[0:3] is left arm in
        # LEFT robot frame; ee_pos[7:10] is right arm, also in LEFT robot
        # frame (yiru's verified convention — see helpers/projection.py).
        left_world_xyz = (
            base_left @ np.array([ee_pos[0], ee_pos[1], ee_pos[2], 1.0])
        )[:3]
        right_world_xyz = (
            base_left @ np.array([ee_pos[7], ee_pos[8], ee_pos[9], 1.0])
        )[:3]
        Z_LEFT = float(left_world_xyz[2])
        Z_RIGHT = float(right_world_xyz[2])
        print(f"[viz] left  gripper Z (world): {Z_LEFT:.4f} m")
        print(f"[viz] right gripper Z (world): {Z_RIGHT:.4f} m")

        # ── PushTWMEnv (only its action normalizer is used) ──
        from env.pusht_wm_env import PushTWMEnv  # noqa: E402
        env = PushTWMEnv(args.wm_ckpt, device="cuda:0")
        action_norm = env._wm.normalizer["action"]

        # ── enumerate niter files per plan; pick the final one ──
        for t in range(N_STEPS):
            plan_dir = debug_dir / f"plan_{t:03d}"
            if not plan_dir.is_dir():
                print(f"[viz] missing plan dir for step {t}; overlay skipped")
                continue
            niter_files = sorted(plan_dir.glob("niter_*.npz"))
            if not niter_files:
                print(f"[viz] no niter NPZ in {plan_dir}; overlay skipped")
                continue
            data = np.load(niter_files[-1])
            plans_by_step[t] = {
                "samples": data["samples"],            # (K, H, 4)
                "sample_rewards": data["sample_rewards"],  # (K,)
            }
        print(f"[viz] loaded {len(plans_by_step)} plan NPZ for trajectory overlay")

    # ── render frames ──
    caption_h = 36
    individual_paths: list[Path] = []
    for k, (rgb_128, row) in enumerate(zip(frames_128, mppi_rows), start=1):
        # Upscale with INTER_CUBIC (the marker positions are in detector
        # 128² space, and bilinear/cubic resizing is what the WM produced).
        rgb_disp = cv2.resize(rgb_128, (target, target),
                              interpolation=cv2.INTER_CUBIC).copy()

        # ── top-2 trajectory overlay (drawn UNDER the markers) ──
        # CSV row "t" matches plan_<t-1> because plans are indexed 0..N-1
        # and produce the action that yields CSV t=1..N.
        plan_idx = k - 1
        plan = plans_by_step.get(plan_idx) if overlay_enabled else None
        if plan is not None and action_norm is not None:
            import torch  # noqa: E402
            samples = plan["samples"]           # (K, H, 4) float32
            rewards = plan["sample_rewards"]    # (K,)
            n_samples, horizon, _ = samples.shape
            order = np.argsort(-rewards)
            n_top = min(N_TOP, n_samples)
            top_idx = order[:n_top]
            sel = samples[top_idx]              # (n_top, H, 4)
            # Unnormalize once over all top-K * H rows.
            flat = torch.from_numpy(sel.reshape(n_top * horizon, 4))
            un = action_norm.unnormalize(flat).cpu().numpy()  # (n_top*H, 4)
            un = un.reshape(n_top, horizon, 4)

            # Left arm: action dims 0:2 + per-arm Z.
            left_xy = un[:, :, 0:2].reshape(n_top * horizon, 2)
            left_xyz = np.concatenate(
                [left_xy, np.full((n_top * horizon, 1), Z_LEFT)], axis=1,
            )
            uv_left = world_xyz_to_pixel(left_xyz, K_cam, ext)
            pix_left = camera_to_canvas_pixel(uv_left, raw_h, raw_w, target)
            pixels_left_per_top = pix_left.reshape(n_top, horizon, 2)

            # Right arm: action dims 2:4 + per-arm Z.
            right_xy = un[:, :, 2:4].reshape(n_top * horizon, 2)
            right_xyz = np.concatenate(
                [right_xy, np.full((n_top * horizon, 1), Z_RIGHT)], axis=1,
            )
            uv_right = world_xyz_to_pixel(right_xyz, K_cam, ext)
            pix_right = camera_to_canvas_pixel(uv_right, raw_h, raw_w, target)
            pixels_right_per_top = pix_right.reshape(n_top, horizon, 2)

            # Visual-only offset; raises trajectories above gripper region for clarity
            pixels_left_per_top -= 10
            pixels_right_per_top -= 10

            # Draw all n_top trajectories (both arms each) on a single overlay
            # buffer in a uniform dim-red → bright-red gradient, then blend
            # back at UNIFORM_OVERLAY_ALPHA so the cloud sits faintly over the
            # decoded RGB. No best/others distinction — all 8 look identical.
            overlay = rgb_disp.copy()
            for slot in range(n_top):
                draw_trajectory(
                    overlay, pixels_left_per_top[slot],
                    start_color_rgb=UNIFORM_GRADIENT_START_RGB,
                    end_color_rgb=UNIFORM_GRADIENT_END_RGB,
                    line_width=UNIFORM_LINE_WIDTH,
                    point_radius=UNIFORM_POINT_RADIUS,
                )
                draw_trajectory(
                    overlay, pixels_right_per_top[slot],
                    start_color_rgb=UNIFORM_GRADIENT_START_RGB,
                    end_color_rgb=UNIFORM_GRADIENT_END_RGB,
                    line_width=UNIFORM_LINE_WIDTH,
                    point_radius=UNIFORM_POINT_RADIUS,
                )
            rgb_disp = cv2.addWeighted(
                overlay, UNIFORM_OVERLAY_ALPHA,
                rgb_disp, 1.0 - UNIFORM_OVERLAY_ALPHA, 0.0,
            )

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
