"""Action sampling iteration visualization for one MPPI plan() call.

Produces 5 PNG frames (one per niter) and an mp4 stitching them at
5 fps. Side-by-side layout per frame:

  ┌─────────────────────────┬─────────────────────────┐
  │ start-frame RGB (128²   │ best-reward-per-niter   │
  │  upscaled to 512²) with │ line plot, niter 0..4   │
  │  top-4 sampled gripper  │                         │
  │  trajectories overlaid  │                         │
  └─────────────────────────┴─────────────────────────┘

Each trajectory has 10 waypoints (the H=10 horizon). Color gradient
light→dark within a trajectory; 4 distinct base hues across trajs.

Pixel projection: actions are 4-dim normalized [-1, +1]. First two
dims (left arm world XY in meters after inverse-normalize) plus the
gripper Z height from the start frame's ee_pos[2] world coord, then
pinhole project through camera_1 intrinsics + extrinsics. Crop+resize
to map original 480² center-cropped camera coords to the 128² space
the decoder operates in.

Data source: audit_log.json from A.3
(overnight/two_fixes_2026_05_12_1438/A3_run/audit_log.json) — the
plan_call_idx==0 entries hold per-niter (16, 10, 4) samples and
(16,) sample_rewards. No MPPI re-run needed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from env.pusht_wm_env import PushTWMEnv  # noqa: E402

EPISODE = REPO / "data/mini/pusht/train/episode_4.hdf5"
START_FRAME = 32
H = 10
N = 16
NITER = 5
TOP_K = 4
COLORS = ["tab:blue", "tab:green", "tab:orange", "tab:purple"]


def world_xyz_to_pixel(
    world_xyz: np.ndarray,    # (M, 3) float64
    K: np.ndarray,            # (3, 3)
    cam_extrinsics: np.ndarray,  # (4, 4) world_t_cam, OpenCV convention
) -> np.ndarray:
    """Pinhole projection world_xyz -> (u, v) in original-camera-image space.

    Copied verbatim from yiru's interactive_world_sim_env/helpers/
    projection.py::world_xy_to_pixel pattern (which itself follows the
    OpenCV convention). Returns (M, 2) float64.
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
    480x640 -> center-crop 480x480 -> resize to (target_size, target_size)."""
    s = min(raw_h, raw_w)         # 480
    crop_x = (raw_w - s) // 2     # 80
    crop_y = (raw_h - s) // 2     # 0
    u = (uv[:, 0] - crop_x) * (target_size / s)
    v = (uv[:, 1] - crop_y) * (target_size / s)
    return np.stack([u, v], axis=-1)


def preprocess_rgb_to_128(raw: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> 128x128 uint8, matching env preprocess."""
    h, w = raw.shape[:2]
    s = min(h, w)
    cropped = raw[(h - s) // 2 : (h - s) // 2 + s,
                  (w - s) // 2 : (w - s) // 2 + s]
    return cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)


def draw_trajectory(
    canvas: np.ndarray,       # (canvas_h, canvas_w, 3) uint8 RGB
    pixels: np.ndarray,       # (H, 2) in canvas-pixel space
    base_color_rgb: tuple[int, int, int],
    line_width: int = 2,
    point_radius: int = 4,
) -> None:
    """Draw a 10-waypoint polyline with light-to-dark gradient WITHIN
    the trajectory, base hue from `base_color_rgb`.

    Canvas is in RGB order (RGB→BGR swap happens at imwrite). cv2 draw
    functions write the literal color tuple, so we pass RGB directly.
    """
    H = pixels.shape[0]
    for t in range(H):
        # Fade from 50% to 100% of the base color
        alpha = 0.4 + 0.6 * (t / max(H - 1, 1))
        color = tuple(int(c * alpha) for c in base_color_rgb)
        cv2.circle(canvas, (int(round(pixels[t, 0])), int(round(pixels[t, 1]))),
                   point_radius, color, thickness=-1)
        if t > 0:
            cv2.line(canvas,
                     (int(round(pixels[t - 1, 0])), int(round(pixels[t - 1, 1]))),
                     (int(round(pixels[t, 0])), int(round(pixels[t, 1]))),
                     color, thickness=line_width)


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    s = h.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


# Matplotlib's tab10 mapped to RGB
TAB_RGB = {
    "tab:blue":   hex_to_rgb("#1f77b4"),
    "tab:green":  hex_to_rgb("#2ca02c"),
    "tab:orange": hex_to_rgb("#ff7f0e"),
    "tab:purple": hex_to_rgb("#9467bd"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audit_log",
        default=str(REPO / "overnight/two_fixes_2026_05_12_1438/A3_run/audit_log.json"),
    )
    ap.add_argument(
        "--output_dir",
        default=str(REPO / "overnight/two_fixes_2026_05_12_1438/viz_action_sampling"),
    )
    ap.add_argument("--display_size", type=int, default=512,
                    help="Upscale the 128² decoded canvas to this size for display.")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load audit log; extract plan_0 niter samples + rewards ──
    with open(args.audit_log) as f:
        audit = json.load(f)
    plan0_inloop = [e for e in audit
                    if e["plan_call_idx"] == 0 and e["niter"] >= 0]
    assert len(plan0_inloop) == NITER, (
        f"expected {NITER} niter entries; got {len(plan0_inloop)}"
    )
    samples_by_niter = {
        e["niter"]: np.asarray(e["samples"], dtype=np.float32)
        for e in plan0_inloop
    }
    rewards_by_niter = {
        e["niter"]: np.asarray(e["sample_rewards"], dtype=np.float32)
        for e in plan0_inloop
    }
    # Sanity
    for n in range(NITER):
        assert samples_by_niter[n].shape == (N, H, 4), (
            f"niter {n} samples shape {samples_by_niter[n].shape} != ({N}, {H}, 4)"
        )

    # Best reward per niter (for the right-side plot).
    best_reward_per_niter = np.array(
        [rewards_by_niter[n].max() for n in range(NITER)]
    )

    # ── Load camera meta + ee_pos at the start frame ──
    with h5py.File(EPISODE, "r") as f:
        raw_start = f["obs/images/camera_1_color"][START_FRAME]
        K_cam = np.asarray(f["obs/images/camera_1_intrinsics"][START_FRAME], dtype=np.float64)
        ext = np.asarray(f["obs/images/camera_1_extrinsics"][START_FRAME], dtype=np.float64)
        ee_pos = np.asarray(f["obs/ee_pos"][START_FRAME], dtype=np.float64)
        base = np.asarray(f["obs/world_t_robot_base"][START_FRAME], dtype=np.float64)
    print(f"[viz] start frame raw shape: {raw_start.shape}")

    # Gripper Z in WORLD frame. ee_pos[0:3] is left arm in LEFT robot
    # frame; ee_pos[7:10] is right arm, also in LEFT robot's frame
    # (yiru's verified convention — see helpers/projection.py). So
    # both arms transform through base[0].
    base_left = base[0]   # (4, 4)
    left_world_xyz = (base_left @ np.array([ee_pos[0], ee_pos[1], ee_pos[2], 1.0]))[:3]
    right_world_xyz = (base_left @ np.array([ee_pos[7], ee_pos[8], ee_pos[9], 1.0]))[:3]
    Z_LEFT = float(left_world_xyz[2])
    Z_RIGHT = float(right_world_xyz[2])
    print(f"[viz] left  gripper Z (world): {Z_LEFT:.4f} m")
    print(f"[viz] right gripper Z (world): {Z_RIGHT:.4f} m")

    # ── Load PushTWMEnv just to access its action normalizer for unnormalize ──
    # Cheap: only the normalizer is used.
    env = PushTWMEnv(
        str(REPO / "outputs/pusht_cam1/checkpoints/best.ckpt"),
        device="cuda:0",
    )
    action_norm = env._wm.normalizer["action"]

    # ── Project all niters' samples to canvas pixels (display_size space) ──
    # Both arms are projected. The action vector is
    # [left_x, left_y, right_x, right_y] in normalized [-1, +1] world XY.
    # We attach the constant per-arm world Z from frame_32's ee_pos and
    # pinhole-project.
    raw_h, raw_w = raw_start.shape[:2]   # 480, 640
    target = int(args.display_size)
    pixels_left_by_niter: dict[int, np.ndarray] = {}    # (N, H, 2) per niter
    pixels_right_by_niter: dict[int, np.ndarray] = {}   # (N, H, 2) per niter
    for n in range(NITER):
        s = samples_by_niter[n]            # (N, H, 4) normalized
        flat = torch.from_numpy(s.reshape(N * H, 4))
        un = action_norm.unnormalize(flat).cpu().numpy()   # (N*H, 4) raw world
        un = un.reshape(N, H, 4)

        # Left arm
        left_xy = un[:, :, 0:2].reshape(N * H, 2)
        left_xyz = np.concatenate(
            [left_xy, np.full((N * H, 1), Z_LEFT)], axis=1
        )
        uv_left = world_xyz_to_pixel(left_xyz, K_cam, ext)
        canvas_left = camera_to_canvas_pixel(uv_left, raw_h, raw_w, target)
        pixels_left_by_niter[n] = canvas_left.reshape(N, H, 2)

        # Right arm — action dims 2:4 + per-arm Z
        right_xy = un[:, :, 2:4].reshape(N * H, 2)
        right_xyz = np.concatenate(
            [right_xy, np.full((N * H, 1), Z_RIGHT)], axis=1
        )
        uv_right = world_xyz_to_pixel(right_xyz, K_cam, ext)
        canvas_right = camera_to_canvas_pixel(uv_right, raw_h, raw_w, target)
        pixels_right_by_niter[n] = canvas_right.reshape(N, H, 2)

    # Pre-process start frame to display_size (same crop, then up-scale to 512)
    start_128 = preprocess_rgb_to_128(raw_start)
    bg = cv2.resize(start_128, (target, target), interpolation=cv2.INTER_CUBIC)

    # ── For each niter: top-4 indices by reward, draw frame ──
    canvas_h = target          # square
    fig_w = target * 2 + 8     # left canvas + divider + right plot of same size
    fig_h = target + 56        # 56-px title strip at top

    individual_paths: list[Path] = []
    for n in range(NITER):
        r = rewards_by_niter[n]
        order = np.argsort(-r)
        top_idx = order[:TOP_K].tolist()
        # Left canvas — draw BOTH arms per trajectory in the same hue.
        # 4 trajectories × 2 arms = 8 polylines total; left and right
        # cluster in different image regions so they read naturally.
        left = bg.copy()
        for slot, idx in enumerate(top_idx):
            color = TAB_RGB[COLORS[slot]]
            pix_left  = pixels_left_by_niter[n][idx]
            pix_right = pixels_right_by_niter[n][idx]
            draw_trajectory(left, pix_left,  color, line_width=2, point_radius=4)
            draw_trajectory(left, pix_right, color, line_width=2, point_radius=4)

        # Right plot: best reward across niter, current niter highlighted
        fig, ax = plt.subplots(figsize=(target / 100, target / 100), dpi=100)
        xs = np.arange(NITER)
        # Plot all up to and including current niter
        x_done = xs[: n + 1]
        y_done = best_reward_per_niter[: n + 1]
        ax.plot(x_done, y_done, color="tab:blue", marker="o", linewidth=2)
        ax.scatter([n], [best_reward_per_niter[n]], s=120,
                   color="tab:red", zorder=10, edgecolor="k",
                   label=f"niter {n}")
        ax.set_xlim(-0.5, NITER - 0.5)
        ymin = best_reward_per_niter.min() - 0.005
        ymax = best_reward_per_niter.max() + 0.005
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(xs)
        ax.set_xlabel("niter")
        ax.set_ylabel("best sample reward")
        ax.set_title("Best reward across niter")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        # Render to numpy
        fig.canvas.draw()
        right = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        # Resize right plot to (target, target)
        right = cv2.resize(right, (target, target), interpolation=cv2.INTER_AREA)

        # Compose: top title strip + [left | divider | right]
        composite = np.full((fig_h, fig_w, 3), 0, dtype=np.uint8)
        title_strip = composite[:56]
        title_strip[:] = 30
        cv2.putText(
            title_strip,
            f"MPPI plan(t=0): niter {n}/{NITER - 1}, K={N}, top-{TOP_K} trajectories",
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA,
        )
        composite[56:56 + target, :target] = left
        composite[56:56 + target, target:target + 8] = 80   # divider
        composite[56:56 + target, target + 8:] = right

        out_path = out_dir / f"frame_{n}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        individual_paths.append(out_path)
        print(f"[viz] wrote {out_path}")

    # ── Combined static PNG (5 frames stacked vertically) ──
    combined = np.concatenate(
        [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
         for p in individual_paths],
        axis=0,
    )
    combined_path = out_dir / "all_niters_combined.png"
    cv2.imwrite(str(combined_path),
                cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"[viz] wrote {combined_path}")

    # ── ffmpeg into mp4 ──
    mp4_path = out_dir / "action_sampling.mp4"
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
