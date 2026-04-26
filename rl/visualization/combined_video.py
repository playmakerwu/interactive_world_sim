"""Combined per-frame trajectory + reward-iteration visualization.

For each MPPI run produced by ``scripts/run_mppi_v2.py``, this module
composes a side-by-side video saved next to the existing
``trajectory.mp4``:

  Left half  — decoded RGB upsampled to 512x512, with crisp vector
               overlays drawn at canvas resolution:
                 * red marker + arrow at the GOAL pose
                 * lime marker + arrow at the CURRENT CV pose
                 * up to 10 candidate endpoints from the LAST refinement
                   iteration of this plan_step, colored by reward rank
                   (yellow = best, dark blue = 10th)
                 * frame label ``frame NNN / control_step NNN``

  Right half — reward-vs-iteration plot for this plan_step (softmax-
               weighted mean line, max sample dashed, +/- 1 std band).
               Frame 0 has no plan executed yet, so the right half is
               a centered placeholder.

The composite frame is 1024x512. Stitched into ``trajectory_combined.mp4``
at the same fps as ``trajectory.mp4`` (default 8).

Pure post-processing: depends on artifacts already on disk
(``trajectory.mp4``, ``summary.json``, ``iteration_log.pt``). Does not
touch the WM, CV, or planner.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ─── Layout constants ──────────────────────────────────────────────────

CANVAS_PX = 512                # square canvas for both halves
UPSAMPLE_FROM = 128            # WM decoder native resolution
UPSAMPLE_FACTOR = CANVAS_PX // UPSAMPLE_FROM  # 4 — must be exact integer

GOAL_RGB = (220, 0, 0)         # red
CURRENT_RGB = (0, 220, 0)      # lime green
ARROW_LEN_PX = 30              # in canvas (512) space
MARKER_RADIUS_PX = 6
LINE_THICKNESS = 2
ARROW_TIP_LENGTH = 0.30

TOP_K = 10                     # how many candidate trajectories to plot
ENDPOINT_RADIUS_PX = 4         # marker drawn at the LAST point of each polyline
POLYLINE_THICKNESS_PX = 2      # cv2 takes int; spec asks ~1.5 — 2 is closest
POLYLINE_ALPHA = 0.6           # blended in via cv2.addWeighted

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.5
LABEL_THICKNESS = 1


# ─── Drawing primitives ────────────────────────────────────────────────

def _draw_marker_and_arrow(
    canvas: np.ndarray,
    cx: float, cy: float,
    sin_theta: float, cos_theta: float,
    color: tuple[int, int, int],
    arrow_len: int = ARROW_LEN_PX,
) -> None:
    """Draw a filled circle + orientation arrow + 1-px white border on
    ``canvas`` (in-place). Coordinates already in canvas (512) space."""
    cx_i, cy_i = int(round(cx)), int(round(cy))
    tip = (
        int(round(cx + arrow_len * cos_theta)),
        int(round(cy + arrow_len * sin_theta)),
    )
    # white halo first (1 px wider) so red/green stays legible on dark BG
    cv2.arrowedLine(canvas, (cx_i, cy_i), tip, (255, 255, 255),
                    thickness=LINE_THICKNESS + 2, tipLength=ARROW_TIP_LENGTH)
    cv2.circle(canvas, (cx_i, cy_i), MARKER_RADIUS_PX + 1, (255, 255, 255), thickness=-1)
    # colored arrow + marker on top
    cv2.arrowedLine(canvas, (cx_i, cy_i), tip, color,
                    thickness=LINE_THICKNESS, tipLength=ARROW_TIP_LENGTH)
    cv2.circle(canvas, (cx_i, cy_i), MARKER_RADIUS_PX, color, thickness=-1)


def _draw_text_with_outline(
    canvas: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = LABEL_SCALE,
    thickness: int = LABEL_THICKNESS,
) -> None:
    """White (or specified) text with black outline for legibility."""
    cv2.putText(canvas, text, org, LABEL_FONT, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, LABEL_FONT, scale, color,
                thickness, cv2.LINE_AA)


def _reward_to_top_k_colors(rewards_top_k: np.ndarray) -> list[tuple[int, int, int]]:
    """Map K rewards (sorted descending) to viridis-like RGB tuples.

    Best reward -> bright yellow ``#FFD700``; worst-of-top-K -> dark blue
    ``#1F3A93``. Linear interpolation in RGB space (good enough for K<=10).
    """
    K = len(rewards_top_k)
    if K == 0:
        return []
    # endpoints
    yellow = np.array([255, 215, 0], dtype=np.float32)   # #FFD700
    dark_blue = np.array([31, 58, 147], dtype=np.float32)  # #1F3A93
    if K == 1:
        return [tuple(int(v) for v in yellow)]
    out: list[tuple[int, int, int]] = []
    for rank in range(K):
        # rank 0 = best (yellow), rank K-1 = worst (dark blue)
        t = rank / (K - 1)
        rgb = (1 - t) * yellow + t * dark_blue
        out.append(tuple(int(round(v)) for v in rgb))
    return out


def _draw_top_k_polylines(
    canvas: np.ndarray,
    top_k_cx: torch.Tensor | np.ndarray,        # (K, H+1)
    top_k_cy: torch.Tensor | np.ndarray,        # (K, H+1)
    top_k_success: torch.Tensor | np.ndarray | None,  # (K, H+1) bool or None
    top_k_rewards: torch.Tensor | np.ndarray,   # (K,) — descending
    upsample_factor: int = UPSAMPLE_FACTOR,
) -> None:
    """Draw the top-K predicted trajectories as alpha-blended polylines.

    Each polyline is colored by its rank (top_k_rewards[0] = best ->
    bright yellow; top_k_rewards[-1] = worst-of-K -> dark blue). The
    final valid point of each polyline gets a filled circle marker so
    the eye can find where each line ends.

    All polylines share one ``cv2.addWeighted`` blend onto the canvas
    (uniform alpha across the K lines). Endpoint markers draw at full
    opacity on top.

    Coordinates are in 128-px decoder space; scaled by ``upsample_factor``.
    Steps with ``success=False`` (CV fail) are skipped within a polyline.
    """
    cx = np.asarray(top_k_cx, dtype=np.float32) * upsample_factor
    cy = np.asarray(top_k_cy, dtype=np.float32) * upsample_factor
    rewards = np.asarray(top_k_rewards, dtype=np.float32)
    K, Hp1 = cx.shape
    if top_k_success is None:
        success = np.ones((K, Hp1), dtype=bool)
    else:
        success = np.asarray(top_k_success, dtype=bool)

    colors = _reward_to_top_k_colors(rewards)
    overlay = canvas.copy()

    # 1) Polylines on the overlay (alpha-blended once at the end).
    #    Draw WORST -> BEST so the best (yellow) line lands on top of any
    #    overlapping worse lines. The single addWeighted at the end then
    #    blends the whole overlay with the canvas at uniform alpha.
    for k in range(K - 1, -1, -1):
        valid_mask = success[k] & np.isfinite(cx[k]) & np.isfinite(cy[k])
        if int(valid_mask.sum()) < 2:
            continue
        pts = np.stack(
            [cx[k][valid_mask].astype(np.int32),
             cy[k][valid_mask].astype(np.int32)],
            axis=1,
        ).reshape(-1, 1, 2)
        cv2.polylines(
            overlay, [pts], isClosed=False, color=colors[k],
            thickness=POLYLINE_THICKNESS_PX, lineType=cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, POLYLINE_ALPHA, canvas, 1 - POLYLINE_ALPHA, 0, dst=canvas)

    # 2) Endpoint markers (full opacity, drawn directly on canvas).
    #    Same worst -> best ordering so the best marker sits on top.
    for k in range(K - 1, -1, -1):
        # Find LAST valid step for this polyline.
        last_valid_idx = -1
        for t in range(Hp1 - 1, -1, -1):
            if success[k, t] and np.isfinite(cx[k, t]) and np.isfinite(cy[k, t]):
                last_valid_idx = t
                break
        if last_valid_idx < 0:
            continue
        x = int(round(cx[k, last_valid_idx]))
        y = int(round(cy[k, last_valid_idx]))
        if not (0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]):
            continue
        # 1 px white border for legibility on T-block / dark backgrounds
        cv2.circle(canvas, (x, y), ENDPOINT_RADIUS_PX + 1, (255, 255, 255), thickness=-1)
        cv2.circle(canvas, (x, y), ENDPOINT_RADIUS_PX, colors[k], thickness=-1)


# ─── Reward-vs-iteration plot ──────────────────────────────────────────

def _render_iteration_plot(
    step_log: list[dict[str, Any]] | None,
    final_reward: float | None,
    distance_to_goal: float | None,
    canvas_px: int = CANVAS_PX,
    y_lim: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render a square (canvas_px x canvas_px) reward-vs-iteration plot.

    Returns RGB uint8. When ``step_log`` is empty/None (e.g. frame 0),
    returns a placeholder card with explanatory text.

    ``y_lim``: when provided, fixes the reward y-axis to that range so
    every frame in the video uses the same vertical scale. Lets the
    viewer compare reward magnitudes across plan_steps without having
    to re-read axis ticks.
    """
    dpi = 100
    inches = canvas_px / dpi
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)

    if not step_log:
        ax.set_axis_off()
        ax.text(0.5, 0.5,
                "frame 0\ninitial state\n(no plan executed yet)",
                ha="center", va="center", fontsize=14, color="dimgray")
    else:
        iterations = np.array([int(rec["iter"]) for rec in step_log], dtype=int)
        sw = np.array([float(rec["reward_softmax_weighted"]) for rec in step_log], dtype=np.float32)
        rmax = np.array([float(rec["reward_max"]) for rec in step_log], dtype=np.float32)
        rmean = np.array([float(rec["reward_mean"]) for rec in step_log], dtype=np.float32)
        rstd = np.array([float(rec["reward_std"]) for rec in step_log], dtype=np.float32)

        ax.fill_between(iterations, rmean - rstd, rmean + rstd,
                        color="tab:blue", alpha=0.16, linewidth=0,
                        label="sample mean +/- 1 std")
        ax.plot(iterations, sw, color="tab:blue", linewidth=2.2,
                marker="o", markersize=4, label="softmax-weighted mean")
        ax.plot(iterations, rmax, color="tab:cyan", linewidth=1.6,
                linestyle="--", marker="s", markersize=3, label="best sample")
        ax.set_xlabel("iteration")
        ax.set_ylabel("reward")
        ax.grid(alpha=0.25)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

        title_bits = []
        if final_reward is not None:
            title_bits.append(f"executed reward = {final_reward:+.3f}")
        if distance_to_goal is not None:
            title_bits.append(f"dist to goal = {distance_to_goal:.1f} px")
        if title_bits:
            ax.set_title("  |  ".join(title_bits), fontsize=10, color="dimgray")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img_rgb.shape[0] != canvas_px or img_rgb.shape[1] != canvas_px:
        img_rgb = cv2.resize(img_rgb, (canvas_px, canvas_px),
                             interpolation=cv2.INTER_AREA)
    return img_rgb


# ─── Per-frame composite ───────────────────────────────────────────────

def _compose_frame(
    rgb_native: np.ndarray,           # (128, 128, 3) RGB uint8
    per_step_row: dict[str, Any],
    goal: dict[str, Any],
    step_log: list[dict[str, Any]] | None,
    frame_idx: int,
    upsample_factor: int = UPSAMPLE_FACTOR,
    y_lim: tuple[float, float] | None = None,
) -> np.ndarray:
    """Build one combined frame (1024x512 RGB uint8).

    Z-order on the left half (drawn back-to-front):
      1. upsampled RGB
      2. top-K predicted polylines + their endpoint markers
      3. current CV pose (lime marker + arrow)
      4. goal pose (red marker + arrow) — sits on top so the target
         remains visible even when the current pose overlaps it.
    """
    canvas_px = rgb_native.shape[0] * upsample_factor

    # Left half: upsample RGB then draw overlays at canvas resolution
    left = cv2.resize(rgb_native, (canvas_px, canvas_px),
                      interpolation=cv2.INTER_LINEAR).copy()

    # 1) Top-K predicted polylines (drawn first, so current/goal sit on top).
    #    Only when a plan has been executed AND the planner captured per-sample
    #    CV state on the last iteration.
    if step_log:
        last_iter = step_log[-1]
        if (
            last_iter.get("top_k_intermediate_cx") is not None
            and last_iter.get("top_k_intermediate_cy") is not None
            and last_iter.get("top_k_rewards") is not None
        ):
            _draw_top_k_polylines(
                left,
                last_iter["top_k_intermediate_cx"],
                last_iter["top_k_intermediate_cy"],
                last_iter.get("top_k_intermediate_success"),
                last_iter["top_k_rewards"],
                upsample_factor=upsample_factor,
            )

    # 2) Current state overlay (drawn before goal so goal sits on top)
    if per_step_row.get("cv_success") and per_step_row.get("cx") is not None:
        cur_cx = float(per_step_row["cx"]) * upsample_factor
        cur_cy = float(per_step_row["cy"]) * upsample_factor
        cur_theta = math.radians(float(per_step_row["theta_deg"]))
        _draw_marker_and_arrow(
            left, cur_cx, cur_cy,
            math.sin(cur_theta), math.cos(cur_theta),
            CURRENT_RGB,
        )
    else:
        _draw_text_with_outline(
            left, "CV-FAIL", (canvas_px // 2 - 40, canvas_px // 2),
            color=(255, 0, 0), scale=1.0, thickness=2,
        )

    # 3) Goal overlay (drawn last so the target is always visible, even
    #    when the current pose overlaps it).
    goal_cx = float(goal["cx"]) * upsample_factor
    goal_cy = float(goal["cy"]) * upsample_factor
    goal_theta = math.radians(float(goal["theta_deg"]))
    _draw_marker_and_arrow(
        left, goal_cx, goal_cy,
        math.sin(goal_theta), math.cos(goal_theta),
        GOAL_RGB,
    )

    # Frame label top-left
    t_label = per_step_row.get("t", frame_idx)
    _draw_text_with_outline(
        left, f"frame {frame_idx:03d} / control_step {int(t_label):03d}",
        (8, 22),
    )

    # Top-K legend bottom-left
    if step_log and step_log[-1].get("top_k_intermediate_cx") is not None:
        _draw_text_with_outline(
            left, "top-10 candidate trajectories (last iter)",
            (8, canvas_px - 22),
            scale=0.45, thickness=1,
        )
        _draw_text_with_outline(
            left, "yellow = best  ->  dark blue = 10th",
            (8, canvas_px - 6),
            scale=0.42, thickness=1,
        )

    # Right half: iteration reward plot
    final_reward = per_step_row.get("reward")
    if (
        per_step_row.get("cv_success")
        and per_step_row.get("cx") is not None
        and goal.get("cx") is not None
    ):
        dist = math.hypot(
            float(per_step_row["cx"]) - float(goal["cx"]),
            float(per_step_row["cy"]) - float(goal["cy"]),
        )
    else:
        dist = None
    right = _render_iteration_plot(step_log, final_reward, dist, canvas_px,
                                   y_lim=y_lim)

    # Stitch horizontally
    combined = np.concatenate([left, right], axis=1)
    return combined


# ─── Shared y-axis across plan_steps ───────────────────────────────────

def _compute_shared_y_lim(
    iteration_logs: list[list[dict[str, Any]]],
) -> tuple[float, float] | None:
    """Find a single (y_min, y_max) covering every reward curve in the run.

    Lower bound  = min over all iterations of ``reward_mean - reward_std``
    Upper bound  = max over all iterations of ``reward_max``

    Deliberately ignores ``reward_min`` so a single bad-but-not-CV-fail
    sample (e.g. one of N=32 lands in a low-reward pose) doesn't
    compress the visible reward curves into a flat line at the top of
    the plot. The std band still draws correctly because we use
    ``reward_mean - reward_std`` as the lower envelope.

    Returns None if the logs are empty (e.g. n_update_iter == 0).
    """
    lows: list[float] = []
    highs: list[float] = []
    for step_log in iteration_logs:
        if not step_log:
            continue
        for rec in step_log:
            rmean = float(rec["reward_mean"])
            rstd = float(rec["reward_std"])
            rmax = float(rec["reward_max"])
            lows.append(rmean - rstd)
            highs.append(rmax)
    if not lows:
        return None
    y_min, y_max = min(lows), max(highs)
    if y_max <= y_min:
        # Degenerate: pad arbitrarily so set_ylim doesn't get a zero range.
        pad = max(abs(y_min), 1e-3)
        return y_min - pad, y_max + pad
    span = y_max - y_min
    return y_min - 0.05 * span, y_max + 0.05 * span


# ─── Public entry point ────────────────────────────────────────────────

def render_combined_video(
    run_dir: Path,
    fps: int = 8,
    upsample_factor: int = UPSAMPLE_FACTOR,
) -> Path:
    """Read run_dir's existing artifacts and write trajectory_combined.mp4.

    Returns the output path.
    """
    run_dir = Path(run_dir)
    src_mp4 = run_dir / "trajectory.mp4"
    summary_path = run_dir / "summary.json"
    iter_log_path = run_dir / "iteration_log.pt"
    out_mp4 = run_dir / "trajectory_combined.mp4"

    if not src_mp4.exists():
        raise FileNotFoundError(src_mp4)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = json.loads(summary_path.read_text())
    per_step = summary["per_step"]
    goal = summary["goal_state"]

    iteration_logs: list[list[dict[str, Any]]] = []
    if iter_log_path.exists():
        iteration_logs = torch.load(str(iter_log_path), map_location="cpu",
                                    weights_only=False)

    # Read all RGB frames once
    cap = cv2.VideoCapture(str(src_mp4))
    frames_bgr: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()
    if not frames_bgr:
        raise RuntimeError(f"no frames read from {src_mp4}")
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

    # Pre-compute a SHARED y-axis range across every plan_step's reward
    # plot so all frames in the video use the same vertical scale (lets
    # the viewer compare reward magnitudes between plan_steps without
    # re-reading axis ticks each time).
    y_lim = _compute_shared_y_lim(iteration_logs)

    n_frames = len(frames_rgb)
    n_per_step = len(per_step)
    n_iter_logs = len(iteration_logs)

    # The runner writes one initial frame + one frame per control_step.
    # iteration_logs has one entry per control_step (no entry for frame 0).
    # If counts disagree, log a warning and proceed with min().
    if n_frames != n_per_step:
        print(f"  WARN  frames={n_frames} per_step={n_per_step}; "
              f"using min({n_frames}, {n_per_step})")
    n = min(n_frames, n_per_step)

    canvas_px = frames_rgb[0].shape[0] * upsample_factor
    out_h, out_w = canvas_px, canvas_px * 2
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (out_w, out_h))
    try:
        for i in range(n):
            # iteration_logs[k] corresponds to per_step[k+1] (k-th plan_step
            # was executed to produce frame k+1). Frame 0 has no plan.
            step_log = None
            if i >= 1 and (i - 1) < n_iter_logs:
                step_log = iteration_logs[i - 1]
            combined_rgb = _compose_frame(
                frames_rgb[i], per_step[i], goal, step_log,
                frame_idx=i, upsample_factor=upsample_factor,
                y_lim=y_lim,
            )
            writer.write(cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return out_mp4
