"""Overnight Phase C — closed-loop MPPI with the multi-frame warmup port.

Drives production's MPPI v2 (rl/mppi/mppi_planner.py) with the
PushTWMEnv multi-frame warmup installed. The warmup eliminates the
1..9-frame OOD ramp at the start of every WM rollout; the rest of the
config matches configs/mppi/default.yaml with only K (n_sample) overridden.

Output layout (all under --output_dir):

    03_run_log.csv          per-step log
    03_reward_curve.png     reward over control steps
    03_gpu_memory.png       per-step peak GPU memory (MB)
    03_closed_loop.mp4      3-row video: real / closed-loop WM / open-loop WM
    03_experiment_report.md textual summary

The 3-row video shows the same frame window (warmup_start_idx ..
warmup_start_idx + window_size + control_steps - 1) along three
trajectories starting from a shared warmup:

    row 1   real episode RGB (the expert trajectory used for the warmup)
    row 2   closed-loop MPPI: warmup decoded, then WM-rolled with MPPI
            actions
    row 3   open-loop expert: warmup decoded, then WM-rolled with the
            expert (HDF5) actions

Row 2 vs row 3 isolates "what changes if MPPI drives the WM vs if the
expert drives the WM"; row 2 vs row 1 isolates "does the closed-loop WM
state track reality."

Phase C of the overnight run brief. Time cap: 4h wall.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.expert_action import expert_action_from_episode  # noqa: E402
from env.pusht_wm_env import PushTWMEnv, _preprocess_rgb_uint8, PUSHT_CAMERA_KEY  # noqa: E402
from rl.mppi.mppi_planner import MPPIPlanner  # noqa: E402


def _frame_to_uint8(rgb: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0,1] -> (H, W, 3) uint8."""
    return (
        rgb.clamp(0, 1).detach().cpu().numpy() * 255
    ).astype(np.uint8).transpose(1, 2, 0)


def _read_real_rgb(hdf5_path: Path, frame_idx: int, resolution: int) -> np.ndarray:
    """Read one preprocessed frame from the HDF5 as (H, W, 3) uint8."""
    with h5py.File(str(hdf5_path), "r") as f:
        raw = f[f"obs/images/{PUSHT_CAMERA_KEY}"][int(frame_idx)]
    pre = _preprocess_rgb_uint8(raw, resolution=resolution)  # (H, W, 3) float
    return (pre * 255).astype(np.uint8)


def _write_3row_mp4(
    real_rgbs: list[np.ndarray],
    closed_rgbs: list[np.ndarray],
    openexp_rgbs: list[np.ndarray],
    out_path: Path,
    fps: int = 8,
    labels: tuple[str, str, str] = (
        "real (expert HDF5)", "closed-loop MPPI", "open-loop expert",
    ),
) -> None:
    """Stack three RGB sequences vertically and write an mp4."""
    n = min(len(real_rgbs), len(closed_rgbs), len(openexp_rgbs))
    if n == 0:
        return
    H, W = real_rgbs[0].shape[:2]
    pad_h = 24
    band_h = H + pad_h
    out_h = 3 * band_h
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, out_h))
    try:
        for t in range(n):
            rows = [real_rgbs[t], closed_rgbs[t], openexp_rgbs[t]]
            frame = np.zeros((out_h, W, 3), dtype=np.uint8)
            for i, (rgb, label) in enumerate(zip(rows, labels)):
                y0 = i * band_h
                frame[y0 : y0 + pad_h] = 0
                cv2.putText(
                    frame, label, (6, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA,
                )
                frame[y0 + pad_h : y0 + band_h] = rgb
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        vw.release()


def _plot_reward(rewards: list[float], out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(len(rewards)), rewards, marker="o", color="tab:blue")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("control step")
    ax.set_ylabel("CV reward")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_gpu(mem_mb_log: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([r["t"] for r in mem_mb_log])
    peak = np.array([r["peak_mb"] for r in mem_mb_log])
    cur = np.array([r["alloc_mb"] for r in mem_mb_log])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, peak, marker="o", color="tab:red", label="peak (per step)")
    ax.plot(t, cur, marker="s", color="tab:orange", label="allocated (end of step)")
    ax.set_xlabel("control step")
    ax.set_ylabel("GPU memory (MB)")
    ax.set_title("MPPI K=16 warmup-mode GPU memory")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _wm_open_loop_with_expert(
    env: PushTWMEnv,
    hdf5_path: Path,
    warmup_start_idx: int,
    window_size: int,
    n_control: int,
) -> tuple[list[np.ndarray], list[torch.Tensor]]:
    """Open-loop WM continuation using the expert HDF5 actions.

    Initialises from the same warmup window (frames warmup_start_idx ..
    warmup_start_idx + window_size - 1), then advances the WM for
    n_control steps using the NORMALIZED expert action at each frame.

    Returns:
        rgbs:    list[(H, W, 3) uint8] of length window_size + n_control
                 (decoded warmup frames + decoded predicted frames).
        latents: list[(C, H_lat, W_lat)] of the same length.
    """
    z_hist, a_hist = env.load_initial_with_warmup(
        hdf5_path, end_frame_index=warmup_start_idx + window_size - 1,
        window_size=window_size,
    )
    # Decode the warmup frames so the video shows the same prefix as
    # the other rows.
    rgbs: list[np.ndarray] = []
    lats: list[torch.Tensor] = []
    for i in range(window_size):
        rgb_i = env.decode(z_hist[i])
        rgbs.append(_frame_to_uint8(rgb_i))
        lats.append(z_hist[i].detach().cpu().clone())

    # Stateless one-shot rollout with expert actions for n_control steps.
    end_idx = warmup_start_idx + window_size - 1
    expert_actions = []
    for k in range(n_control):
        frame_idx = end_idx + k  # action AT this frame drives INTO frame+1
        a_np = expert_action_from_episode(env, str(hdf5_path), frame_idx)
        expert_actions.append(torch.from_numpy(a_np))
    actions_t = torch.stack(expert_actions, dim=0).to(env.device)  # (n_control, A)
    traj = env.rollout_with_warmup(z_hist, a_hist, actions_t)        # (n_control+1, C, H, W)
    # traj[0] == z_hist[-1] (already in rgbs); skip index 0
    for k in range(1, traj.shape[0]):
        z_k = traj[k]
        rgb_k = env.decode(z_k)
        rgbs.append(_frame_to_uint8(rgb_k))
        lats.append(z_k.detach().cpu().clone())
    return rgbs, lats


def _gather_real_rgbs(
    hdf5_path: Path,
    warmup_start_idx: int,
    window_size: int,
    n_control: int,
    resolution: int,
) -> list[np.ndarray]:
    """Return preprocessed real RGB frames for the same window."""
    rgbs: list[np.ndarray] = []
    n_total = window_size + n_control
    with h5py.File(str(hdf5_path), "r") as f:
        dset = f[f"obs/images/{PUSHT_CAMERA_KEY}"]
        T_dset = dset.shape[0]
        for k in range(n_total):
            idx = warmup_start_idx + k
            if idx >= T_dset:
                # Episode shorter than warmup + control; pad with the last
                # available frame so video lengths match.
                idx = T_dset - 1
            raw = dset[int(idx)]
            pre = _preprocess_rgb_uint8(raw, resolution=resolution)
            rgbs.append((pre * 255).astype(np.uint8))
    return rgbs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mppi/default.yaml")
    ap.add_argument(
        "--wm_ckpt", default="outputs/pusht_cam1/checkpoints/best.ckpt",
    )
    ap.add_argument(
        "--initial_hdf5",
        default="data/mini/pusht/val/episode_0.hdf5",
        help="HDF5 episode used to provide the multi-frame warmup window.",
    )
    ap.add_argument(
        "--warmup_start_frame", type=int, default=0,
        help="Episode frame index of the FIRST warmup frame (inclusive).",
    )
    ap.add_argument(
        "--window_size", type=int, default=10,
        help="Number of warmup frames. Capped by the WM context (10).",
    )
    ap.add_argument(
        "--goal", default="tests/goal_selection/state_goal.pt",
    )
    ap.add_argument(
        "--output_dir", default="/tmp/overnight_run/phase_c_artifacts",
    )
    ap.add_argument(
        "--n_sample", type=int, default=16,
        help="MPPI K. Overrides config.n_sample.",
    )
    ap.add_argument(
        "--control_steps", type=int, default=50,
        help="Number of outer control steps.",
    )
    ap.add_argument(
        "--wall_cap_seconds", type=float, default=4 * 3600.0,
        help="Hard wall-time cap; stop the loop if exceeded.",
    )
    ap.add_argument(
        "--seed", type=int, default=0,
    )
    ap.add_argument(
        "--cv_n_workers", type=int, default=4,
        help="CV labeler workers. Lower than the K=100 default since N=16.",
    )
    ap.add_argument(
        "--decode_batch_size", type=int, default=16,
        help="Decoder batch when scoring N candidate trajectories.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + override config (only K and the per-run knobs).
    cfg = OmegaConf.load(args.config)
    cfg.n_sample = int(args.n_sample)
    cfg.control_steps = int(args.control_steps)
    cfg.seed = int(args.seed)
    cfg.cv_n_workers = int(args.cv_n_workers)
    cfg.decode_batch_size = int(args.decode_batch_size)

    device = "cuda:0"
    print(f"[Phase C] device={device}  K={cfg.n_sample}  H={cfg.n_look_ahead}  "
          f"niter={cfg.n_update_iter}  control_steps={cfg.control_steps}")
    print(f"[Phase C] warmup: start_frame={args.warmup_start_frame} "
          f"window_size={args.window_size}")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    env = PushTWMEnv(
        args.wm_ckpt,
        device=device,
        cv_n_workers=int(cfg.cv_n_workers),
    )

    # Initial warmup from the HDF5.
    end_idx = args.warmup_start_frame + args.window_size - 1
    z_history, action_history = env.load_initial_with_warmup(
        args.initial_hdf5,
        end_frame_index=end_idx,
        window_size=args.window_size,
    )
    print(f"[Phase C] loaded warmup: z_history={tuple(z_history.shape)} "
          f"action_history={tuple(action_history.shape)}")

    goal = env.load_goal(args.goal)
    print(f"[Phase C] goal: cx={goal['cx']:.2f} cy={goal['cy']:.2f} "
          f"theta={goal['theta_deg']:.2f}deg")

    # Install the warmup on the env. From this point on, env.rollout()
    # (which the planner calls internally and we also call for execution)
    # uses the warmup history as the dynamics context.
    env.set_warmup(z_history, action_history)

    planner = MPPIPlanner(env, cfg)

    # ── main control loop ────────────────────────────────────────────
    per_step: list[dict[str, Any]] = []
    closed_rgbs: list[np.ndarray] = []
    closed_latents: list[torch.Tensor] = []
    mem_log: list[dict[str, float]] = []

    # Decode the warmup frames once for the video.
    for i in range(args.window_size):
        rgb_i = env.decode(z_history[i])
        closed_rgbs.append(_frame_to_uint8(rgb_i))
        closed_latents.append(z_history[i].detach().cpu().clone())
    # Initial-decoded reward (using the latest warmup frame).
    state0 = env.estimate_state(env.decode(z_history[-1]))
    r0 = env.compute_reward(
        state0, goal, cv_fail_penalty=float(cfg.cv_fail_penalty),
    )
    per_step.append({
        "t": 0,
        "action": None,
        "reward": float(r0),
        "cv_success": bool(state0["success"]),
        "cx": float(state0["cx"]) if state0["success"] else None,
        "cy": float(state0["cy"]) if state0["success"] else None,
        "theta_deg": (
            float(state0["theta_deg"]) if state0["success"] else None
        ),
        "plan_wall_s": None,
        "exec_wall_s": None,
        "peak_gpu_mb": None,
    })
    print(f"[Phase C] t=0  warmup-init  r={float(r0):+.4f}  "
          + (f"cv=({float(state0['cx']):.1f},{float(state0['cy']):.1f},"
             f"{float(state0['theta_deg']):+.1f}deg)"
             if state0["success"] else "CV-FAIL"))

    # Warm-start state for the planner (shift-and-pad).
    H = int(cfg.n_look_ahead)
    A = int(cfg.action_dim)
    act_seq_running = torch.zeros(H, A, device=env.device, dtype=torch.float32)

    t0_run = time.time()
    last_action: torch.Tensor | None = None
    for t in range(int(cfg.control_steps)):
        torch.cuda.reset_peak_memory_stats(device)
        wall_now = time.time() - t0_run
        if wall_now > float(args.wall_cap_seconds):
            print(f"[Phase C] wall cap {args.wall_cap_seconds}s reached at "
                  f"t={t}; stopping.")
            break

        # ── PLAN ──
        z_placeholder = z_history[-1]  # planner ignores content when warmup set
        t0_plan = time.time()
        a, iter_log = planner.plan_step(
            z_placeholder, goal,
            init_act_seq=act_seq_running,
            return_iteration_log=True,
        )
        converged = planner.last_stats.act_seq.detach().cpu().clone()  # (H, A)
        plan_wall = time.time() - t0_plan

        # ── EXEC: one WM step with the chosen first action ──
        t0_exec = time.time()
        a_dev = a.to(env.device)  # (A,)
        actions_one = a_dev.unsqueeze(0).unsqueeze(0)  # (1, 1, A) batched
        with torch.no_grad():
            # The warmup path returns (B, 2, C, H, W); index 1 is z_next.
            traj_b = env.rollout(z_placeholder.unsqueeze(0), actions_one)
        z_next = traj_b[0, 1]  # (C, H, W)
        exec_wall = time.time() - t0_exec
        last_action = a_dev.detach().clone()

        # ── SLIDE warmup window ──
        z_history = torch.cat(
            [z_history[1:], z_next.unsqueeze(0)], dim=0,
        )  # (T_warm, C, H, W)
        action_history = torch.cat(
            [action_history[1:], a_dev.unsqueeze(0)], dim=0,
        )  # (T_warm, A)
        env.set_warmup(z_history, action_history)

        # ── decode + CV reward on the executed latent ──
        rgb_t = env.decode(z_next)
        rgb_t_u8 = _frame_to_uint8(rgb_t)
        state_t = env.estimate_state(rgb_t)
        r_t = env.compute_reward(
            state_t, goal, cv_fail_penalty=float(cfg.cv_fail_penalty),
        )
        closed_rgbs.append(rgb_t_u8)
        closed_latents.append(z_next.detach().cpu().clone())

        # ── memory bookkeeping ──
        peak_mb = float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2)
        alloc_mb = float(torch.cuda.memory_allocated(device)) / (1024 ** 2)
        mem_log.append({
            "t": t + 1, "peak_mb": peak_mb, "alloc_mb": alloc_mb,
        })

        per_step.append({
            "t": t + 1,
            "action": [float(v) for v in a_dev.cpu().tolist()],
            "reward": float(r_t),
            "cv_success": bool(state_t["success"]),
            "cx": float(state_t["cx"]) if state_t["success"] else None,
            "cy": float(state_t["cy"]) if state_t["success"] else None,
            "theta_deg": (
                float(state_t["theta_deg"]) if state_t["success"] else None
            ),
            "plan_wall_s": round(plan_wall, 3),
            "exec_wall_s": round(exec_wall, 3),
            "peak_gpu_mb": round(peak_mb, 1),
        })
        print(
            f"[Phase C] t={t + 1:02d}  r={float(r_t):+.4f}  "
            + (
                f"cv=({float(state_t['cx']):.1f},{float(state_t['cy']):.1f},"
                f"{float(state_t['theta_deg']):+.1f}deg)"
                if state_t['success'] else "CV-FAIL"
            )
            + f"  plan={plan_wall:.2f}s  exec={exec_wall:.2f}s  "
              f"peak={peak_mb:.0f}MB"
        )

        # Shift-and-pad warm-start (USE_WARM_START is True in production).
        converged_dev = converged.to(env.device)
        new_tail = converged_dev[-1:].repeat(1, 1)
        act_seq_running = torch.cat(
            [converged_dev[1:], new_tail], dim=0,
        )

    wall_total = time.time() - t0_run
    n_done = len(per_step) - 1
    print(f"\n[Phase C] run wall time: {wall_total:.1f}s for "
          f"{n_done} control steps")

    # ── artifacts ────────────────────────────────────────────────────
    # run_log.csv
    with open(out_dir / "03_run_log.csv", "w", newline="") as fh:
        cols = ["t", "reward", "cv_success", "cx", "cy", "theta_deg",
                "plan_wall_s", "exec_wall_s", "peak_gpu_mb"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in per_step:
            w.writerow({k: r.get(k) for k in cols})

    # reward curve
    rewards = [r["reward"] for r in per_step]
    _plot_reward(
        rewards, out_dir / "03_reward_curve.png",
        title=f"Phase C: K={cfg.n_sample}, warmup={args.window_size}f, "
              f"H={cfg.n_look_ahead}, niter={cfg.n_update_iter}",
    )
    _plot_gpu(mem_log, out_dir / "03_gpu_memory.png")

    # 3-row video: real | closed-loop | open-loop expert
    # n_done may be < control_steps if wall cap fired.
    print("[Phase C] gathering real episode frames + open-loop WM trajectory")
    real_rgbs = _gather_real_rgbs(
        Path(args.initial_hdf5),
        warmup_start_idx=args.warmup_start_frame,
        window_size=args.window_size,
        n_control=n_done,
        resolution=env.resolution,
    )
    # Clear warmup so the open-loop helper installs its own.
    env.clear_warmup()
    open_rgbs, _ = _wm_open_loop_with_expert(
        env, Path(args.initial_hdf5),
        warmup_start_idx=args.warmup_start_frame,
        window_size=args.window_size,
        n_control=n_done,
    )
    # closed_rgbs already has window_size warmup-decoded + n_done MPPI frames.
    assert len(closed_rgbs) == args.window_size + n_done
    assert len(open_rgbs) == args.window_size + n_done, (
        f"open_rgbs len {len(open_rgbs)} != expected "
        f"{args.window_size + n_done}"
    )
    assert len(real_rgbs) == args.window_size + n_done

    _write_3row_mp4(
        real_rgbs, closed_rgbs, open_rgbs,
        out_dir / "03_closed_loop.mp4", fps=8,
    )
    print(f"[Phase C] wrote 3-row video {out_dir / '03_closed_loop.mp4'}")

    # summary.json (machine-readable)
    final = per_step[-1]
    initial = per_step[0]
    final_pos_distance = None
    final_angle_error = None
    if final["cv_success"] and final["cx"] is not None:
        dx = final["cx"] - goal["cx"]
        dy = final["cy"] - goal["cy"]
        final_pos_distance = math.sqrt(dx * dx + dy * dy)
    if final["cv_success"] and final["theta_deg"] is not None:
        a_diff = final["theta_deg"] - goal["theta_deg"]
        final_angle_error = ((a_diff + 180) % 360) - 180
    initial_pos_distance = None
    if initial["cv_success"] and initial["cx"] is not None:
        dx = initial["cx"] - goal["cx"]
        dy = initial["cy"] - goal["cy"]
        initial_pos_distance = math.sqrt(dx * dx + dy * dy)
    mean_r_last_10 = float(np.mean([r["reward"] for r in per_step[-10:]]))
    n_cv_fail = sum(1 for r in per_step if not r["cv_success"])
    peak_overall = max((r["peak_gpu_mb"] or 0.0) for r in per_step)
    summary = {
        "phase": "C",
        "config_overrides": {
            "n_sample": int(cfg.n_sample),
            "control_steps": int(cfg.control_steps),
            "seed": int(cfg.seed),
            "cv_n_workers": int(cfg.cv_n_workers),
            "decode_batch_size": int(cfg.decode_batch_size),
        },
        "warmup": {
            "hdf5": str(args.initial_hdf5),
            "start_frame": int(args.warmup_start_frame),
            "window_size": int(args.window_size),
        },
        "goal": {
            "cx": float(goal["cx"]),
            "cy": float(goal["cy"]),
            "theta_deg": float(goal["theta_deg"]),
        },
        "wall_time_s": round(wall_total, 1),
        "n_control_steps_completed": int(n_done),
        "n_cv_failures": int(n_cv_fail),
        "initial_reward": float(rewards[0]),
        "initial_pos_distance_px": (
            None if initial_pos_distance is None
            else round(initial_pos_distance, 3)
        ),
        "final_reward": float(rewards[-1]),
        "final_pos_distance_px": (
            None if final_pos_distance is None
            else round(final_pos_distance, 3)
        ),
        "final_angle_error_deg": (
            None if final_angle_error is None
            else round(final_angle_error, 2)
        ),
        "best_reward": float(max(rewards)),
        "mean_reward_last_10": round(mean_r_last_10, 4),
        "peak_gpu_mb_overall": round(peak_overall, 1),
    }
    (out_dir / "03_summary.json").write_text(json.dumps(summary, indent=2))

    # textual report
    lines = [
        "# Phase C — closed-loop MPPI with multi-frame warmup",
        "",
        "## Run parameters",
        "",
        f"  K (n_sample)      = {cfg.n_sample}",
        f"  H (n_look_ahead)  = {cfg.n_look_ahead}",
        f"  niter             = {cfg.n_update_iter}",
        f"  control_steps     = {cfg.control_steps}",
        f"  seed              = {cfg.seed}",
        f"  warmup window     = {args.window_size} frames "
        f"(episode {args.warmup_start_frame}..{end_idx})",
        f"  episode HDF5      = {args.initial_hdf5}",
        f"  WM checkpoint     = {args.wm_ckpt}",
        "",
        "## Results",
        "",
        f"  wall time              = {wall_total:.1f} s "
        f"({wall_total / 60:.2f} min)",
        f"  steps completed        = {n_done} / {cfg.control_steps}",
        f"  initial reward         = {rewards[0]:+.4f}",
        f"  final reward           = {rewards[-1]:+.4f}",
        f"  best reward            = {max(rewards):+.4f}",
        f"  mean reward last 10    = {mean_r_last_10:+.4f}",
        f"  initial pos dist (px)  = {initial_pos_distance}",
        f"  final pos dist (px)    = {final_pos_distance}",
        f"  final angle err (deg)  = {final_angle_error}",
        f"  n CV failures          = {n_cv_fail}",
        f"  peak GPU (overall)     = {peak_overall:.1f} MB",
        "",
        "## Artifacts",
        "",
        f"  03_run_log.csv          per-step log",
        f"  03_reward_curve.png     reward over control steps",
        f"  03_gpu_memory.png       per-step peak + allocated GPU MB",
        f"  03_closed_loop.mp4      3-row video (real / MPPI / open-loop)",
        f"  03_summary.json         machine-readable summary",
        "",
    ]
    (out_dir / "03_experiment_report.md").write_text("\n".join(lines))
    print(f"\n[Phase C] artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
