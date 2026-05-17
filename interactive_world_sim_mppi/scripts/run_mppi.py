"""Closed-loop MPPI driver — yiru-side analog of ``scripts/run_mppi_v2.py``.

Mirrors the semantics of production's closed-loop driver:
  * argparse signature (per-flag defaults + algorithm-change override
    gate) matching production
  * SEI / step_each_iter while-loop
  * audit_log.json + trajectory.mp4 + summary.json + reward_curve.png
  * anchor advancement when sample_delta_clip or delta_mode is on
  * cross-step warm-start via shift-and-pad

Differences from production:
  * No multi-GPU dispatch (single-process; yiru does not have the
    distributed sample sharder)
  * No multi-process audit/debug dumps (debug_mode hooks present but
    only the audit_log path is exercised by default)

Usage example:
  python interactive_world_sim_mppi/scripts/run_mppi.py \\
      --initial_hdf5 data/mini/pusht/val/episode_0.hdf5 \\
      --initial_frame 9 \\
      --goal_frame 150 \\
      --output_dir outputs/mppi/yiru_runs/example
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interactive_world_sim_env import WorldModelEnv  # noqa: E402

from interactive_world_sim_mppi.config import Config  # noqa: E402
from interactive_world_sim_mppi._planner import (  # noqa: E402
    Planner,
    detect_goal_state_from_episode,
)


def _write_mp4(frames: list[np.ndarray], out_path: Path, fps: int = 8) -> None:
    """Mirrors production's ``_write_mp4``: RGB→BGR + mp4v fourcc."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def _apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> tuple[Config, dict]:
    """Apply per-flag overrides to the Config dataclass.

    Returns (new_cfg, config_deviation_dict). Algorithm-level overrides
    (--n_sample, --n_update_iter, --horizon) require --override_reason.
    """
    changed: list[str] = []
    overrides: dict[str, Any] = {}

    if args.control_steps is not None:
        overrides["control_steps"] = int(args.control_steps)
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.decode_batch_size is not None:
        overrides["decode_batch_size"] = int(args.decode_batch_size)
    if args.cv_n_workers is not None:
        overrides["cv_n_workers"] = int(args.cv_n_workers)

    algo_overrides: dict[str, Any] = {}
    if args.n_sample is not None:
        algo_overrides["n_sample"] = int(args.n_sample)
        changed.append("n_sample")
    if args.n_update_iter is not None:
        algo_overrides["n_update_iter"] = int(args.n_update_iter)
        changed.append("n_update_iter")
    if args.horizon is not None:
        h = int(args.horizon)
        if h < 1 or h > 50:
            raise ValueError(
                f"--horizon must be in [1, 50] (drift validated up to 50). "
                f"Got {h}."
            )
        algo_overrides["n_look_ahead"] = h
        changed.append("n_look_ahead")

    if changed and not args.override_reason:
        raise SystemExit(
            "Algorithm-changing override(s) require --override_reason: "
            f"{changed}"
        )

    config_deviation = {
        "changed": changed,
        "reason": args.override_reason,
        "expected_impact_quantified": args.expected_impact
        or "Caller did not provide an expected_impact_quantified.",
    }

    overrides.update(algo_overrides)
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg, config_deviation


def _run_episode(args: argparse.Namespace, cfg: Config, out_dir: Path) -> None:
    """Run one MPPI control episode."""
    device = args.device or cfg.device
    print(f"Effective config:\n{cfg}")
    print(f"[step_each_iter] = {cfg.step_each_iter}")
    print(f"Loading WorldModelEnv (task={args.task!r}, device={device})")

    env = WorldModelEnv(args.task, device=device)
    init_window = int(args.init_window_size)
    env.reset(
        init_episode_path=args.initial_hdf5,
        init_episode_index=int(args.initial_frame),
        init_window_size=init_window,
    )

    # Goal: either an explicit state_goal.pt (production-style) or
    # detect it from a chosen episode frame (yiru convenience).
    if args.goal is not None and Path(args.goal).suffix == ".pt":
        raw = torch.load(args.goal, map_location="cpu", weights_only=False)
        goal_state = {
            "cx": float(raw["cx"]),
            "cy": float(raw["cy"]),
            "sin_theta": float(raw["sin_theta"]),
            "cos_theta": float(raw["cos_theta"]),
            "theta_deg": float(raw["theta_deg"]),
        }
    else:
        goal_path = args.goal or args.initial_hdf5
        goal_state = detect_goal_state_from_episode(
            str(goal_path),
            t=int(args.goal_frame),
            processing_resolution=int(cfg.cv_processing_resolution),
        )
    print(
        f"  goal: cx={goal_state['cx']:.2f} cy={goal_state['cy']:.2f} "
        f"theta={goal_state['theta_deg']:.2f}deg"
    )

    planner = Planner(env, cfg)

    # Initial decode + state for the recording.
    rgb0_hwc = env.render()  # (H, W, 3) uint8
    state0 = _initial_cv_state(rgb0_hwc, int(cfg.cv_processing_resolution))
    r0 = _compute_reward_scalar(state0, goal_state, cfg)
    rgb_frames: list[np.ndarray] = [rgb0_hwc]
    per_step: list[dict] = [{
        "t": 0,
        "action": None,
        "reward": float(r0),
        "cv_success": bool(state0["success"]),
        "cx": float(state0["cx"]) if state0["success"] else None,
        "cy": float(state0["cy"]) if state0["success"] else None,
        "theta_deg": float(state0["theta_deg"]) if state0["success"] else None,
        "plan_wall_s": None,
    }]

    H = int(cfg.n_look_ahead)
    A = int(cfg.action_dim)
    act_seq_running = torch.zeros(H, A, device=env.device, dtype=torch.float32)

    # Anchor setup. The anchor lives in normalized [-1, 1] action space
    # because that is the space MPPI samples in. yiru's WorldModelEnv
    # already keeps the action_window in this space (env.step clips to
    # [-1, 1]).
    delta_mode = bool(cfg.delta_mode)
    sample_delta_clip = bool(cfg.sample_delta_clip)
    needs_anchor = delta_mode or sample_delta_clip
    anchor_norm: torch.Tensor | None = None
    if needs_anchor:
        snap = env.snapshot()
        anchor_norm = snap.action_window[-1].detach().clone().to(env.device)
        print(
            f"[{'delta_mode' if delta_mode else 'sample_delta_clip'}] "
            f"anchor (normalized): {anchor_norm.cpu().tolist()}"
        )

    use_warm_start = not args.no_warm_start
    step_each_iter = int(cfg.step_each_iter)
    t0_run = time.time()
    n_actions_done = 0
    converged_plans: list[torch.Tensor] = []
    iteration_logs: list[list[dict]] = []
    plan_wall_s_log: list[float] = []
    while n_actions_done < int(cfg.control_steps):
        n_this = min(
            step_each_iter, int(cfg.control_steps) - n_actions_done,
        )
        snap = env.snapshot()
        t0_plan = time.time()
        init_act = act_seq_running if use_warm_start else None
        a, iter_log = planner.plan_step(
            z_current_unused=None,
            goal_state=goal_state,
            init_act_seq=init_act,
            return_iteration_log=True,
            anchor=anchor_norm,
            snapshot=snap,
        )
        converged = planner.last_stats.act_seq.detach().cpu().clone()
        plan_wall = time.time() - t0_plan
        plan_wall_s_log.append(plan_wall)
        converged_plans.append(converged)
        iteration_logs.append(iter_log)

        # Execute n_this actions for real (advancing env state).
        for s in range(n_this):
            a_s = a if s == 0 else converged[s].to(env.device)
            env.step(a_s)
            rgb_t_hwc = env.render()
            state_t = _initial_cv_state(rgb_t_hwc, int(cfg.cv_processing_resolution))
            r_t = _compute_reward_scalar(state_t, goal_state, cfg)
            rgb_frames.append(rgb_t_hwc)
            per_step.append({
                "t": n_actions_done + s + 1,
                "action": [float(v) for v in a_s.cpu().tolist()],
                "reward": float(r_t),
                "cv_success": bool(state_t["success"]),
                "cx": float(state_t["cx"]) if state_t["success"] else None,
                "cy": float(state_t["cy"]) if state_t["success"] else None,
                "theta_deg": float(state_t["theta_deg"]) if state_t["success"] else None,
                "plan_wall_s": round(plan_wall, 3) if s == 0 else 0.0,
            })
            cv_str = (
                f"cv=({float(state_t['cx']):.1f},{float(state_t['cy']):.1f},"
                f"{float(state_t['theta_deg']):+.1f}deg)"
                if state_t['success'] else "CV-FAIL"
            )
            tail = (
                f"plan={plan_wall:.2f}s" if s == 0 else "(sub-step)"
            )
            print(
                f"  t={n_actions_done + s + 1:02d}  "
                f"r={float(r_t):+.4f}  {cv_str}  {tail}"
            )

        if use_warm_start:
            converged_dev = converged.to(env.device)
            new_tail = converged_dev[-1:].repeat(n_this, 1)
            act_seq_running = torch.cat(
                [converged_dev[n_this:], new_tail], dim=0,
            )

        if needs_anchor:
            anchor_norm = (
                converged[n_this - 1].to(env.device).detach().clone()
            )

        n_actions_done += n_this

    wall_total = time.time() - t0_run
    print(f"\nRun wall time: {wall_total:.1f} s")

    # Output dump.
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack(converged_plans), out_dir / "action_history.pt")
    torch.save(iteration_logs, out_dir / "iteration_log.pt")
    _write_mp4(rgb_frames, out_dir / "trajectory.mp4")

    # Reward curve.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rewards = np.array([row["reward"] for row in per_step])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(len(rewards)), rewards, marker="o", color="tab:blue")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("control step")
    ax.set_ylabel("reward")
    ax.set_title(f"yiru MPPI — {len(rewards)-1} steps")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Summary.
    final = per_step[-1]
    initial = per_step[0]
    final_pos = None
    final_ang_err = None
    final_cos_sim = None
    if final["cv_success"] and initial["cv_success"]:
        dx = final["cx"] - goal_state["cx"]
        dy = final["cy"] - goal_state["cy"]
        final_pos = math.sqrt(dx * dx + dy * dy)
        a_diff = final["theta_deg"] - goal_state["theta_deg"]
        final_ang_err = ((a_diff + 180) % 360) - 180
        final_cos_sim = (
            math.sin(math.radians(final["theta_deg"])) * goal_state["sin_theta"]
            + math.cos(math.radians(final["theta_deg"])) * goal_state["cos_theta"]
        )
    success_strict = (
        final_pos is not None and final_pos <= 5.0
        and final_ang_err is not None and abs(final_ang_err) <= 10.0
    )
    success_cos = (
        final_pos is not None and final_pos <= 5.0
        and final_cos_sim is not None and final_cos_sim >= 0.94
    )
    last_10 = [row["reward"] for row in per_step[-10:]]
    mean_reward_last_10 = float(np.mean(last_10))

    summary = {
        "config": dataclasses.asdict(cfg),
        "config_deviation": {"changed": [], "reason": None,
                             "expected_impact_quantified": None},
        "initial_state": {
            "hdf5": args.initial_hdf5,
            "frame": int(args.initial_frame),
            "cv": initial,
        },
        "goal_state": {
            "cx": goal_state["cx"], "cy": goal_state["cy"],
            "theta_deg": goal_state["theta_deg"],
        },
        "final_state": final,
        "final_pos_distance_px": None if final_pos is None else round(final_pos, 3),
        "final_angle_error_deg": None if final_ang_err is None else round(final_ang_err, 2),
        "final_angle_sim": None if final_cos_sim is None else round(final_cos_sim, 4),
        "success_strict": bool(success_strict),
        "success_cos": bool(success_cos),
        "n_cv_failures": sum(1 for r in per_step if not r["cv_success"]),
        "mean_reward_last_10_steps": round(mean_reward_last_10, 4),
        "wall_time_s": round(wall_total, 1),
        "wall_per_step_s": round(wall_total / max(1, int(cfg.control_steps)), 3),
        "per_step": per_step,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Audit log.
    if bool(cfg.audit_log_enabled) or bool(cfg.debug_mode):
        audit_log = planner.get_audit_log()
        audit_out = (
            Path(cfg.audit_log_path)
            if cfg.audit_log_path else out_dir / "audit_log.json"
        )
        audit_out.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_out, "w") as af:
            json.dump(audit_log, af)
        print(f"  audit log: {audit_out} ({len(audit_log)} entries)")

    print(f"\nWrote artifacts to {out_dir}")
    print(f"  final_pos_distance_px   = {final_pos}")
    print(f"  final_angle_error_deg   = {final_ang_err}")
    print(f"  n_cv_failures           = {summary['n_cv_failures']}")
    print(f"  mean_reward_last_10     = {mean_reward_last_10:.4f}")
    print(f"  success_strict / _cos   = {success_strict} / {success_cos}")


def _initial_cv_state(
    rgb_hwc: np.ndarray, processing_resolution: int,
) -> dict[str, Any]:
    """Run CV on a single HWC uint8 RGB frame, return a scalar state dict
    compatible with ``compute_reward_production``."""
    from interactive_world_sim_mppi._planner import _detect_rgbs_to_state_dict

    batched = _detect_rgbs_to_state_dict(
        rgb_hwc[None], processing_resolution=processing_resolution, n_workers=0,
    )
    return {k: v[0] for k, v in batched.items()}


def _compute_reward_scalar(
    state: dict[str, Any], goal_state: dict[str, Any], cfg: Config,
) -> float:
    """Production reward formula on a single state."""
    if not bool(state["success"]):
        return float(cfg.cv_fail_penalty)
    dx = float(state["cx"]) - float(goal_state["cx"])
    dy = float(state["cy"]) - float(goal_state["cy"])
    pos_dist = math.sqrt(dx * dx + dy * dy)
    cos_delta = max(-1.0, min(
        1.0,
        float(state["sin_theta"]) * float(goal_state["sin_theta"])
        + float(state["cos_theta"]) * float(goal_state["cos_theta"]),
    ))
    return -pos_dist / float(cfg.image_diagonal) - (1.0 - cos_delta)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Closed-loop MPPI driver — yiru-side analog of "
            "scripts/run_mppi_v2.py."
        ),
    )
    ap.add_argument("--task", default="pusht_cam1",
                    help="WorldModelEnv task name (registry key).")
    ap.add_argument("--initial_hdf5", required=True)
    ap.add_argument("--initial_frame", type=int, default=9,
                    help="HDF5 frame index used to initialise the env. "
                         "When init_window_size>1, this is the LAST "
                         "warmup frame.")
    ap.add_argument("--init_window_size", type=int, default=10,
                    help="WorldModelEnv reset window size (warmup frames). "
                         "Default 10 matches production's WM context.")
    ap.add_argument("--goal", default=None,
                    help="Optional path to a state_goal.pt (production "
                         "convention). When omitted, the goal is detected "
                         "on the requested frame of --initial_hdf5.")
    ap.add_argument("--goal_frame", type=int, default=150,
                    help="HDF5 frame index used as the goal when --goal "
                         "is not provided.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default=None,
                    help="Torch device. Default: cuda if available.")

    # Algorithm-change overrides (require --override_reason).
    ap.add_argument("--n_sample", type=int, default=None,
                    help="Override config.n_sample. Algorithm change: "
                         "requires --override_reason.")
    ap.add_argument("--n_update_iter", type=int, default=None,
                    help="Override config.n_update_iter. Algorithm change: "
                         "requires --override_reason.")
    ap.add_argument("--horizon", type=int, default=None,
                    help="Override config.n_look_ahead. Range [1, 50].")
    ap.add_argument("--override_reason", type=str, default=None)
    ap.add_argument("--expected_impact", type=str, default=None)

    # Per-run knobs (no override_reason required).
    ap.add_argument("--control_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--decode_batch_size", type=int, default=None)
    ap.add_argument("--cv_n_workers", type=int, default=None)

    # Driver-side toggle.
    ap.add_argument("--no_warm_start", action="store_true",
                    help="Disable cross-step warm-start (init_act_seq=None "
                         "every call).")

    args = ap.parse_args()
    cfg = Config()
    cfg, _config_deviation = _apply_cli_overrides(cfg, args)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    _run_episode(args, cfg, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
