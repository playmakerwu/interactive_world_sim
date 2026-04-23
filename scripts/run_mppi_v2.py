"""Run the refactored MPPI (rl.mppi.mppi_planner) over a control episode.

Thin wrapper: all heavy logic lives in env.PushTWMEnv and rl.mppi.MPPIPlanner.
Only responsibility here is glue + artifact dumping.

Usage:

    python scripts/run_mppi_v2.py \\
        --config configs/mppi/default.yaml \\
        --initial_hdf5 data/mini/pusht/val/episode_0.hdf5 \\
        --initial_frame 0 \\
        --goal tests/goal_selection/state_goal.pt \\
        --output_dir outputs/mppi/refactor_sanity \\
        --wm_ckpt outputs/pusht_cam1/checkpoints/best.ckpt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.pusht_wm_env import PushTWMEnv  # noqa: E402
from rl.mppi.mppi_planner import MPPIPlanner  # noqa: E402


def _apply_cli_overrides(cfg, args) -> dict:
    """Apply CLI overrides to the loaded OmegaConf in place.

    Returns a ``config_deviation`` dict (always with the same shape).
    Only ``--n_sample`` counts as an algorithm deviation requiring an
    explicit ``--override_reason``; ``--control_steps`` and ``--seed``
    are legitimate per-run knobs (episode length and which run variant)
    that don't change MPPI's algorithmic behavior.

    Split out so tests can exercise override logic without loading the
    WM or constructing PushTWMEnv.
    """
    config_deviation = {"changed": [], "reason": None, "expected_impact_quantified": None}

    if getattr(args, "control_steps", None) is not None:
        cfg.control_steps = int(args.control_steps)
    if getattr(args, "seed", None) is not None:
        cfg.seed = int(args.seed)

    if getattr(args, "n_sample", None) is not None and int(args.n_sample) != int(cfg.n_sample):
        config_deviation["changed"].append(
            f"n_sample: {int(cfg.n_sample)} -> {int(args.n_sample)}"
        )
        cfg.n_sample = int(args.n_sample)

    if config_deviation["changed"]:
        if not getattr(args, "override_reason", None):
            raise SystemExit(
                "ERROR: --n_sample (or other algorithm-level override) used "
                "without --override_reason. Every config deviation must be "
                "justified in writing so the audit trail in summary.json "
                "explains why."
            )
        config_deviation["reason"] = args.override_reason
        config_deviation["expected_impact_quantified"] = (
            getattr(args, "expected_impact", None) or
            "Fewer samples per iteration means sparser coverage of action "
            "space per plan_step. Iterative refinement (n_update_iter) "
            "partially compensates. Empirical performance may differ from "
            "the configured-default N. Replication at the configured N is "
            "recommended for paper numbers."
        )
    return config_deviation


def _write_mp4(frames: list[np.ndarray], out_path: Path, fps: int = 8) -> None:
    if not frames:
        return
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    try:
        for frame in frames:
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        vw.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mppi/default.yaml")
    ap.add_argument("--wm_ckpt", default="outputs/pusht_cam1/checkpoints/best.ckpt")
    ap.add_argument("--initial_hdf5", required=True)
    ap.add_argument("--initial_frame", type=int, default=0)
    ap.add_argument("--goal", required=True, help="Path to a state_goal.pt")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--n_sample", type=int, default=None,
        help="Override config.n_sample at runtime (for VRAM-constrained "
             "local runs). When used, summary.json records a config_deviation "
             "block listing the override and its rationale (which the caller "
             "must pass via --override_reason).",
    )
    ap.add_argument(
        "--override_reason", type=str, default=None,
        help="Free-text justification for any --n_sample override. Stored in "
             "summary.json under config_deviation.reason.",
    )
    ap.add_argument(
        "--expected_impact", type=str, default=None,
        help="Free-text quantified expected impact of the override. Stored in "
             "summary.json under config_deviation.expected_impact_quantified. "
             "If omitted, a generic placeholder is used.",
    )
    ap.add_argument(
        "--control_steps", type=int, default=None,
        help="Override config.control_steps (episode length). Not an "
             "algorithm deviation, so no --override_reason required.",
    )
    ap.add_argument(
        "--seed", type=int, default=None,
        help="Override config.seed (MPPIPlanner sampler RNG). Useful for "
             "multi-seed variance studies. Not an algorithm deviation, so "
             "no --override_reason required.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    # Every runtime override is applied here. Algorithm-level deviations
    # (currently only --n_sample) go into the returned config_deviation
    # block and require --override_reason; per-run knobs (control_steps,
    # seed) mutate cfg silently because they don't change MPPI's algorithm.
    config_deviation = _apply_cli_overrides(cfg, args)
    if config_deviation["changed"]:
        print(f"\n[!! config deviation] {config_deviation}\n")
    print(f"Effective config:\n{OmegaConf.to_yaml(cfg)}")

    print(f"Loading WM from {args.wm_ckpt}")
    env = PushTWMEnv(args.wm_ckpt, device="cuda:0")

    print(f"Encoding initial state from {args.initial_hdf5} frame {args.initial_frame}")
    z = env.load_initial_from_hdf5(args.initial_hdf5, frame_idx=args.initial_frame)
    if z.dim() == 4 and z.shape[0] == 1:
        z = z[0]  # (C, H_lat, W_lat) — planner takes single latent

    print(f"Loading goal from {args.goal}")
    goal = env.load_goal(args.goal)
    print(
        f"  goal: cx={goal['cx']:.2f} cy={goal['cy']:.2f} theta={goal['theta_deg']:.2f}deg"
    )

    planner = MPPIPlanner(env, cfg)

    # ── main control loop ──
    trajectory_latents: list[torch.Tensor] = [z.cpu().clone()]
    action_history: list[torch.Tensor] = []
    rgb_frames: list[np.ndarray] = []
    per_step: list[dict] = []

    # initial decode + state for the recording
    rgb0 = env.decode(z)
    rgb0_u8 = (rgb0.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    state0 = env.estimate_state(rgb0)
    r0 = env.compute_reward(state0, goal, cv_fail_penalty=float(cfg.cv_fail_penalty))
    rgb_frames.append(rgb0_u8)
    per_step.append({
        "t": 0,
        "action": None,
        "reward": float(r0),
        "cv_success": bool(state0["success"]),
        "cx": float(state0["cx"]) if state0["success"] else None,
        "cy": float(state0["cy"]) if state0["success"] else None,
        "theta_deg": float(state0["theta_deg"]) if state0["success"] else None,
        "plan_wall_s": None,
    })

    t0_run = time.time()
    for t in range(int(cfg.control_steps)):
        t0_step = time.time()
        a = planner.plan_step(z, goal)              # (action_dim,)
        z = env.dynamics_step(z, a)                 # (C, H_lat, W_lat)
        rgb_t = env.decode(z)
        rgb_t_u8 = (rgb_t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        state_t = env.estimate_state(rgb_t)
        r_t = env.compute_reward(state_t, goal, cv_fail_penalty=float(cfg.cv_fail_penalty))
        wall = time.time() - t0_step

        trajectory_latents.append(z.cpu().clone())
        action_history.append(a.detach().cpu().clone())
        rgb_frames.append(rgb_t_u8)
        per_step.append({
            "t": t + 1,
            "action": [float(v) for v in a.tolist()],
            "reward": float(r_t),
            "cv_success": bool(state_t["success"]),
            "cx": float(state_t["cx"]) if state_t["success"] else None,
            "cy": float(state_t["cy"]) if state_t["success"] else None,
            "theta_deg": float(state_t["theta_deg"]) if state_t["success"] else None,
            "plan_wall_s": round(wall, 3),
        })
        print(
            f"  t={t+1:02d}  r={float(r_t):+.4f}  "
            + (
                f"cv=({float(state_t['cx']):.1f},{float(state_t['cy']):.1f},"
                f"{float(state_t['theta_deg']):+.1f}deg)"
                if state_t['success'] else "CV-FAIL"
            )
            + f"  plan={wall:.2f}s"
        )

    wall_total = time.time() - t0_run
    print(f"\nRun wall time: {wall_total:.1f} s")

    # ── save artifacts ──
    torch.save(torch.stack(trajectory_latents), out_dir / "trajectory_latents.pt")
    torch.save(torch.stack(action_history), out_dir / "action_history.pt")
    torch.save(trajectory_latents[0], out_dir / "initial_latent.pt")
    _write_mp4(rgb_frames, out_dir / "trajectory.mp4")

    # reward curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rewards = np.array([row["reward"] for row in per_step])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(len(rewards)), rewards, marker="o", color="tab:blue")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("control step")
    ax.set_ylabel("reward")
    ax.set_title(f"refactored MPPI — {len(rewards)-1} steps")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # success summary
    final = per_step[-1]
    initial = per_step[0]
    final_pos = None
    final_ang_err = None
    final_cos_sim = None
    if final["cv_success"] and initial["cv_success"]:
        dx = final["cx"] - goal["cx"]
        dy = final["cy"] - goal["cy"]
        final_pos = math.sqrt(dx * dx + dy * dy)
        a = final["theta_deg"] - goal["theta_deg"]
        final_ang_err = ((a + 180) % 360) - 180
        final_cos_sim = (
            math.sin(math.radians(final["theta_deg"])) * goal["sin_theta"]
            + math.cos(math.radians(final["theta_deg"])) * goal["cos_theta"]
        )

    # Initial measured distance (when CV succeeds on the initial decode).
    initial_distance_to_goal = None
    if initial["cv_success"]:
        idx = initial["cx"] - goal["cx"]
        idy = initial["cy"] - goal["cy"]
        initial_distance_to_goal = math.sqrt(idx * idx + idy * idy)

    # Phase 2 acceptance metrics
    success_strict = (
        final_pos is not None and final_pos <= 5.0
        and final_ang_err is not None and abs(final_ang_err) <= 10.0
    )
    success_cos = (
        final_pos is not None and final_pos <= 5.0
        and final_cos_sim is not None and final_cos_sim >= 0.94
    )
    last_10_rewards = [row["reward"] for row in per_step[-10:]]
    mean_reward_last_10 = float(np.mean(last_10_rewards))

    # Latent drift over the executed trajectory
    latents_t = torch.stack(trajectory_latents).float()  # (T+1, C, H, W)
    z0_flat = latents_t[0].flatten()
    cos_to_z0 = torch.tensor([
        torch.nn.functional.cosine_similarity(
            latents_t[t].flatten().unsqueeze(0), z0_flat.unsqueeze(0)
        ).item() for t in range(latents_t.shape[0])
    ])
    min_cos_to_z0 = float(cos_to_z0.min())

    summary = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "config_deviation": config_deviation,
        "initial_state": {
            "hdf5": args.initial_hdf5,
            "frame": args.initial_frame,
            "cv": initial,
            "distance_to_goal_px": (
                None if initial_distance_to_goal is None
                else round(initial_distance_to_goal, 3)
            ),
        },
        "goal_state": {"cx": goal["cx"], "cy": goal["cy"], "theta_deg": goal["theta_deg"]},
        "final_state": final,
        "final_pos_distance_px": None if final_pos is None else round(final_pos, 3),
        "final_angle_error_deg": None if final_ang_err is None else round(final_ang_err, 2),
        "final_angle_sim": None if final_cos_sim is None else round(final_cos_sim, 4),
        "success_strict": bool(success_strict),
        "success_cos": bool(success_cos),
        "min_latent_cosine_sim_to_z0": round(min_cos_to_z0, 4),
        "n_cv_failures": sum(1 for r in per_step if not r["cv_success"]),
        "mean_reward_last_10_steps": round(mean_reward_last_10, 4),
        "best_reward_in_trajectory": round(
            float(max(row["reward"] for row in per_step)), 4
        ),
        "wall_time_s": round(wall_total, 1),
        "wall_per_step_s": round(wall_total / max(1, int(cfg.control_steps)), 3),
        "per_step": per_step,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # If the run involved any config deviation, also drop a cloud-ready
    # reproduction script that uses the configured-default values.
    if config_deviation["changed"]:
        cloud_script = out_dir / "reproduce_on_cloud.sh"
        cloud_script.write_text(
            "#!/usr/bin/env bash\n"
            "# Cloud reproduction of this run at the configured-default "
            "hyperparameters\n"
            "# (no --n_sample override). Requires GPU with >=20 GiB free for "
            "decoder\n"
            "# attention scratch at the default n_sample.\n"
            "#\n"
            f"# This local run used: {', '.join(config_deviation['changed'])}\n"
            f"# Reason: {config_deviation['reason']}\n"
            "set -e\n"
            "cd \"$(dirname \"$0\")/../../..\"\n"
            f"python scripts/run_mppi_v2.py \\\n"
            f"    --config {args.config} \\\n"
            f"    --wm_ckpt {args.wm_ckpt} \\\n"
            f"    --initial_hdf5 {args.initial_hdf5} \\\n"
            f"    --initial_frame {args.initial_frame} \\\n"
            f"    --goal {args.goal} \\\n"
            f"    --output_dir {args.output_dir}_cloud_repro\n"
        )
        cloud_script.chmod(0o755)
        print(f"  reproduction script: {cloud_script}")

    print(f"\nWrote artifacts to {out_dir}")
    print(f"  final_pos_distance_px      = {final_pos}")
    print(f"  final_angle_error_deg      = {final_ang_err}")
    print(f"  min_latent_cosine_sim_to_z0= {min_cos_to_z0:.4f}")
    print(f"  n_cv_failures              = {summary['n_cv_failures']}")
    print(f"  mean_reward_last_10_steps  = {mean_reward_last_10:.4f}")
    print(f"  success_strict / _cos      = {success_strict} / {success_cos}")


if __name__ == "__main__":
    main()
