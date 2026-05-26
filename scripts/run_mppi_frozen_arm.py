"""Frozen-arm MPPI experiment — Strategy A (subclass + entry script in one file).

Implements the proposal in
``outputs/diagnostics/freeze_arm_feasibility_20260522_040242Z.md``:
freeze a subset of action dims at the previous-action "anchor" (the
normalized-action value at ``initial_frame - 1`` in the demo HDF5), let
MPPI control the remaining dims, with ``reward_weight=50`` and ``H=30``.

Constraint: ZERO edits to existing files. Both the subclass and the entry
glue live here. We duplicate the small slice of ``scripts/run_mppi_v2.py``
control-loop logic that's needed; nothing imported from there.

Correctness sketch: the subclass overrides ``sample_action_sequences``,
calls ``super()`` (preserving the AR(1) noise + waypoint/clip pipeline,
so the free-dim landscape is byte-equivalent to vanilla MPPI), then
overwrites the frozen dims of every sample with the anchor's frozen-dim
values. Because every sample has identical values on the frozen dims, the
softmax-weighted mean in ``optimize_action_mppi`` reproduces those values
exactly — i.e. the executed action's frozen dims are guaranteed to equal
the anchor with zero numerical drift.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.pusht_wm_env import PushTWMEnv  # noqa: E402
from rl.mppi.mppi_planner import MPPIPlanner  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "mppi" / "default.yaml"
DEFAULT_WM_CKPT = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
WARMUP_WINDOW = 10  # WM hardcoded n_frames=10 (rl/models/world_model.py:62-63)

PAIRS: dict[str, dict[str, Any]] = {
    "medium": {
        "hdf5": "data/mini/pusht/val/episode_3.hdf5",
        "initial_frame": 71,
        "goal": "outputs/goals_from_candidates/20260519_002420/medium_goal.pt",
    },
    "hard": {
        "hdf5": "data/mini/pusht/val/episode_1.hdf5",
        "initial_frame": 59,
        "goal": "outputs/goals_from_candidates/20260519_002420/hard_goal.pt",
    },
    "top1": {
        "hdf5": "data/full/pusht/train/episode_472.hdf5",
        "initial_frame": 11,
        "goal": "outputs/goals_from_candidates/20260525_033022/top1_goal.pt",
    },
    "top5": {
        "hdf5": "data/full/pusht/train/episode_483.hdf5",
        "initial_frame": 5,
        "goal": "outputs/goals_from_candidates/20260525_033022/top5_goal.pt",
    },
    "top10": {
        "hdf5": "data/full/pusht/train/episode_233.hdf5",
        "initial_frame": 26,
        "goal": "outputs/goals_from_candidates/20260525_033022/top10_goal.pt",
    },
    "synthetic_top1plus": {
        "hdf5": "data/full/pusht/train/episode_472.hdf5",
        "initial_frame": 11,
        "goal": "outputs/goals_from_candidates/20260525_163637_synthetic/synthetic_top1plus_goal.pt",
    },
    "synthetic_top1plus_v2": {
        "hdf5": "data/full/pusht/train/episode_472.hdf5",
        "initial_frame": 11,
        "goal": "outputs/goals_from_candidates/20260525_164912_synthetic_v2/synthetic_top1plus_v2_goal.pt",
    },
}


# ─────────────────────────────────────────────────────────────────────
# Subclass
# ─────────────────────────────────────────────────────────────────────

class FrozenArmMPPIPlanner(MPPIPlanner):
    """MPPIPlanner that locks ``frozen_dims`` of every sample to a fixed value.

    The override applies AFTER ``super().sample_action_sequences(...)`` so
    the parent's full pipeline (AR(1) noise, optional waypoint interp,
    optional sample_delta_clip) runs unchanged on the free dims. The
    in-place writes to the returned tensor are safe because the parent
    constructs ``act_seqs`` afresh inside the method; we still ``clone()``
    defensively to insulate from any future refactor that might share the
    tensor with caller state.
    """

    def __init__(
        self,
        env: PushTWMEnv,
        cfg,
        frozen_dims,
        frozen_anchor_norm: torch.Tensor,
    ) -> None:
        super().__init__(env, cfg)
        A = int(self.cfg.action_dim)
        dims = tuple(int(d) for d in frozen_dims)
        if len(set(dims)) != len(dims):
            raise ValueError(f"frozen_dims must be unique; got {dims}")
        for d in dims:
            if d < 0 or d >= A:
                raise ValueError(
                    f"frozen_dims contains {d}, out of range [0, {A})."
                )
        if len(dims) == 0:
            raise ValueError("frozen_dims is empty; nothing to freeze.")
        if len(dims) >= A:
            raise ValueError(
                f"frozen_dims covers all {A} action dims; no free dim left "
                f"for MPPI to optimize over."
            )
        self.frozen_dims = dims

        anchor = frozen_anchor_norm.clone().detach().to(
            device=self.device, dtype=torch.float32,
        )
        if anchor.shape != (A,):
            raise ValueError(
                f"frozen_anchor_norm must have shape ({A},); "
                f"got {tuple(anchor.shape)}"
            )
        self.frozen_anchor_norm = anchor

    def sample_action_sequences(self, act_seq: torch.Tensor) -> torch.Tensor:
        samples = super().sample_action_sequences(act_seq)
        samples = samples.clone()
        for d in self.frozen_dims:
            samples[:, :, d] = self.frozen_anchor_norm[d]
        return samples


# ─────────────────────────────────────────────────────────────────────
# Cfg patch helper
# ─────────────────────────────────────────────────────────────────────

def build_cfg_with_overrides(
    base_cfg_path: Path,
    reward_weight: float,
    horizon: int,
    control_steps: int,
    output_dir: Path,
):
    """Load default.yaml, patch reward_weight + n_look_ahead + control_steps,
    save the resolved cfg next to the run artifacts, return the cfg."""
    cfg = OmegaConf.load(str(base_cfg_path))
    cfg.reward_weight = float(reward_weight)
    cfg.n_look_ahead = int(horizon)
    cfg.control_steps = int(control_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config_resolved.yaml")
    return cfg


# ─────────────────────────────────────────────────────────────────────
# Anchor read + normalize (duplicates the small slice of run_mppi_v2.py)
# ─────────────────────────────────────────────────────────────────────

def read_anchor_norm(
    env: PushTWMEnv,
    hdf5_path: str,
    initial_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (raw_anchor_cpu, normalized_anchor_dev, anchor_frame_idx).

    Mirrors scripts/run_mppi_v2.py:601-608 exactly. The "anchor" is the
    action that DROVE INTO ``initial_frame`` — i.e. the action stored at
    ``initial_frame - 1`` (clipped to 0 for episode start). Normalized via
    the WM's training-time action normalizer to land in the [-1, +1] cube
    that MPPI samples in.
    """
    anchor_frame = max(0, int(initial_frame) - 1)
    with h5py.File(hdf5_path, "r") as f:
        raw_anchor_np = f["action"][anchor_frame]
    raw_anchor = torch.as_tensor(raw_anchor_np, dtype=torch.float32)
    anchor_norm = env._wm.normalizer["action"].normalize(
        raw_anchor.unsqueeze(0),
    ).squeeze(0).to(device=env.device, dtype=torch.float32)
    return raw_anchor, anchor_norm, anchor_frame


# ─────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────

def run_one_frozen_episode(
    *,
    pair_name: str,
    hdf5_path: str,
    initial_frame: int,
    goal_path: str,
    frozen_dims: tuple[int, ...],
    output_dir: Path,
    reward_weight: float,
    horizon: int,
    control_steps: int,
    override_reason: str,
) -> dict[str, Any]:
    """Single frozen-arm control episode.

    Mirrors the relevant slice of ``scripts/run_mppi_v2.py::_run_episode``
    — env construction, warmup-window install, anchor read+normalize, the
    50-step plan/execute loop, and the artifact dump. No private helpers
    from run_mppi_v2 are imported; the glue is duplicated locally.
    Per-iter heatmap and trajectory_combined.mp4 are intentionally skipped
    here (they belong to the v2 viz pipeline whose helpers aren't part of
    the public surface). The pilot doesn't need them; the full run can
    add them later via the rl.visualization public API if desired.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg_with_overrides(
        DEFAULT_CONFIG, reward_weight, horizon, control_steps, output_dir,
    )

    print(f"Effective config:\n{OmegaConf.to_yaml(cfg)}")
    print(f"[frozen_arm] pair={pair_name}  frozen_dims={frozen_dims}")
    print(f"[frozen_arm] Loading WM from {DEFAULT_WM_CKPT}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = PushTWMEnv(
        wm_ckpt_path=str(DEFAULT_WM_CKPT),
        device=device,
        cv_n_workers=int(getattr(cfg, "cv_n_workers", 0) or 0),
    )

    raw_anchor, anchor_norm, anchor_frame = read_anchor_norm(
        env, hdf5_path, initial_frame,
    )
    print(
        f"[frozen_arm] anchor from frame {anchor_frame}: "
        f"raw={raw_anchor.tolist()}  normalized={anchor_norm.tolist()}"
    )
    print(
        f"[frozen_arm] frozen anchor values "
        f"= {[float(anchor_norm[d]) for d in frozen_dims]} "
        f"(at dims {list(frozen_dims)})"
    )

    print(
        f"[frozen_arm] Encoding warmup window ending at frame {initial_frame} "
        f"(window_size={WARMUP_WINDOW})"
    )
    z_history, action_history = env.load_initial_with_warmup(
        hdf5_path=hdf5_path,
        end_frame_index=int(initial_frame),
        window_size=WARMUP_WINDOW,
    )
    env.set_warmup(z_history, action_history)
    z = z_history[-1]  # (C, H_lat, W_lat); rollout uses the warmup when set

    print(f"[frozen_arm] Loading goal from {goal_path}")
    goal = env.load_goal(goal_path)
    print(
        f"  goal: cx={goal['cx']:.2f}  cy={goal['cy']:.2f}  "
        f"theta={goal['theta_deg']:.2f}deg"
    )

    planner = FrozenArmMPPIPlanner(
        env, cfg, frozen_dims=frozen_dims, frozen_anchor_norm=anchor_norm,
    )

    # Initial decode + CV for the step-0 log entry.
    rgb0 = env.decode(z)
    state0 = env.estimate_state(rgb0)
    r0 = env.compute_reward(
        state0, goal, cv_fail_penalty=float(cfg.cv_fail_penalty),
    )

    # Warm-start: anchor.expand(H, A) for ALL dims. Matches run_mppi_v2.py:612-614.
    H = int(cfg.n_look_ahead)
    A = int(cfg.action_dim)
    act_seq_running = anchor_norm.unsqueeze(0).expand(H, A).clone().to(
        device=env.device, dtype=torch.float32,
    )

    trajectory_latents: list[torch.Tensor] = [z.cpu().clone()]
    rgb_frames: list[np.ndarray] = [
        (rgb0.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    ]
    converged_plans: list[torch.Tensor] = []
    executed_actions: list[torch.Tensor] = []
    iteration_logs: list[list[dict[str, Any]]] = []
    per_step: list[dict[str, Any]] = [{
        "t": 0,
        "action": None,
        "reward": float(r0),
        "cv_success": bool(state0["success"]),
        "cx": float(state0["cx"]) if state0["success"] else None,
        "cy": float(state0["cy"]) if state0["success"] else None,
        "theta_deg": float(state0["theta_deg"]) if state0["success"] else None,
        "plan_wall_s": None,
    }]

    t0_run = time.time()
    for t in range(int(cfg.control_steps)):
        t0_plan = time.time()
        a, iter_log = planner.plan_step(
            z, goal, init_act_seq=act_seq_running, return_iteration_log=True,
            anchor=anchor_norm,  # required because sample_delta_clip=true (default)
        )
        converged = planner.last_stats.act_seq.detach().cpu().clone()
        plan_wall = time.time() - t0_plan

        converged_plans.append(converged)
        iteration_logs.append(iter_log)
        a_dev = a.to(env.device)
        executed_actions.append(a_dev.detach().cpu().clone())

        # Execute the action single-step against the WM (warmup-aware).
        actions_one = a_dev.unsqueeze(0).unsqueeze(0)  # (1, 1, A)
        with torch.no_grad():
            traj_b = env.rollout(z.unsqueeze(0), actions_one)
        z = traj_b[0, 1]

        # Slide the warmup window. Matches run_mppi_v2.py:689-695.
        z_history = torch.cat([z_history[1:], z.unsqueeze(0)], dim=0)
        action_history = torch.cat(
            [action_history[1:], a_dev.unsqueeze(0)], dim=0,
        )
        env.set_warmup(z_history, action_history)

        # Decode + CV for log.
        rgb_t = env.decode(z)
        state_t = env.estimate_state(rgb_t)
        r_t = env.compute_reward(
            state_t, goal, cv_fail_penalty=float(cfg.cv_fail_penalty),
        )
        rgb_frames.append(
            (rgb_t.clamp(0, 1).cpu().numpy() * 255)
            .astype(np.uint8).transpose(1, 2, 0)
        )
        trajectory_latents.append(z.cpu().clone())

        frozen_check = ",".join(
            f"{float(a[d]):.6f}" for d in frozen_dims
        )
        per_step.append({
            "t": t + 1,
            "action": [float(v) for v in a.tolist()],
            "reward": float(r_t),
            "cv_success": bool(state_t["success"]),
            "cx": float(state_t["cx"]) if state_t["success"] else None,
            "cy": float(state_t["cy"]) if state_t["success"] else None,
            "theta_deg": float(state_t["theta_deg"]) if state_t["success"] else None,
            "plan_wall_s": round(plan_wall, 3),
        })
        print(
            f"  t={t + 1:02d}  r={float(r_t):+.4f}  "
            + (
                f"cv=({float(state_t['cx']):.1f},{float(state_t['cy']):.1f},"
                f"{float(state_t['theta_deg']):+.1f}deg)"
                if state_t["success"] else "CV-FAIL"
            )
            + f"  plan={plan_wall:.2f}s"
            + f"  exec_frozen=[{frozen_check}]"
        )

        # Warm-start carry-forward: shift converged + repeat-last.
        converged_dev = converged.to(env.device)
        new_tail = converged_dev[-1:].repeat(1, 1)
        act_seq_running = torch.cat(
            [converged_dev[1:], new_tail], dim=0,
        )

    wall_total = time.time() - t0_run

    # ── artifact dump ──
    torch.save(
        torch.stack(executed_actions),                 # (T, A)
        output_dir / "executed_actions.pt",
    )
    torch.save(
        torch.stack(converged_plans),                  # (T, H, A)
        output_dir / "action_history.pt",
    )
    torch.save(
        torch.stack(trajectory_latents),               # (T+1, C, H_lat, W_lat)
        output_dir / "trajectory_latents.pt",
    )
    torch.save(iteration_logs, output_dir / "iteration_log.pt")

    final_state = per_step[-1]
    final_pos_distance_px: float | None = None
    final_angle_error_deg: float | None = None
    if final_state.get("cv_success") and final_state["cx"] is not None:
        final_pos_distance_px = float(
            np.sqrt(
                (final_state["cx"] - goal["cx"]) ** 2
                + (final_state["cy"] - goal["cy"]) ** 2,
            )
        )
        # Signed delta in (-180, 180].
        dth = float(final_state["theta_deg"]) - float(goal["theta_deg"])
        while dth > 180.0:
            dth -= 360.0
        while dth <= -180.0:
            dth += 360.0
        final_angle_error_deg = dth

    rewards = [p["reward"] for p in per_step[1:]]
    mean_reward = float(np.mean(rewards)) if rewards else None

    summary = {
        "experiment": "frozen_arm_strategy_A",
        "pair": pair_name,
        "frozen_dims": list(frozen_dims),
        "frozen_anchor_norm_full": [float(v) for v in anchor_norm.tolist()],
        "frozen_anchor_values_on_frozen_dims": [
            float(anchor_norm[d]) for d in frozen_dims
        ],
        "anchor_frame": anchor_frame,
        "initial_state": {
            "hdf5": hdf5_path,
            "frame": int(initial_frame),
            "cv": {
                "cv_success": bool(state0["success"]),
                "cx": float(state0["cx"]) if state0["success"] else None,
                "cy": float(state0["cy"]) if state0["success"] else None,
                "theta_deg": (
                    float(state0["theta_deg"]) if state0["success"] else None
                ),
                "reward": float(r0),
            },
        },
        "goal_state": {
            "cx": float(goal["cx"]),
            "cy": float(goal["cy"]),
            "theta_deg": float(goal["theta_deg"]),
        },
        "final_state": final_state,
        "final_pos_distance_px": final_pos_distance_px,
        "final_angle_error_deg": final_angle_error_deg,
        "mean_reward_all_steps": mean_reward,
        "wall_time_s": wall_total,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "config_deviation": {
            "changed": [
                f"reward_weight: 200.0 -> {float(cfg.reward_weight)}",
                f"n_look_ahead: 10 -> {int(cfg.n_look_ahead)}",
            ],
            "reason": override_reason,
            "expected_impact_quantified": (
                "Frozen-arm ablation: 2 of 4 action dims locked to the demo "
                "anchor (previous-action value at initial_frame-1, normalized). "
                "Reward weight reduced 200->50 for softer softmax; horizon "
                "extended 10->30 to span more future steps."
            ),
        },
        "per_step": per_step,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote artifacts to {output_dir}")
    print(f"  wall_time_s              = {wall_total:.1f}")
    print(f"  final_pos_distance_px    = {final_pos_distance_px}")
    print(f"  final_angle_error_deg    = {final_angle_error_deg}")
    print(f"  mean_reward_all_steps    = {mean_reward}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Frozen-arm MPPI experiment (Strategy A subclass).",
    )
    ap.add_argument(
        "--pair", choices=tuple(PAIRS), required=True,
        help="Initial state + goal pair from the medium/hard candidates.",
    )
    ap.add_argument(
        "--frozen_dims", required=True,
        help="Comma-separated dim indices to freeze, e.g. '0,1' or '2,3'. "
             "Each index must be in [0, action_dim).",
    )
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument(
        "--horizon", type=int, default=30,
        help="MPPI planning horizon (default: 30; baseline default.yaml is 10).",
    )
    ap.add_argument(
        "--reward_weight", type=float, default=50.0,
        help="Softmax reward multiplier (default: 50; baseline 200).",
    )
    ap.add_argument(
        "--control_steps", type=int, default=50,
        help="Number of executed actions in the episode.",
    )
    ap.add_argument(
        "--override_reason", type=str, default=None,
        help=(
            "Required when --horizon != 10 (matches run_mppi_v2.py's gate). "
            "Free-text justification for any algorithm-level deviation; "
            "stored in summary.json under config_deviation.reason."
        ),
    )
    args = ap.parse_args()

    if int(args.horizon) != 10 and not args.override_reason:
        raise SystemExit(
            "ERROR: --override_reason is required when --horizon != 10. "
            "Every algorithm-level config deviation must be justified in "
            "writing so the audit trail in summary.json explains why."
        )

    frozen_dims = tuple(
        int(s.strip()) for s in args.frozen_dims.split(",") if s.strip()
    )

    pair = PAIRS[args.pair]
    run_one_frozen_episode(
        pair_name=args.pair,
        hdf5_path=pair["hdf5"],
        initial_frame=int(pair["initial_frame"]),
        goal_path=pair["goal"],
        frozen_dims=frozen_dims,
        output_dir=Path(args.output_dir),
        reward_weight=float(args.reward_weight),
        horizon=int(args.horizon),
        control_steps=int(args.control_steps),
        override_reason=args.override_reason or "",
    )


if __name__ == "__main__":
    main()
