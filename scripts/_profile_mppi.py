"""PROFILE: instrument MPPI plan_step end-to-end with CUDA-aware timers.

Self-contained: monkey-patches MPPIPlanner / PushTWMEnv methods with timing
wrappers, runs a short MPPI episode at n_sample=16, prints a stage-by-stage
breakdown. Delete this file to revert.

Run:
    PYTHONUNBUFFERED=1 python -u scripts/_profile_mppi.py \
        > /tmp/mppi_profile.log 2>&1
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.pusht_wm_env import PushTWMEnv  # noqa: E402
from rl.mppi.mppi_planner import MPPIPlanner  # noqa: E402

# ── timing infra ───────────────────────────────────────────────────────

ACC: dict[str, list[float]] = defaultdict(list)
SHAPES_LOGGED: set[str] = set()


@contextmanager
def cuda_timer(name: str):
    """CUDA-aware wall-clock timer. Sync before AND after."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ACC[name].append(time.perf_counter() - t0)


def log_shape_once(key: str, **kwargs: torch.Tensor | tuple) -> None:
    if key in SHAPES_LOGGED:
        return
    SHAPES_LOGGED.add(key)
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            parts.append(f"{k}={tuple(v.shape)}/{v.dtype}/{v.device}")
        else:
            parts.append(f"{k}={v}")
    print(f"[SHAPE] {key}: " + " | ".join(parts))


# ── monkey-patches ─────────────────────────────────────────────────────


def patch_env(env: PushTWMEnv) -> None:
    """Wrap env.rollout, env.decode, env.estimate_state, env.compute_reward."""
    orig_rollout = env.rollout
    orig_decode = env.decode
    orig_estimate_state = env.estimate_state
    orig_compute_reward = env.compute_reward

    def rollout_t(z0, actions):  # type: ignore[no-untyped-def]
        log_shape_once("env.rollout in", z0=z0, actions=actions)
        with cuda_timer("env.rollout (WM)"):
            out = orig_rollout(z0, actions)
        log_shape_once("env.rollout out", traj=out)
        return out

    def decode_t(z):  # type: ignore[no-untyped-def]
        log_shape_once("env.decode in", z=z)
        with cuda_timer("env.decode (decoder)"):
            out = orig_decode(z)
        log_shape_once("env.decode out", rgb=out)
        return out

    def estimate_state_t(rgb):  # type: ignore[no-untyped-def]
        log_shape_once("env.estimate_state in", rgb=rgb)
        with cuda_timer("env.estimate_state (CV labeler loop)"):
            out = orig_estimate_state(rgb)
        log_shape_once("env.estimate_state out", N=len(out["success"]))
        return out

    def compute_reward_t(*args, **kwargs):  # type: ignore[no-untyped-def]
        with cuda_timer("env.compute_reward"):
            return orig_compute_reward(*args, **kwargs)

    env.rollout = rollout_t  # type: ignore[method-assign]
    env.decode = decode_t  # type: ignore[method-assign]
    env.estimate_state = estimate_state_t  # type: ignore[method-assign]
    env.compute_reward = compute_reward_t  # type: ignore[method-assign]


def patch_planner(planner: MPPIPlanner) -> None:
    """Wrap planner.sample_action_sequences and optimize_action_mppi."""
    orig_sample = planner.sample_action_sequences
    orig_optimize = planner.optimize_action_mppi
    orig_eval = planner.evaluate_trajectories
    orig_estimate_final = planner._estimate_final_states
    orig_traj_opt = planner.trajectory_optimization

    def sample_t(act_seq):  # type: ignore[no-untyped-def]
        log_shape_once("sample_action_sequences in", act_seq=act_seq)
        with cuda_timer("sample_action_sequences (AR1 sampler)"):
            out = orig_sample(act_seq)
        log_shape_once("sample_action_sequences out", act_seqs=out)
        return out

    def optimize_t(act_seqs, rewards):  # type: ignore[no-untyped-def]
        log_shape_once("optimize_action_mppi in", act_seqs=act_seqs, rewards=rewards)
        with cuda_timer("optimize_action_mppi (softmax+weighted mean)"):
            return orig_optimize(act_seqs, rewards)

    def eval_t(z, act_seqs, goal):  # type: ignore[no-untyped-def]
        with cuda_timer("evaluate_trajectories (rollout+decode+CV+reward)"):
            return orig_eval(z, act_seqs, goal)

    def estimate_final_t(z_final):  # type: ignore[no-untyped-def]
        log_shape_once("_estimate_final_states in", z_final=z_final)
        with cuda_timer("_estimate_final_states (decode+CV chunked)"):
            return orig_estimate_final(z_final)

    def traj_opt_t(z_current, goal_state, init_act_seq=None):  # type: ignore[no-untyped-def]
        with cuda_timer("trajectory_optimization (full plan_step)"):
            return orig_traj_opt(z_current, goal_state, init_act_seq)

    planner.sample_action_sequences = sample_t  # type: ignore[method-assign]
    planner.optimize_action_mppi = optimize_t  # type: ignore[method-assign]
    planner.evaluate_trajectories = eval_t  # type: ignore[method-assign]
    planner._estimate_final_states = estimate_final_t  # type: ignore[method-assign]
    planner.trajectory_optimization = traj_opt_t  # type: ignore[method-assign]


# ── reporter ───────────────────────────────────────────────────────────


def report() -> None:
    print("\n" + "=" * 86)
    print(f"{'STAGE':<55s} {'total(s)':>9s} {'mean(ms)':>9s} {'calls':>6s}")
    print("-" * 86)
    plan_total = sum(ACC.get("trajectory_optimization (full plan_step)", []))
    rows = []
    for name, samples in ACC.items():
        total = float(np.sum(samples))
        mean_ms = 1000.0 * (total / len(samples)) if samples else 0.0
        pct = (100.0 * total / plan_total) if plan_total > 0 else 0.0
        rows.append((total, name, mean_ms, len(samples), pct))
    rows.sort(reverse=True)
    for total, name, mean_ms, n, pct in rows:
        marker = "*" if name == "trajectory_optimization (full plan_step)" else " "
        print(f"{marker} {name:<53s} {total:>9.3f} {mean_ms:>9.1f} {n:>6d}  {pct:>5.1f}%")
    print("=" * 86)
    print("(* = total includes the others; rows above sum to >100% by design)")
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"\nPeak CUDA memory allocated: {peak_mb:.1f} MiB")


# ── main ───────────────────────────────────────────────────────────────


def main() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    cfg = OmegaConf.load(REPO_ROOT / "configs/mppi/default.yaml")
    cfg.n_sample = 16
    cfg.control_steps = 5
    cfg.seed = 0
    # 11.5 GB GPU OOMs on the post-loop top-K decode (10 traj × 11 frames =
    # 110-frame batch). Use modest chunking to fit; per-chunk timing still
    # shows whether the bottleneck is per-call overhead or compute.
    cfg.decode_batch_size = 8
    print(f"Profile config: n_sample={cfg.n_sample}, n_update_iter={cfg.n_update_iter}, "
          f"n_look_ahead={cfg.n_look_ahead}, control_steps={cfg.control_steps}, "
          f"decode_batch_size={cfg.decode_batch_size}")

    print("Loading WM…")
    env = PushTWMEnv(
        str(REPO_ROOT / "outputs/pusht_cam1/checkpoints/best.ckpt"),
        device="cuda:0",
    )

    z = env.load_initial_from_hdf5(
        REPO_ROOT / "data/mini/pusht/val/episode_0.hdf5", frame_idx=0,
    )
    if z.dim() == 4 and z.shape[0] == 1:
        z = z[0]
    goal = env.load_goal(REPO_ROOT / "tests/goal_selection/state_goal.pt")

    planner = MPPIPlanner(env, cfg)

    patch_env(env)
    patch_planner(planner)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\nRunning {int(cfg.control_steps)} control steps…")
    for t in range(int(cfg.control_steps)):
        with cuda_timer("plan_step (caller side)"):
            a = planner.plan_step(z, goal)
        z = env.dynamics_step(z, a)
        print(f"  t={t+1}  done")

    report()


if __name__ == "__main__":
    main()
