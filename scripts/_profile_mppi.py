"""PROFILE: instrument MPPI plan_step end-to-end with CUDA-aware timers.

Self-contained: monkey-patches MPPIPlanner / PushTWMEnv methods with timing
wrappers, runs MPPI work, prints a stage-by-stage breakdown. Delete this
file to revert.

Two modes:
  * default — runs ``control_steps=5`` plan_steps at ``n_sample=16``,
    prints the historical stage breakdown.
  * ``--n_sweep N1,N2,...`` — runs ONE plan_step at each requested
    ``n_sample``, splits decoder time into in-loop vs post-loop top-K
    viz, writes a CSV under outputs/profiling/ and a per-sample-µs
    markdown table to stdout.

Run:
    PYTHONUNBUFFERED=1 python -u scripts/_profile_mppi.py \
        > /tmp/mppi_profile.log 2>&1

    PYTHONUNBUFFERED=1 python -u scripts/_profile_mppi.py \
        --n_sweep 1,2,4,8 \
        2>&1 | tee outputs/profiling/n_sweep_$(date +%Y%m%d_%H%M%S).log
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
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
IN_TOP_K: bool = False  # set True around _compute_top_k_intermediate_cv to bucket decode/CV calls


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
    """Wrap env.rollout, env.decode, env.estimate_state, env.compute_reward.

    decode and estimate_state inspect the IN_TOP_K module flag so that
    calls inside ``_compute_top_k_intermediate_cv`` get a separate bucket
    from the inner-loop calls.
    """
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
        bucket = (
            "env.decode (post-loop top-K)" if IN_TOP_K
            else "env.decode (in-loop)"
        )
        with cuda_timer(bucket):
            out = orig_decode(z)
        log_shape_once("env.decode out", rgb=out)
        return out

    def estimate_state_t(rgb):  # type: ignore[no-untyped-def]
        log_shape_once("env.estimate_state in", rgb=rgb)
        bucket = (
            "env.estimate_state (post-loop top-K)" if IN_TOP_K
            else "env.estimate_state (in-loop)"
        )
        with cuda_timer(bucket):
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
    """Wrap planner methods. Also wraps _compute_top_k_intermediate_cv to
    set the IN_TOP_K flag during its execution, so decode/CV calls
    inside it route to the correct bucket."""
    orig_sample = planner.sample_action_sequences
    orig_optimize = planner.optimize_action_mppi
    orig_eval = planner.evaluate_trajectories
    orig_estimate_final = planner._estimate_final_states
    orig_traj_opt = planner.trajectory_optimization
    orig_top_k = planner._compute_top_k_intermediate_cv

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

    def top_k_t(*args, **kwargs):  # type: ignore[no-untyped-def]
        global IN_TOP_K
        IN_TOP_K = True
        try:
            with cuda_timer("_compute_top_k_intermediate_cv (post-loop viz)"):
                return orig_top_k(*args, **kwargs)
        finally:
            IN_TOP_K = False

    planner.sample_action_sequences = sample_t  # type: ignore[method-assign]
    planner.optimize_action_mppi = optimize_t  # type: ignore[method-assign]
    planner.evaluate_trajectories = eval_t  # type: ignore[method-assign]
    planner._estimate_final_states = estimate_final_t  # type: ignore[method-assign]
    planner.trajectory_optimization = traj_opt_t  # type: ignore[method-assign]
    planner._compute_top_k_intermediate_cv = top_k_t  # type: ignore[method-assign]


# ── reporters ──────────────────────────────────────────────────────────


def report_default() -> None:
    """Multi-step default report (kept for back-compat with the old mode)."""
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
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"\nPeak CUDA memory allocated: {peak_mb:.1f} MiB")


# ── n-sweep mode ───────────────────────────────────────────────────────

# Stage labels reported in the markdown table. Maps display name → list of
# ACC keys that should be summed for that stage.
SWEEP_STAGES: list[tuple[str, list[str]]] = [
    ("WM rollout (env.rollout)",
        ["env.rollout (WM)"]),
    ("Decoder — in-loop",
        ["env.decode (in-loop)"]),
    ("Decoder — post-loop top-K viz",
        ["env.decode (post-loop top-K)"]),
    ("CV labeler (env.estimate_state)",
        ["env.estimate_state (in-loop)", "env.estimate_state (post-loop top-K)"]),
    ("MPPI optimize (optimize_action_mppi)",
        ["optimize_action_mppi (softmax+weighted mean)"]),
    ("AR(1) sample + clamp",
        ["sample_action_sequences (AR1 sampler)"]),
]


def _aggregate_for_stage(keys: list[str]) -> tuple[float, int]:
    total = 0.0
    calls = 0
    for k in keys:
        if k in ACC:
            total += float(np.sum(ACC[k]))
            calls += len(ACC[k])
    return total, calls


def run_n_sweep(
    n_values: list[int],
    seed: int,
    decode_batch_size: int,
    output_csv: Path,
) -> dict[int, dict[str, tuple[float, int]]]:
    """For each N, build a fresh planner, run ONE plan_step, snapshot
    stage stats. Catches OOM so subsequent N values still run.

    Returns {n: {stage_label: (total_s, calls)}}.
    """
    results: dict[int, dict[str, tuple[float, int]]] = {}

    print("Loading WM…")
    env = PushTWMEnv(
        str(REPO_ROOT / "outputs/pusht_cam1/checkpoints/best.ckpt"),
        device="cuda:0",
    )
    patch_env(env)

    z0 = env.load_initial_from_hdf5(
        REPO_ROOT / "data/mini/pusht/val/episode_0.hdf5", frame_idx=0,
    )
    if z0.dim() == 4 and z0.shape[0] == 1:
        z0 = z0[0]
    goal = env.load_goal(REPO_ROOT / "tests/goal_selection/state_goal.pt")

    base_cfg = OmegaConf.load(REPO_ROOT / "configs/mppi/default.yaml")
    base_cfg.seed = seed
    base_cfg.decode_batch_size = decode_batch_size

    for n in n_values:
        print(f"\n=== N={n} ===")
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        cfg.n_sample = n
        # Pin RNG state across N values for reproducibility comparisons.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        planner = MPPIPlanner(env, cfg)
        patch_planner(planner)

        ACC.clear()  # snapshot per-N
        try:
            with cuda_timer("plan_step (caller side)"):
                _ = planner.plan_step(z0, goal)
            ok = True
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  OOM at N={n}: {exc!r}")
            ok = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        per_stage: dict[str, tuple[float, int]] = {}
        if ok:
            for label, keys in SWEEP_STAGES:
                total_s, calls = _aggregate_for_stage(keys)
                per_stage[label] = (total_s, calls)
                print(f"  {label:<46s} total={total_s*1000:7.1f} ms  calls={calls}")
        else:
            for label, _keys in SWEEP_STAGES:
                per_stage[label] = (math.nan, 0)
        results[n] = per_stage

        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"  Peak CUDA: {peak_mb:.1f} MiB")

    # ── CSV ────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_sample", "stage", "total_s", "per_sample_us", "calls"])
        for n in n_values:
            for label, _keys in SWEEP_STAGES:
                total_s, calls = results[n].get(label, (math.nan, 0))
                per_sample_us = (
                    total_s * 1e6 / n
                    if isinstance(total_s, float) and not math.isnan(total_s) and n > 0
                    else math.nan
                )
                w.writerow([n, label, f"{total_s:.6f}", f"{per_sample_us:.2f}", calls])
    print(f"\nCSV → {output_csv}")

    # ── markdown table ─────────────────────────────────────────────────
    print("\n## Per-sample microseconds (total stage time / N) by stage and N\n")
    header = "| Stage | " + " | ".join(f"N={n}" for n in n_values) + " |"
    sep = "|" + "|".join(["---"] * (len(n_values) + 1)) + "|"
    print(header)
    print(sep)
    for label, _keys in SWEEP_STAGES:
        cells: list[str] = []
        for n in n_values:
            total_s, _calls = results[n].get(label, (math.nan, 0))
            if isinstance(total_s, float) and not math.isnan(total_s) and n > 0:
                cells.append(f"{total_s * 1e6 / n:>8.0f} µs")
            else:
                cells.append("    NaN  ")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print()
    return results


# ── main ───────────────────────────────────────────────────────────────


def _parse_n_sweep(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1:
            raise argparse.ArgumentTypeError(f"N values must be ≥ 1; got {n}")
        out.append(n)
    if not out:
        raise argparse.ArgumentTypeError("--n_sweep must have at least one value")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n_sweep", type=_parse_n_sweep, default=None,
        help="Comma-separated N values (e.g. '1,2,4,8'). When set, runs one "
             "plan_step at each N, splits decoder by in-loop vs post-loop, "
             "and writes a CSV+markdown table. When omitted, runs the "
             "historical 5-step default at n_sample=16.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode_batch_size", type=int, default=8)
    ap.add_argument(
        "--csv_dir", type=str, default="outputs/profiling",
        help="Directory for the n_sweep CSV.",
    )
    args = ap.parse_args()

    if args.n_sweep is not None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = REPO_ROOT / args.csv_dir / f"n_sweep_{ts}.csv"
        print(f"n_sweep mode: N={args.n_sweep}, seed={args.seed}, "
              f"decode_batch_size={args.decode_batch_size}")
        run_n_sweep(
            args.n_sweep,
            seed=args.seed,
            decode_batch_size=args.decode_batch_size,
            output_csv=csv_path,
        )
        return

    # ── default mode (preserved for back-compat) ─────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = OmegaConf.load(REPO_ROOT / "configs/mppi/default.yaml")
    cfg.n_sample = 16
    cfg.control_steps = 5
    cfg.seed = args.seed
    cfg.decode_batch_size = args.decode_batch_size
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

    report_default()


if __name__ == "__main__":
    main()
