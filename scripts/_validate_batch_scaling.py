"""Standalone batch-scaling validation for env.rollout.

Hypothesis being tested: at batch=1 the GPU is underutilized so adding
samples is nearly free, but somewhere between N=1 and N=16 the GPU
saturates and per-sample time stops dropping. If true, MPPI's N=16
slowness is intrinsic compute, not a batching bug.

Method: measure a single env.rollout call at each N, identical shapes
and dtype to production. CUDA-aware timing with sync-before-and-after.
3 warmup iterations + 10 measured per N; report median.

Output: table of N, total time, per-sample time, speedup vs N=1,
effective parallelism, GPU util, peak memory.

Run:
    PYTHONUNBUFFERED=1 python -u scripts/_validate_batch_scaling.py \
        > /tmp/batch_scaling.log 2>&1

Delete to revert. No source files modified.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.pusht_wm_env import PushTWMEnv  # noqa: E402

# ── workload constants (match production) ──────────────────────────────

LATENT_C, LATENT_H, LATENT_W = 4, 32, 32
HORIZON = 10           # cfg.n_look_ahead
ACTION_DIM = 4         # cfg.action_dim
DYN_INFER_STEPS = 10   # WM internal denoising substeps per horizon step
WARMUP_ITERS = 3
MEASURE_ITERS = 10
BATCH_SIZES = [1, 2, 4, 8, 16, 32]


# ── GPU util sidecar ───────────────────────────────────────────────────


class GPUUtilSampler:
    """Background thread that polls nvidia-smi every ~100ms.

    Started/stopped per measurement window. Returns the median utilization
    sampled during that window.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._samples: list[int] = []
        self._t: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2.0,
                )
                m = re.search(r"\d+", out)
                if m:
                    self._samples.append(int(m.group(0)))
            except (subprocess.SubprocessError, OSError):
                pass
            self._stop.wait(0.1)

    def start(self) -> None:
        self._stop.clear()
        self._samples = []
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> int | None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)
        return int(median(self._samples)) if self._samples else None


# ── measurement ────────────────────────────────────────────────────────


def measure_rollout(
    env: PushTWMEnv, n: int,
) -> dict[str, Any] | None:
    """Run warmup + measured iterations at batch size n. Return stats or
    None on OOM."""
    device = env.device
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    try:
        z0 = torch.randn(
            n, LATENT_C, LATENT_H, LATENT_W,
            device=device, dtype=torch.float32,
        )
        # Match the rollout's L2-normalised latent magnitude (the WM
        # produces unit-norm chunks at z.shape[2]). Approximation via
        # division by tensor norm — small bias is fine; we're measuring
        # compute, not correctness.
        z0 = z0 / (z0.flatten(1).norm(dim=1, keepdim=True)
                   .view(-1, 1, 1, 1) + 1e-8) * float(LATENT_H)
        actions = torch.zeros(
            n, HORIZON, ACTION_DIM, device=device, dtype=torch.float32,
        )

        # Warmup
        for _ in range(WARMUP_ITERS):
            with torch.no_grad():
                _ = env.rollout(z0, actions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure
        sampler = GPUUtilSampler()
        sampler.start()
        timings_ms: list[float] = []
        for _ in range(MEASURE_ITERS):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = env.rollout(z0, actions)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings_ms.append(1000.0 * (time.perf_counter() - t0))
        gpu_util = sampler.stop()

        peak_gb = (
            torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else 0.0
        )
        return {
            "n": n,
            "median_total_ms": float(median(timings_ms)),
            "min_total_ms": float(min(timings_ms)),
            "max_total_ms": float(max(timings_ms)),
            "all_ms": timings_ms,
            "gpu_util": gpu_util,
            "peak_gb": peak_gb,
        }
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  OOM at N={n}: {exc!r}")
        return None


# ── main ───────────────────────────────────────────────────────────────


def main() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("Loading WM…")
    env = PushTWMEnv(
        str(REPO_ROOT / "outputs/pusht_cam1/checkpoints/best.ckpt"),
        device="cuda:0",
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Latent shape:  ({{N}}, {LATENT_C}, {LATENT_H}, {LATENT_W}) fp32")
    print(f"Actions shape: ({{N}}, {HORIZON}, {ACTION_DIM}) fp32")
    print(f"Iters: {WARMUP_ITERS} warmup + {MEASURE_ITERS} measured per N")
    print(f"Sweep: N ∈ {BATCH_SIZES}")
    print()

    results: list[dict[str, Any]] = []
    for n in BATCH_SIZES:
        print(f"Measuring N={n}…")
        r = measure_rollout(env, n)
        if r is None:
            print(f"  Sweep stopped at N={n} due to OOM.")
            break
        per_sample = r["median_total_ms"] / n
        print(f"  median_total = {r['median_total_ms']:7.1f} ms   "
              f"per-sample = {per_sample:6.2f} ms   "
              f"GPU util = {r['gpu_util']}%   "
              f"peak = {r['peak_gb']:.2f} GB   "
              f"(min/max = {r['min_total_ms']:.1f}/{r['max_total_ms']:.1f} ms)")
        results.append(r)

    if not results:
        print("No measurements collected; aborting.")
        return

    # ── summary table ────────────────────────────────────────────────
    n1 = results[0]["median_total_ms"]
    print()
    print("=" * 96)
    print(f"{'N':>3s} | {'total(ms)':>10s} | {'per-sample(ms)':>15s} | "
          f"{'speedup':>8s} | {'eff parallelism':>16s} | {'GPU util':>9s} | "
          f"{'peak GB':>8s}")
    print("-" * 96)
    for r in results:
        n = r["n"]
        total = r["median_total_ms"]
        per_s = total / n
        # speedup = how much faster than running n×N=1 sequentially
        seq_total = n1 * n
        speedup = seq_total / total
        eff_par = speedup  # samples-of-work-per-unit-time
        gpu = f"{r['gpu_util']}%" if r['gpu_util'] is not None else "?"
        print(f"{n:>3d} | {total:>10.1f} | {per_s:>15.2f} | "
              f"{speedup:>8.2f} | {eff_par:>16.2f} | {gpu:>9s} | "
              f"{r['peak_gb']:>8.2f}")
    print("=" * 96)

    # ── supervisor comparison ────────────────────────────────────────
    print()
    print("Supervisor comparison ('15 fps on RTX 4090'):")
    n1_total_ms = results[0]["median_total_ms"]
    # The WM rollout at N=1 with H=10 horizon × dyn_infer_steps=10 substeps
    # = 100 single-step inferences total.
    single_step_at_n1_ms = n1_total_ms / (HORIZON * DYN_INFER_STEPS)
    fps_at_n1 = 1000.0 / single_step_at_n1_ms
    print(f"  Single-step throughput @ N=1 : {fps_at_n1:.1f} fps "
          f"(WM forward, batch=1, fp32)")

    # Find the largest N we measured
    largest = results[-1]
    n_large = largest["n"]
    total_large_ms = largest["median_total_ms"]
    samples_per_sec = 1000.0 * n_large / total_large_ms
    single_step_at_nlarge_ms = total_large_ms / (HORIZON * DYN_INFER_STEPS)
    fps_at_nlarge = 1000.0 / single_step_at_nlarge_ms
    print(f"  Single-step throughput @ N={n_large}: "
          f"{fps_at_nlarge:.1f} fps (per call)  "
          f"effective {samples_per_sec * HORIZON * DYN_INFER_STEPS:.0f} "
          f"single-step inferences/sec")

    # plan_step lower bound = (N × H × dyn_infer_steps) / effective_throughput_single_step
    # at N=16 (production-ish value) using our best measured per-sample time
    n_target = 16
    target = next((r for r in results if r["n"] == n_target), None)
    if target is not None:
        target_total_ms = target["median_total_ms"]
        n_iter = 5
        n_rollouts_per_plan = n_iter + 1  # 5 inner + 1 post-loop top-K
        rollout_lower_bound_ms = target_total_ms * n_rollouts_per_plan
        print()
        print(f"  Lower bound on plan_step time at N={n_target}, "
              f"n_update_iter={n_iter} (rollouts only, no decode/CV):")
        print(f"    {n_rollouts_per_plan} rollouts × "
              f"{target_total_ms:.0f} ms each = "
              f"{rollout_lower_bound_ms / 1000:.1f} s")
        print(f"  Previously measured plan_step (rollout+decode+CV+viz) = 63.4 s")
        print(f"  Pure-rollout fraction of measured plan_step = "
              f"{100 * rollout_lower_bound_ms / 1000 / 63.4:.1f}%")


if __name__ == "__main__":
    main()
