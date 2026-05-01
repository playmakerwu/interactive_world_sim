"""FP32 vs FP16 drift test for the IWS world-model rollout.

Loads the same LatentWorldModel checkpoint twice — once kept in FP32, once
``.half()``ed to FP16 — runs identical multi-step dynamics rollouts on both
across N ∈ {1, 4, 16, 64} sample sizes at H=50 horizon, and reports:

  - per-N timing (median of 5 measured iterations, 3 warmup discarded)
  - per-N peak GPU memory
  - per-N L1 drift metrics (overall mean/max, final-step mean/max)
  - per-step drift trajectory at N=16 (mean/max abs diff + cosine sim)

Pure measurement — does not modify any source files. CUDA-aware timing
with sync before AND after every perf_counter call.

Usage (no args):
    python scripts/drift_test_fp16.py 2>&1 | tee /tmp/drift_test.log
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import (  # noqa: E402
    LatentWorldModel,
)


# ── workload constants ─────────────────────────────────────────────────

CKPT_PATH = REPO_ROOT / "outputs/pusht_cam1/checkpoints/best.ckpt"
HYDRA_CFG = REPO_ROOT / "outputs/pusht_cam1/.hydra/config.yaml"
DEVICE = "cuda:0"

LATENT_C, LATENT_H, LATENT_W = 4, 32, 32  # confirmed from the WM checkpoint
ACTION_DIM = 4
HORIZON = 50

WARMUP_ITERS = 3
MEASURE_ITERS = 5
BATCH_SIZES = (1, 4, 16, 64)


def _register_resolvers() -> None:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "torch", lambda x: getattr(torch, x), replace=True,
    )


def _patch_attention_backends() -> None:
    """Prevent flash-attention dispatch on consumer SM-12 GPUs."""
    from torch.nn.attention import SDPBackend
    from interactive_world_sim.algorithms.models.attention import Attention

    cap = torch.cuda.get_device_capability()
    if cap[0] >= 8 and cap[0] != 8:
        _orig_init = Attention.__init__

        def _patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            _orig_init(self, *args, **kwargs)
            self.cuda_backends = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

        Attention.__init__ = _patched_init  # type: ignore[method-assign]


def load_world_model() -> LatentWorldModel:
    """Match mppi-baseline DifferentiableDynamics' loading recipe.

    Overrides n_frames -> 10 (sliding window) and sampling_timesteps -> 10
    (denoising substeps) at load time, since the saved hydra config
    targets training with horizon=1.
    """
    _patch_attention_backends()
    _register_resolvers()
    cfg = OmegaConf.load(HYDRA_CFG)
    cfg.n_frames = 10
    cfg.algorithm.n_frames = 10
    if "diffusion" in cfg.algorithm and "sampling_timesteps" in cfg.algorithm.diffusion:
        cfg.algorithm.diffusion.sampling_timesteps = 10
    if (
        "diffusion" in cfg.algorithm.dynamics
        and "sampling_timesteps" in cfg.algorithm.dynamics.diffusion
    ):
        cfg.algorithm.dynamics.diffusion.sampling_timesteps = 10
    cfg.algorithm.load_ae = None

    wm = LatentWorldModel.load_from_checkpoint(
        str(CKPT_PATH),
        cfg=cfg.algorithm,
        map_location=DEVICE,
        strict=False,
        weights_only=False,
    )
    wm = wm.to(DEVICE).eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm


# ── timing + rollout ───────────────────────────────────────────────────


def _make_inputs(n: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-seed initial latent + zero action sequence for batch n."""
    g = torch.Generator(device="cuda").manual_seed(42 + n)
    z0 = torch.randn(
        n, 1, LATENT_C, LATENT_H, LATENT_W,
        generator=g, device=DEVICE, dtype=torch.float32,
    )
    # Match the WM's per-view L2-normalised latent magnitude ≈ resolution.
    z0_norm = z0.flatten(2).norm(dim=2, keepdim=True).view(n, 1, 1, 1, 1)
    z0 = z0 / (z0_norm + 1e-8) * float(LATENT_H)
    actions = torch.zeros(
        n, HORIZON + 1, ACTION_DIM, device=DEVICE, dtype=torch.float32,
    )
    return z0.to(dtype), actions.to(dtype)


def _rollout(wm: LatentWorldModel, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """LatentWorldModel.dynamics_forward returns (B, T_act, C, H, W)."""
    with torch.no_grad():
        z_seq = wm.dynamics_forward(z0, actions)
    return z_seq


def _time_rollout(
    wm: LatentWorldModel,
    n: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, float, float]:
    """Warmup + measured timed rollout. Returns (last_traj, median_ms, peak_gb)."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    z0, actions = _make_inputs(n, dtype)

    for _ in range(WARMUP_ITERS):
        _ = _rollout(wm, z0, actions)
    torch.cuda.synchronize()

    timings_ms: list[float] = []
    last_traj: torch.Tensor | None = None
    for _ in range(MEASURE_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        last_traj = _rollout(wm, z0, actions)
        torch.cuda.synchronize()
        timings_ms.append(1000.0 * (time.perf_counter() - t0))

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    assert last_traj is not None
    return last_traj, float(median(timings_ms)), peak_gb


# ── main ───────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Torch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"GPU {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Workload: N ∈ {BATCH_SIZES}, H={HORIZON}, latent=({LATENT_C},{LATENT_H},{LATENT_W}), "
          f"action_dim={ACTION_DIM}")
    print()

    print("Loading FP32 model…")
    model_fp32 = load_world_model()
    print("Cloning + halving FP16 model…")
    model_fp16 = copy.deepcopy(model_fp32).half()

    p32 = next(model_fp32.parameters())
    p16 = next(model_fp16.parameters())
    assert p32.dtype == torch.float32, f"FP32 model has dtype {p32.dtype}"
    assert p16.dtype == torch.float16, f"FP16 model has dtype {p16.dtype}"
    print(f"  model_fp32 param dtype = {p32.dtype}")
    print(f"  model_fp16 param dtype = {p16.dtype}")
    print()

    rows: list[dict] = []
    n16_per_step: dict | None = None

    for n in BATCH_SIZES:
        print(f"--- N={n} ---")
        try:
            traj_fp32, t32_ms, mem32_gb = _time_rollout(model_fp32, n, torch.float32)
            print(f"  FP32: {t32_ms:7.1f} ms (median of {MEASURE_ITERS}), "
                  f"peak {mem32_gb:.2f} GB, out shape {tuple(traj_fp32.shape)}")
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  FP32 OOM at N={n}: {exc!r}")
            break

        try:
            traj_fp16, t16_ms, mem16_gb = _time_rollout(model_fp16, n, torch.float16)
            print(f"  FP16: {t16_ms:7.1f} ms (median of {MEASURE_ITERS}), "
                  f"peak {mem16_gb:.2f} GB, out shape {tuple(traj_fp16.shape)}")
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  FP16 OOM at N={n}: {exc!r}")
            break

        diff = (traj_fp32 - traj_fp16.float()).abs()
        overall_mean = float(diff.mean())
        overall_max = float(diff.max())
        final_mean = float(diff[:, -1].mean())
        final_max = float(diff[:, -1].max())
        speedup = t32_ms / t16_ms if t16_ms > 0 else float("nan")
        print(f"  drift: overall mean={overall_mean:.4e} max={overall_max:.4e} | "
              f"final-step mean={final_mean:.4e} max={final_max:.4e}")
        print(f"  speedup (FP32/FP16) = {speedup:.2f}×")

        rows.append({
            "n": n,
            "t32_ms": t32_ms, "t16_ms": t16_ms, "speedup": speedup,
            "mem32_gb": mem32_gb, "mem16_gb": mem16_gb,
            "drift_mean": overall_mean, "drift_max": overall_max,
            "final_mean": final_mean, "final_max": final_max,
        })

        if n == 16:
            T = traj_fp32.shape[1]
            spatial_axes = tuple(range(2, diff.dim()))  # collapse C,H,W
            per_step_mean = diff.mean(dim=spatial_axes).mean(0)  # (T,)
            per_step_max = diff.amax(dim=spatial_axes).amax(0)   # (T,)
            flat32 = traj_fp32.reshape(n, T, -1)
            flat16 = traj_fp16.float().reshape(n, T, -1)
            per_step_cos = F.cosine_similarity(flat32, flat16, dim=-1).mean(0)
            n16_per_step = {
                "mean": per_step_mean.cpu().tolist(),
                "max":  per_step_max.cpu().tolist(),
                "cos":  per_step_cos.cpu().tolist(),
            }
        del traj_fp32, traj_fp16, diff
        torch.cuda.empty_cache()

    # ── summary table ────────────────────────────────────────────────
    print()
    print("=" * 124)
    print(f"{'N':>3s} | {'t_fp32 (ms)':>11s} | {'t_fp16 (ms)':>11s} | "
          f"{'speedup':>8s} | {'mem32 GB':>8s} | {'mem16 GB':>8s} | "
          f"{'drift_mean':>11s} | {'drift_max':>11s} | "
          f"{'final_mean':>11s} | {'final_max':>11s}")
    print("-" * 124)
    for r in rows:
        print(f"{r['n']:>3d} | {r['t32_ms']:>11.1f} | {r['t16_ms']:>11.1f} | "
              f"{r['speedup']:>7.2f}× | {r['mem32_gb']:>8.2f} | {r['mem16_gb']:>8.2f} | "
              f"{r['drift_mean']:>11.4e} | {r['drift_max']:>11.4e} | "
              f"{r['final_mean']:>11.4e} | {r['final_max']:>11.4e}")
    print("=" * 124)

    if n16_per_step is not None:
        print()
        print(f"Per-step drift @ N=16, H={HORIZON}:")
        print(f"{'step':>5s} | {'drift_mean':>11s} | {'drift_max':>11s} | {'cos_sim':>10s}")
        print("-" * 50)
        for i, (m, mx, c) in enumerate(zip(
            n16_per_step["mean"],
            n16_per_step["max"],
            n16_per_step["cos"],
        )):
            print(f"{i:>5d} | {m:>11.4e} | {mx:>11.4e} | {c:>10.6f}")


if __name__ == "__main__":
    main()
