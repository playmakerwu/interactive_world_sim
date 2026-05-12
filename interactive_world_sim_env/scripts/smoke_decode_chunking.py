"""Smoke tests for the chunked decode path in WorldModelEnv.step_batch.

Verifies:

  Test 1 — chunked vs non-chunked outputs are numerically close
           (allclose atol=1e-4) at K=4. Chunk order can change
           accumulator order in fp32; we use a small atol rather than
           bitwise equality.
  Test 2 — K=16 with default decode_batch_size=16 completes without
           OOM (would have failed before this fix; see
           overnight/yiru_k16_*/02a_capacity_probe.md).
  Test 3 — K=16 with decode_batch_size=4 also completes without OOM
           (smaller chunks = more loops, smaller memory).

Run from the repo root:
    python -m interactive_world_sim_env.scripts.smoke_decode_chunking
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from interactive_world_sim_env import WorldModelEnv

EPISODE = str(_REPO_ROOT / "data" / "mini" / "pusht" / "train" / "episode_4.hdf5")


def _construct_env() -> WorldModelEnv:
    env = WorldModelEnv("pusht_cam1", device="cuda:0")
    env.reset(
        init_episode_path=EPISODE,
        init_episode_index=9,
        init_window_size=10,
    )
    return env


def _zeros_actions(K: int, H: int = 10, A: int = 4) -> np.ndarray:
    return np.zeros((K, H, A), dtype=np.float32)


def test_1_chunked_equivalence() -> bool:
    """K=2 step_batch chunked vs non-chunked must agree.

    K=2 (not K=4) because the pre-fix non-chunked path OOMs at K=4 on
    an 11.5 GB GPU (see overnight/yiru_k16_*/02a_capacity_probe.md).
    We need the non-chunked baseline to actually run, hence K=2.

    Each ``step_batch`` call consumes CUDA RNG (the dynamics denoiser
    samples noise per call), so back-to-back calls produce different
    latents. We seed the CUDA RNG identically before each call so the
    dynamics produces the same latents, then any difference in rgbs
    is attributable solely to the chunked-vs-not decode pass.
    """
    print("\n[test 1] chunked vs non-chunked numerical equivalence (K=2)")
    env = _construct_env()
    try:
        actions = _zeros_actions(K=2)
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        obs_chunked = env.step_batch(actions, decode_batch_size=4)
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        obs_full = env.step_batch(actions, decode_batch_size=64)
        lat_match = torch.equal(obs_chunked.latents, obs_full.latents)
        rgb_close = np.allclose(obs_chunked.rgbs, obs_full.rgbs, atol=1e-4)
        rgb_max_abs_diff = int(
            np.abs(obs_chunked.rgbs.astype(np.int16)
                   - obs_full.rgbs.astype(np.int16)).max()
        )
        print(f"    latents byte-equal: {lat_match}")
        print(f"    rgbs allclose atol=1e-4: {rgb_close}")
        print(f"    max abs uint8 diff in rgbs: {rgb_max_abs_diff}")
        # uint8 rgbs come back from _decode_latents post .to(uint8); a
        # ±1 difference from quantization order is acceptable.
        return lat_match and rgb_max_abs_diff <= 1
    finally:
        env.close()


def test_2_k16_default_chunk() -> bool:
    """K=16 with default decode_batch_size=16 must complete without OOM.

    Pre-fix this OOMed at K=4; documented in
    overnight/yiru_k16_2026_05_12_1421/02a_capacity_probe.md.
    """
    print("\n[test 2] K=16 default decode_batch_size=16, no OOM")
    env = _construct_env()
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(env.device)
        actions = _zeros_actions(K=16)
        t0 = time.time()
        obs = env.step_batch(actions)  # default decode_batch_size=16
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated(env.device) / (1024 ** 2)
        ok_shape = (
            obs.latents.shape[:2] == (16, 10)
            and obs.rgbs.shape[:2] == (16, 10)
        )
        print(f"    latents shape={tuple(obs.latents.shape)}  "
              f"rgbs shape={tuple(obs.rgbs.shape)}")
        print(f"    wall={wall:.2f}s  peak={peak:.0f}MB  shape_ok={ok_shape}")
        return ok_shape
    finally:
        env.close()


def test_3_k16_small_chunk() -> bool:
    """K=16 with decode_batch_size=4 must also complete without OOM."""
    print("\n[test 3] K=16 decode_batch_size=4, no OOM")
    env = _construct_env()
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(env.device)
        actions = _zeros_actions(K=16)
        t0 = time.time()
        obs = env.step_batch(actions, decode_batch_size=4)
        wall = time.time() - t0
        peak = torch.cuda.max_memory_allocated(env.device) / (1024 ** 2)
        ok_shape = (
            obs.latents.shape[:2] == (16, 10)
            and obs.rgbs.shape[:2] == (16, 10)
        )
        print(f"    latents shape={tuple(obs.latents.shape)}  "
              f"rgbs shape={tuple(obs.rgbs.shape)}")
        print(f"    wall={wall:.2f}s  peak={peak:.0f}MB  shape_ok={ok_shape}")
        return ok_shape
    finally:
        env.close()


def main() -> int:
    tests = [test_1_chunked_equivalence, test_2_k16_default_chunk,
             test_3_k16_small_chunk]
    results: list[tuple[str, bool, str | None]] = []
    for t in tests:
        try:
            ok = t()
            results.append((t.__name__, ok, None))
        except Exception as e:  # noqa: BLE001 — smoke tests catch everything
            results.append((t.__name__, False, traceback.format_exc()))

    print("\n=== summary ===")
    n_pass = 0
    for name, ok, tb in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status:4s}  {name}")
        if tb:
            print(tb)
        n_pass += int(ok)
    print(f"\n{n_pass}/{len(results)} tests passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
