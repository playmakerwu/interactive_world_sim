"""Drift quantification for env.PushTWMEnv.rollout at long horizons.

The IWS WM's internal sliding 10-frame window already supports H up to
``env.pusht_wm_env.MAX_HORIZON`` natively. These tests measure how much
the WM's predicted latent drifts from the encoded initial latent as H
grows, so we know whether reward signals at long horizons are still
informative or already dominated by hallucination.

Methodology:
  * Encode one real frame from data/mini/pusht/val/episode_2 to z0.
  * Roll out H steps under two action policies:
      - zero actions  (passive: how much does the WM hallucinate motion
                       on its own, with no input?)
      - demo actions  (real ALOHA actions from the dataset for this
                       segment — exercises the WM the same way an MPPI
                       sample would)
  * For each (policy, H), record:
      cos_sim(z0, z_H)        — cosine similarity, [-1, 1]
      l2_drift = ||z_H - z0|| — euclidean distance
      decoded_z_H_finite      — sanity: decoded RGB has no NaN/Inf

Hard failures:
  * NaN in any rolled-out latent
  * cos_sim < 0.5 (catastrophic: rollout chaining is broken)

Soft warning (printed, does not fail the test):
  * H=50 cos_sim < 0.85

Output report: outputs/drift_report.json + per-test stdout table.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from env.pusht_wm_env import MAX_HORIZON, PUSHT_CAMERA_KEY, PushTWMEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = REPO_ROOT / "outputs" / "pusht_cam1" / "checkpoints" / "best.ckpt"
SAMPLE_HDF5 = REPO_ROOT / "data" / "mini" / "pusht" / "val" / "episode_2.hdf5"
SAMPLE_FRAME_IDX = 27  # easy_pair_2 starting frame — matches Item C smoke
REPORT_PATH = REPO_ROOT / "outputs" / "drift_report.json"
RES = 128

HORIZONS = (10, 20, 30, 40, 50)
SOFT_WARN_COS_AT_H50 = 0.85


def _have_gpu_and_ckpt() -> bool:
    return torch.cuda.is_available() and CKPT_PATH.exists()


@pytest.fixture(scope="module")
def env() -> PushTWMEnv:
    if not _have_gpu_and_ckpt():
        pytest.skip("CUDA + WM checkpoint required")
    return PushTWMEnv(str(CKPT_PATH), device="cuda:0", resolution=RES)


@pytest.fixture(scope="module")
def starting_z0(env) -> torch.Tensor:
    """Encode the real episode_2 frame 27 RGB to a single z0 latent."""
    if not SAMPLE_HDF5.exists():
        pytest.skip(f"missing dataset: {SAMPLE_HDF5}")
    with h5py.File(str(SAMPLE_HDF5), "r") as f:
        raw = f[f"obs/images/{PUSHT_CAMERA_KEY}"][SAMPLE_FRAME_IDX]
    h, w = raw.shape[:2]
    s = min(h, w)
    cropped = raw[(h - s) // 2 : (h - s) // 2 + s,
                  (w - s) // 2 : (w - s) // 2 + s]
    resized = cv2.resize(cropped, (RES, RES), interpolation=cv2.INTER_AREA)
    pre = resized.astype(np.float32) / 255.0
    rgb_t = torch.from_numpy(pre).permute(2, 0, 1)
    return env.encode(rgb_t)  # (C, H_lat, W_lat)


@pytest.fixture(scope="module")
def demo_actions() -> torch.Tensor:
    """Real ALOHA action sequence from episode_2 starting at SAMPLE_FRAME_IDX.

    Returns a (MAX_HORIZON, action_dim) tensor — long enough to slice
    for every horizon under test.
    """
    if not SAMPLE_HDF5.exists():
        pytest.skip(f"missing dataset: {SAMPLE_HDF5}")
    with h5py.File(str(SAMPLE_HDF5), "r") as f:
        actions = f["action"][SAMPLE_FRAME_IDX:SAMPLE_FRAME_IDX + MAX_HORIZON]  # (MAX_HORIZON, A)
    return torch.from_numpy(actions.astype(np.float32))


def _measure_drift(
    env: PushTWMEnv,
    z0: torch.Tensor,
    actions_full: torch.Tensor,   # (MAX_HORIZON, A) on any device
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[int, dict[str, float]]:
    """Run rollouts at each H, return per-H {cos_sim, l2_drift, finite}."""
    z0_flat = z0.flatten().float()
    z0_norm = float(z0_flat.norm())
    out: dict[int, dict[str, float]] = {}
    for H in horizons:
        actions = actions_full[:H].unsqueeze(0).to(env.device)  # (1, H, A)
        traj = env.rollout(z0.unsqueeze(0), actions)            # (1, H+1, C, h, w)
        z_H = traj[0, -1]
        cos = float(F.cosine_similarity(
            z0_flat.unsqueeze(0), z_H.flatten().float().unsqueeze(0),
        ).item())
        l2 = float((z_H - z0).flatten().float().norm().item())
        finite = bool(torch.all(torch.isfinite(traj)).item())
        out[H] = {"cos_sim": cos, "l2_drift": l2, "finite": finite,
                  "z0_norm": z0_norm}
    return out


def _print_report(label: str, results: dict[int, dict[str, float]]) -> None:
    print(f"\n=== {label} drift report ===")
    print(f"{'H':>4} {'cos_sim':>9} {'l2_drift':>10} {'finite':>7}")
    for H in sorted(results):
        r = results[H]
        print(f"{H:>4} {r['cos_sim']:>9.4f} {r['l2_drift']:>10.4f} "
              f"{str(r['finite']):>7}")


def _hard_fail_on_nan_or_catastrophic(label: str, results: dict[int, dict[str, float]]) -> None:
    for H, r in results.items():
        assert r["finite"], f"{label} H={H}: NaN or Inf in rolled-out latents"
        assert not math.isnan(r["cos_sim"]), f"{label} H={H}: cos_sim is NaN"
        assert r["cos_sim"] > 0.5, (
            f"{label} H={H}: catastrophic drift cos_sim={r['cos_sim']:.4f}; "
            f"WM rollout chaining likely broken"
        )


def _soft_warn(label: str, results: dict[int, dict[str, float]]) -> None:
    cos50 = results.get(50, {}).get("cos_sim")
    if cos50 is not None and cos50 < SOFT_WARN_COS_AT_H50:
        print(
            f"\nSOFT WARNING: {label} cos_sim at H=50 is {cos50:.4f} "
            f"(< {SOFT_WARN_COS_AT_H50}). Sliding-window rollout has "
            f"noticeable drift; reward signal at H=50 may be less reliable "
            f"than at H=10. Consider H<=30 instead."
        )


def _write_report(zero_results, demo_results) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "starting_state": {
            "hdf5": str(SAMPLE_HDF5.relative_to(REPO_ROOT)),
            "frame_idx": SAMPLE_FRAME_IDX,
        },
        "horizons_under_test": list(HORIZONS),
        "soft_warning_threshold_cos_at_h50": SOFT_WARN_COS_AT_H50,
        "zero_action": {str(H): r for H, r in zero_results.items()},
        "demo_action": {str(H): r for H, r in demo_results.items()},
    }
    REPORT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {REPORT_PATH.relative_to(REPO_ROOT)}")


# ─── 1. zero-action drift quantification ───────────────────────────────

def test_zero_action_drift_quantification(env, starting_z0):
    zero_actions = torch.zeros(MAX_HORIZON, env.action_dim)
    results = _measure_drift(env, starting_z0, zero_actions)
    _print_report("zero-action", results)
    _hard_fail_on_nan_or_catastrophic("zero-action", results)
    _soft_warn("zero-action", results)
    pytest.module_zero_action_results = results  # type: ignore[attr-defined]


# ─── 2. demo-action drift quantification ───────────────────────────────

def test_demo_action_drift_quantification(env, starting_z0, demo_actions):
    results = _measure_drift(env, starting_z0, demo_actions)
    _print_report("demo-action", results)
    _hard_fail_on_nan_or_catastrophic("demo-action", results)
    _soft_warn("demo-action", results)
    # Persist combined report (only if the zero-action test ran first;
    # otherwise build a partial report so the file always exists).
    zero_results = getattr(pytest, "module_zero_action_results", None)
    if zero_results is None:
        zero_results = {}
    _write_report(zero_results, results)
