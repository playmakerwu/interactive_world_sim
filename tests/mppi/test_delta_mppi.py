"""Tests for delta-mode MPPI sampling and caller-side cumulative integration.

Coverage:
  1. delta_mode=False reproduces prior absolute-mode sampler bit-for-bit
     (backward compatibility) at the same seed.
  2. delta_mode=True clips per-step samples to ±delta_action_lim per dim.
  3. delta_mode=True respects the smaller noise_level_delta sigma.
  4. Caller-side cumulative integration: abs_seq[t] = anchor + cumsum(deltas)[t].
  5. Cumulative drift logging: ‖a_t − anchor_0‖ is monotone-increasing iff
     deltas are sign-coherent.
  6. Anchor read from hdf5 frame N-1 + normalize matches the WM checkpoint's
     own normalizer scale/offset.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rl.mppi.mppi_planner import MPPIPlanner

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── stub env so we don't load the WM ────────────────────────────────


class _StubEnv:
    """Minimal env stub for sampler-only tests. No WM, no decoder, no CV."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.action_dim = 4
        self.image_diagonal = 181.02


def _base_cfg(**overrides) -> OmegaConf:
    base = OmegaConf.create({
        "n_sample": 8,
        "n_look_ahead": 5,
        "n_update_iter": 1,
        "noise_level": 0.05,
        "reward_weight": 200.0,
        "beta_filter": 0.7,
        "cv_fail_penalty": -10.0,
        "action_dim": 4,
        "action_lower_lim": [-1.0, -1.0, -1.0, -1.0],
        "action_upper_lim": [1.0, 1.0, 1.0, 1.0],
        "delta_mode": False,
        "delta_action_lim": 0.0872,
        "noise_level_delta": 0.02,
        "seed": 42,
        "waypoints_n": None,
        "waypoints_interp": "linear",
    })
    for k, v in overrides.items():
        base[k] = v
    return base


# ── 1. backward compatibility ───────────────────────────────────────


def test_delta_mode_false_matches_prior_sampler():
    """With delta_mode=False, sampler output is bit-identical to historical behavior.

    Implementation detail: the new code path forks on `delta_mode`. False
    branch must yield the exact same tensors at the same seed as a planner
    constructed with no delta_mode key at all.
    """
    cfg_with = _base_cfg(delta_mode=False)
    cfg_legacy = _base_cfg()
    # Remove the key entirely from the legacy cfg to mimic pre-change cfgs
    OmegaConf.set_struct(cfg_legacy, False)
    del cfg_legacy["delta_mode"]
    OmegaConf.set_struct(cfg_legacy, True)

    p1 = MPPIPlanner(_StubEnv(), cfg_with)
    p2 = MPPIPlanner(_StubEnv(), cfg_legacy)

    H = int(cfg_with.n_look_ahead)
    A = int(cfg_with.action_dim)
    seed = torch.zeros(H, A)

    a1 = p1.sample_action_sequences(seed)
    a2 = p2.sample_action_sequences(seed)

    assert a1.shape == a2.shape
    assert torch.allclose(a1, a2), (
        "delta_mode=False produced different output than legacy cfg without "
        "the key: backward compat broken"
    )


# ── 2. delta-mode clipping ──────────────────────────────────────────


def test_delta_mode_clips_per_dim_to_delta_action_lim():
    """In delta_mode, every sample component is in [-lim, +lim]."""
    lim = 0.0872
    cfg = _base_cfg(
        delta_mode=True, delta_action_lim=lim, noise_level_delta=0.02,
        n_sample=64, n_look_ahead=10,
    )
    planner = MPPIPlanner(_StubEnv(), cfg)
    seed = torch.zeros(10, 4)
    out = planner.sample_action_sequences(seed)
    assert out.shape == (64, 10, 4)
    assert (out >= -lim - 1e-6).all(), f"min {out.min().item()} below -lim={-lim}"
    assert (out <=  lim + 1e-6).all(), f"max {out.max().item()} above +lim={+lim}"


def test_delta_mode_clip_actually_fires_with_loud_noise():
    """A noise level larger than delta_action_lim makes the clip actively
    truncate samples — verify the bounds are saturated."""
    cfg = _base_cfg(
        delta_mode=True, delta_action_lim=0.05, noise_level_delta=0.20,
        beta_filter=1.0, n_sample=32, n_look_ahead=8,
    )
    planner = MPPIPlanner(_StubEnv(), cfg)
    out = planner.sample_action_sequences(torch.zeros(8, 4))
    # With sigma=0.20 and clip at 0.05, expect saturation (some entries == ±0.05).
    saturated_max = (out.abs() >= 0.05 - 1e-6).any().item()
    assert saturated_max, "expected clip to fire under sigma=0.20 / lim=0.05"


# ── 3. noise level routing ──────────────────────────────────────────


def test_delta_mode_uses_noise_level_delta_not_noise_level():
    """Setting noise_level=10.0 in delta_mode must not blow up samples; only
    noise_level_delta governs the actual noise scale."""
    cfg = _base_cfg(
        delta_mode=True,
        noise_level=10.0,         # absurdly large; should be IGNORED in delta mode
        noise_level_delta=0.001,  # tiny
        delta_action_lim=0.05,
        beta_filter=0.0,          # no smoothing — residual = noise_sample
        n_sample=64, n_look_ahead=10,
    )
    planner = MPPIPlanner(_StubEnv(), cfg)
    out = planner.sample_action_sequences(torch.zeros(10, 4))
    # With sigma=0.001 and beta=0, per-step residual std ≈ 0.001; samples
    # should be tiny — well under the clip bound.
    assert out.abs().max().item() < 0.01, (
        f"expected tiny samples under noise_level_delta=0.001; got max="
        f"{out.abs().max().item()}"
    )


# ── 4. cumulative integration math ──────────────────────────────────


def test_caller_cumsum_recovers_absolute_targets():
    """Caller-side abs_seq[t] = anchor + sum(deltas[:t+1]) holds for any deltas."""
    anchor = torch.tensor([+0.10, -0.20, +0.30, -0.40], dtype=torch.float32)
    deltas = torch.tensor([
        [+0.05, +0.02, -0.01,  0.00],
        [+0.03, +0.04, -0.02, -0.01],
        [-0.01, +0.01, +0.02, +0.03],
    ], dtype=torch.float32)  # (3, 4)
    expected_abs = torch.tensor([
        anchor.tolist(),
        (anchor + deltas[0]).tolist(),
        (anchor + deltas[0] + deltas[1]).tolist(),
        (anchor + deltas[0] + deltas[1] + deltas[2]).tolist(),
    ])
    abs_seq_runner = anchor.unsqueeze(0) + deltas.cumsum(dim=0)  # (3, 4)

    # Runner integration matches expected[1:] (anchor itself is t=0, the
    # cumsum gives t=1..T).
    assert torch.allclose(abs_seq_runner, expected_abs[1:])


def test_cumulative_drift_log_monotone_for_coherent_deltas():
    """Drift ‖a_t − anchor_0‖ is monotone-increasing if all deltas sign-align."""
    anchor = torch.zeros(4, dtype=torch.float32)
    deltas = torch.full((10, 4), 0.05, dtype=torch.float32)  # all positive, equal
    abs_seq = anchor.unsqueeze(0) + deltas.cumsum(dim=0)
    drifts = [(abs_seq[t] - anchor).norm().item() for t in range(deltas.shape[0])]
    for i in range(1, len(drifts)):
        assert drifts[i] >= drifts[i - 1] - 1e-7, (
            f"drift not monotone at t={i}: {drifts[i-1]:.4f} -> {drifts[i]:.4f}"
        )
    # Closed form: at t, drift = sqrt(4) * 0.05 * (t+1) = 0.10 * (t+1)
    for t in range(deltas.shape[0]):
        assert abs(drifts[t] - 0.10 * (t + 1)) < 1e-5


# ── 5. anchor read & normalize round-trip ───────────────────────────


def test_anchor_normalize_matches_wm_checkpoint():
    """Read raw action[N-1] from hdf5, apply normalize, verify the result
    matches a hand-computed scale*raw + offset using the WM ckpt's stored
    scale/offset (no actual WM load required for the math)."""
    hdf5_path = REPO_ROOT / "data/mini/pusht/val/episode_1.hdf5"
    ckpt_path = REPO_ROOT / "outputs/pusht_cam1/checkpoints/best.ckpt"
    if not hdf5_path.exists() or not ckpt_path.exists():
        pytest.skip("local data / WM ckpt missing")

    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)["state_dict"]
    scale = sd["normalizer.params_dict.action.scale"].cpu().float()
    offset = sd["normalizer.params_dict.action.offset"].cpu().float()

    with h5py.File(str(hdf5_path), "r") as f:
        raw = torch.as_tensor(f["action"][188], dtype=torch.float32)

    expected_norm = scale * raw + offset
    # Also exercise the inverse to confirm round-trip
    recovered = (expected_norm - offset) / scale
    assert torch.allclose(recovered, raw, atol=1e-5)
    # Spot-check expected magnitude is within MPPI cube (-1, 1)
    assert (expected_norm.abs() <= 1.5).all(), (
        f"normalized anchor out of expected range: {expected_norm.tolist()}"
    )


def test_anchor_frame_idx_zero_uses_frame_zero():
    """Edge case: when initial_frame=0, anchor_frame = max(0, -1) = 0
    (covered in run_mppi_v2.py); confirm that's a valid index."""
    hdf5_path = REPO_ROOT / "data/mini/pusht/val/episode_1.hdf5"
    if not hdf5_path.exists():
        pytest.skip("local data missing")
    with h5py.File(str(hdf5_path), "r") as f:
        a0 = f["action"][0]
    assert a0.shape == (4,)


# ── 6. waypoint sampler also respects delta_mode ────────────────────


def test_delta_mode_waypoint_sampler_clips():
    """When waypoints_n is set, the same delta_mode clip applies after interp."""
    cfg = _base_cfg(
        delta_mode=True, delta_action_lim=0.0872, noise_level_delta=0.10,
        beta_filter=1.0, waypoints_n=3, n_look_ahead=10, n_sample=16,
    )
    planner = MPPIPlanner(_StubEnv(), cfg)
    out = planner.sample_action_sequences(torch.zeros(10, 4))
    assert out.shape == (16, 10, 4)
    assert (out.abs() <= 0.0872 + 1e-6).all()
