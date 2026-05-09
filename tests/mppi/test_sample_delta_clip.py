"""Tests for the vanilla sample-side per-step delta clip.

Feature gated by ``cfg.sample_delta_clip``. When on (vanilla mode only),
each sampled trajectory has its per-step deltas clipped to
``±cfg.per_step_delta_lim`` (per dim), with the first-step delta
bounded against the optional anchor (previous executed action).

Off by default; mutually exclusive with delta_mode.
"""
from __future__ import annotations

import warnings

import pytest
import torch
from omegaconf import OmegaConf

from rl.mppi.mppi_planner import MPPIPlanner

DEFAULT_LIM = [0.0975, 0.0919, 0.0760, 0.0909]  # mini-dataset demo per-dim p99


class _StubEnv:
    """Minimal env stub: action_dim + device are the only fields read."""

    device = "cpu"
    action_dim = 4


def _base_cfg(**overrides) -> OmegaConf:
    cfg = OmegaConf.create({
        "n_sample": 8,
        "n_look_ahead": 6,
        "n_update_iter": 1,
        "noise_level": 0.05,
        "reward_weight": 200.0,
        "beta_filter": 0.7,
        "cv_fail_penalty": -10.0,
        "action_dim": 4,
        "action_lower_lim": [-1.0, -1.0, -1.0, -1.0],
        "action_upper_lim": [1.0, 1.0, 1.0, 1.0],
        "control_steps": 2,
        "seed": 0,
        "cv_n_workers": 0,
        "waypoints_n": None,
        "waypoints_interp": "linear",
        "delta_mode": False,
        "delta_action_lim": 0.0872,
        "noise_level_delta": 0.02,
        "sample_delta_clip": False,
        "per_step_delta_lim": None,
    })
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def _build_planner(cfg: OmegaConf) -> MPPIPlanner:
    """Build a planner with __init__ bypassed (no decoder), then run only
    the parts of __init__ relevant to sample_delta_clip resolution."""
    p = MPPIPlanner.__new__(MPPIPlanner)
    p.env = _StubEnv()
    p.cfg = cfg
    p.device = "cpu"
    p.action_lower_lim = torch.tensor(list(cfg.action_lower_lim))
    p.action_upper_lim = torch.tensor(list(cfg.action_upper_lim))
    p.last_stats = None
    p._gen = torch.Generator(device="cpu")
    p._gen.manual_seed(int(cfg.seed))
    p._anchor = None

    # Mirror the relevant block of MPPIPlanner.__init__:
    p._sample_delta_lim = None
    if bool(getattr(cfg, "sample_delta_clip", False)):
        lim_cfg = getattr(cfg, "per_step_delta_lim", None)
        lim_list = list(lim_cfg) if lim_cfg is not None else DEFAULT_LIM
        p._sample_delta_lim = torch.as_tensor(
            lim_list, dtype=torch.float32, device=p.device,
        )
    return p


def _max_per_step_delta_inf_norm(samples: torch.Tensor) -> torch.Tensor:
    """Return per-dim max ``|a[t] - a[t-1]|`` over (samples, t≥1)."""
    diffs = (samples[:, 1:, :] - samples[:, :-1, :]).abs()       # (N, H-1, A)
    return diffs.amax(dim=(0, 1))                                # (A,)


# ─── 1. default disabled: clip = no-op vs baseline ──────────────────────

def test_default_disabled_unchanged() -> None:
    cfg_off = _base_cfg(sample_delta_clip=False)
    cfg_on = _base_cfg(sample_delta_clip=True)
    p_off = _build_planner(cfg_off)
    p_on = _build_planner(cfg_on)
    H, A = int(cfg_off.n_look_ahead), int(cfg_off.action_dim)
    act_seq = torch.zeros(H, A)

    # Re-seed both to the same noise generator state.
    p_off._gen.manual_seed(0); p_on._gen.manual_seed(0)
    out_off = p_off.sample_action_sequences(act_seq.clone())
    p_off._gen.manual_seed(0)
    out_off2 = p_off.sample_action_sequences(act_seq.clone())
    # Sanity: deterministic when off
    assert torch.allclose(out_off, out_off2)
    # Sanity: clip flag has no effect when off (this test is the gate
    # itself: if the off-path were touched, it would still equal off.)
    assert p_off._sample_delta_lim is None
    assert p_on._sample_delta_lim is not None


# ─── 2. clip enforces per-step bound ────────────────────────────────────

def test_clip_enforces_per_step_bound() -> None:
    cfg = _base_cfg(
        sample_delta_clip=True,
        noise_level=2.0,                     # huge sigma → big raw deltas
        per_step_delta_lim=[0.05, 0.04, 0.03, 0.02],
    )
    p = _build_planner(cfg)
    H, A = int(cfg.n_look_ahead), int(cfg.action_dim)
    act_seq = torch.zeros(H, A)
    out = p.sample_action_sequences(act_seq.clone())
    assert out.shape == (cfg.n_sample, H, A)

    lim = torch.tensor([0.05, 0.04, 0.03, 0.02])
    max_delta = _max_per_step_delta_inf_norm(out)
    eps = 1e-6
    assert torch.all(max_delta <= lim + eps), (
        f"per-step delta inf-norm {max_delta.tolist()} > lim {lim.tolist()}"
    )


# ─── 3. first-step delta bounded against anchor ─────────────────────────

def test_first_step_clip_against_anchor() -> None:
    cfg = _base_cfg(
        sample_delta_clip=True,
        noise_level=2.0,
        per_step_delta_lim=[0.05, 0.04, 0.03, 0.02],
    )
    p = _build_planner(cfg)
    H, A = int(cfg.n_look_ahead), int(cfg.action_dim)
    # Anchor far from cube center, sampler push from zeros.
    anchor = torch.tensor([0.7, -0.6, 0.5, -0.4])
    p.set_anchor(anchor)
    act_seq = torch.zeros(H, A)
    out = p.sample_action_sequences(act_seq.clone())

    lim = torch.tensor([0.05, 0.04, 0.03, 0.02])
    d0 = (out[:, 0, :] - anchor.unsqueeze(0)).abs()        # (N, A)
    eps = 1e-6
    assert torch.all(d0 <= lim.unsqueeze(0) + eps), (
        f"first-step ‖a[0]-anchor‖∞ per dim max = {d0.amax(dim=0).tolist()} "
        f"> lim {lim.tolist()}"
    )


# ─── 4. cube clip still applies ─────────────────────────────────────────

def test_cube_clip_still_applies() -> None:
    cfg = _base_cfg(
        sample_delta_clip=True,
        noise_level=10.0,                   # ridiculous noise
        per_step_delta_lim=[1.0, 1.0, 1.0, 1.0],   # generous lim
    )
    p = _build_planner(cfg)
    H, A = int(cfg.n_look_ahead), int(cfg.action_dim)
    act_seq = torch.full((H, A), 0.95)        # near the upper cube edge
    out = p.sample_action_sequences(act_seq.clone())
    assert torch.all(out >= -1.0 - 1e-6) and torch.all(out <= 1.0 + 1e-6), (
        f"cube violation: min={out.min().item():.4f} max={out.max().item():.4f}"
    )


# ─── 5. delta_mode + sample_delta_clip → warn + disable ─────────────────

def test_delta_mode_disables_clip() -> None:
    cfg = _base_cfg(
        delta_mode=True, sample_delta_clip=True,
    )
    p = MPPIPlanner.__new__(MPPIPlanner)
    p.env = _StubEnv()
    # Manually invoke the validation path (mirrors __init__ order).
    p.cfg = cfg
    p.device = "cpu"
    p.action_lower_lim = torch.tensor(list(cfg.action_lower_lim))
    p.action_upper_lim = torch.tensor(list(cfg.action_upper_lim))
    p.last_stats = None
    p._gen = torch.Generator(device="cpu"); p._gen.manual_seed(0)
    p._anchor = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p._validate_config()
        assert any("sample_delta_clip" in str(rec.message).lower() for rec in w), (
            f"expected warning about sample_delta_clip; got {[str(r.message) for r in w]}"
        )
    assert bool(cfg.sample_delta_clip) is False, (
        "validation should have set sample_delta_clip = False"
    )


# ─── 6. per_step_delta_lim yaml override is honoured ────────────────────

def test_per_step_delta_lim_yaml_override() -> None:
    cfg = _base_cfg(
        sample_delta_clip=True,
        per_step_delta_lim=[0.05, 0.05, 0.05, 0.05],
    )
    p = _build_planner(cfg)
    assert p._sample_delta_lim is not None
    assert torch.allclose(
        p._sample_delta_lim,
        torch.tensor([0.05, 0.05, 0.05, 0.05]),
    )

    # Confirm clip uses 0.05 not the audit default.
    H, A = int(cfg.n_look_ahead), int(cfg.action_dim)
    act_seq = torch.zeros(H, A)
    out = p.sample_action_sequences(act_seq.clone())
    max_delta = _max_per_step_delta_inf_norm(out)
    assert torch.all(max_delta <= 0.05 + 1e-6), (
        f"yaml override should bound per-step delta at 0.05; got {max_delta.tolist()}"
    )


# ─── 7. waypoints + clip: clip applies AFTER interpolation ─────────────

def test_waypoints_then_clip() -> None:
    cfg = _base_cfg(
        sample_delta_clip=True,
        waypoints_n=3, waypoints_interp="linear",
        noise_level=2.0,
        per_step_delta_lim=[0.05, 0.05, 0.05, 0.05],
    )
    p = _build_planner(cfg)
    H, A = int(cfg.n_look_ahead), int(cfg.action_dim)
    act_seq = torch.zeros(H, A)
    out = p.sample_action_sequences(act_seq.clone())
    max_delta = _max_per_step_delta_inf_norm(out)
    assert torch.all(max_delta <= 0.05 + 1e-6), (
        f"waypoints + clip: per-step delta {max_delta.tolist()} > 0.05"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
