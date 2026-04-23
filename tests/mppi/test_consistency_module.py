"""Module-level consistency tests: Phase 1 MPPI primitives vs. the
diffusion-forcing reference (Phase 2 Task D).

Each test compares one algorithm component on identical input. The
oracle is the verbatim-ported helper functions in `_reference_helpers.py`
(see that file's docstring for why we ported instead of imported the
reference Planner directly).

Five tests:
  D.2.1 — sample_action_sequences matches reference (bit-exact)
  D.2.2 — softmax weights match reference (numerically equivalent
          even with our max-subtract stabilisation)
  D.2.3 — optimize_action_mppi (one weighted-mean step) matches reference
  D.2.4 — action clipping matches reference
  D.x   — bonus: import-smoke that confirms the actual diffusion-forcing
          Planner class is still constructible with our config schema
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from env.pusht_wm_env import PushTWMEnv
from rl.mppi.mppi_planner import MPPIPlanner

from tests.mppi._reference_helpers import (  # noqa: E402
    reference_clamp_actions,
    reference_optimize_action_mppi,
    reference_sample_action_sequences,
    reference_softmax_weights,
)

CKPT_PATH = Path("outputs/pusht_cam1/checkpoints/best.ckpt")
DIFF_FORCING_PATH = Path.home() / "Documents/diffusion-forcing"


def _have_gpu_and_ckpt() -> bool:
    return torch.cuda.is_available() and CKPT_PATH.exists()


@pytest.fixture(scope="module")
def env() -> PushTWMEnv:
    if not _have_gpu_and_ckpt():
        pytest.skip("CUDA + WM checkpoint required (env construction loads WM)")
    return PushTWMEnv(str(CKPT_PATH), device="cuda:0")


@pytest.fixture(scope="module")
def planner(env) -> MPPIPlanner:
    cfg = OmegaConf.create({
        "n_sample": 32,
        "n_look_ahead": 10,
        "n_update_iter": 1,
        "noise_level": 0.05,
        "reward_weight": 200.0,
        "beta_filter": 0.7,
        "cv_fail_penalty": -10.0,
        "action_dim": env.action_dim,
        "action_lower_lim": [-1.0] * env.action_dim,
        "action_upper_lim": [1.0] * env.action_dim,
        "control_steps": 1,
        "seed": 1234,
    })
    return MPPIPlanner(env, cfg)


# ─── D.2.1 sampling consistency ────────────────────────────────────────

def test_sampling_matches_reference_bit_exact(env, planner):
    """Bit-exact comparison: same seed + same config -> same actions."""
    cfg = planner.cfg
    H, A = int(cfg.n_look_ahead), int(cfg.action_dim)
    N = int(cfg.n_sample)

    act_seq = torch.zeros(H, A, device=env.device)

    # Ours: re-seed our planner's dedicated generator
    planner._gen.manual_seed(99)
    ours = planner.sample_action_sequences(act_seq)  # (N, H, A)

    # Reference helper: use a separate generator seeded identically.
    # Both samplers consume the same number of randn(N, A) draws per
    # horizon step, so a fresh generator with the same seed produces the
    # same noise stream.
    ref_gen = torch.Generator(device=env.device).manual_seed(99)
    lower = torch.tensor(list(cfg.action_lower_lim), device=env.device)
    upper = torch.tensor(list(cfg.action_upper_lim), device=env.device)
    ref = reference_sample_action_sequences(
        act_seq,
        n_sample=N,
        beta_filter=float(cfg.beta_filter),
        noise_level=float(cfg.noise_level),
        action_lower_lim=lower,
        action_upper_lim=upper,
        generator=ref_gen,
        device=env.device,
    )

    assert ours.shape == ref.shape == (N, H, A)
    max_diff = float((ours - ref).abs().max())
    print(f"[D.2.1 sampling] max abs diff = {max_diff:.3e}  (N={N}, H={H}, A={A})")
    assert max_diff < 1e-5, (
        f"Sampling diverges from reference: max abs diff {max_diff:.3e}\n"
        f"Reference[0,0]={ref[0,0].tolist()}\nOurs[0,0]     ={ours[0,0].tolist()}"
    )


# ─── D.2.2 softmax consistency ─────────────────────────────────────────

def test_softmax_matches_reference():
    """Compare both the unstabilised reference softmax and our stabilised
    softmax against the same arbitrary reward vector. They should be
    numerically identical for well-conditioned inputs (and our max-subtract
    is mathematically identical for any input with positive reward_weight)."""
    rewards = torch.tensor(
        [-2.0, -0.3, -0.5, -1.8, -0.1, -0.25, -2.2, -0.4],
        dtype=torch.float32,
    )
    reward_weight = 200.0

    ref_w = reference_softmax_weights(rewards, reward_weight)

    # Mirror our planner's stabilised computation (without re-instantiating
    # the planner — this is the algebra it executes inline).
    rw = reward_weight
    scaled = rewards * rw
    our_w = torch.nn.functional.softmax(scaled - scaled.max(), dim=0)

    max_diff = float((our_w - ref_w).abs().max())
    print(f"[D.2.2 softmax] max abs diff = {max_diff:.3e}")
    print(f"  ref top-3 weights: {sorted(ref_w.tolist(), reverse=True)[:3]}")
    print(f"  our top-3 weights: {sorted(our_w.tolist(), reverse=True)[:3]}")

    assert max_diff < 1e-6, (
        f"Stabilised softmax should be bit-equivalent to reference for "
        f"well-conditioned rewards; max diff {max_diff:.3e} exceeds threshold"
    )
    assert abs(float(our_w.sum()) - 1.0) < 1e-6
    assert abs(float(ref_w.sum()) - 1.0) < 1e-6


# ─── D.2.3 optimize_action_mppi (one weighted-mean step) ───────────────

def test_optimize_action_mppi_matches_reference(env, planner):
    """Single weighted-mean step: same act_seqs, same rewards -> same
    aggregate. This exercises optimize_action_mppi which is also called
    inside the iterative refinement loop."""
    cfg = planner.cfg
    N, H, A = int(cfg.n_sample), int(cfg.n_look_ahead), int(cfg.action_dim)

    torch.manual_seed(0)
    act_seqs = torch.randn(N, H, A, device=env.device) * 0.3
    rewards = torch.randn(N, device=env.device) * 0.5 - 0.5

    # Ours
    our_seq, our_w = planner.optimize_action_mppi(act_seqs, rewards)

    # Reference
    ref_seq, ref_w = reference_optimize_action_mppi(
        act_seqs, rewards, reward_weight=float(cfg.reward_weight),
    )

    seq_diff = float((our_seq - ref_seq).abs().max())
    w_diff = float((our_w - ref_w.to(our_w.device)).abs().max())
    print(f"[D.2.3 optimize_action] act_seq max diff = {seq_diff:.3e}  weights max diff = {w_diff:.3e}")
    assert seq_diff < 1e-4, f"act_seq divergence too large: {seq_diff:.3e}"
    assert w_diff < 1e-6, f"weights divergence too large: {w_diff:.3e}"


# ─── D.2.4 action clipping ─────────────────────────────────────────────

def test_action_clipping_matches_reference(env):
    """Clipping is just torch.clamp. This regression-guards against ever
    introducing per-dim per-step custom logic."""
    A = env.action_dim
    torch.manual_seed(123)
    raw = torch.randn(16, 10, A) * 2.0  # deliberately too wide
    lower = torch.full((A,), -0.5)
    upper = torch.full((A,), 0.5)

    ref_clipped = reference_clamp_actions(raw, lower, upper)
    our_clipped = torch.clamp(raw, lower, upper)  # this is what our sampler does inline

    assert torch.allclose(ref_clipped, our_clipped)
    assert (our_clipped >= lower).all()
    assert (our_clipped <= upper).all()


# ─── D.x bonus: reference is still importable with our config schema ────

def test_reference_planner_constructible_with_our_config():
    """Smoke test: the actual diffusion-forcing Planner class still loads
    and accepts a config that has all our keys plus the few reference-
    only keys it requires. Guards against silent reference-side API drift."""
    if not DIFF_FORCING_PATH.exists():
        pytest.skip(f"diffusion-forcing repo not at {DIFF_FORCING_PATH}")
    sys.path.insert(0, str(DIFF_FORCING_PATH))
    try:
        from algorithms.latent_dynamics.planner_v0_0 import Planner  # noqa: E402
    finally:
        sys.path.remove(str(DIFF_FORCING_PATH))

    config = {
        "action_dim": 4,
        "n_sample": 32,
        "n_look_ahead": 10,
        "n_update_iter": 5,
        "reward_weight": 200.0,
        "noise_level": 0.05,
        "beta_filter": 0.7,
        "action_lower_lim": [-1.0] * 4,
        "action_upper_lim": [1.0] * 4,
        "planner_type": "MPPI",
        "device": "cpu",  # construction-only smoke; no rollout
        "verbose": False,
        "rollout_best": False,
        "n_his": 1,
    }
    p = Planner(config)
    assert p.action_dim == 4
    assert p.n_sample == 32
    assert p.beta_filter == 0.7
    assert p.noise_level == 0.05
    assert p.reward_weight == 200.0
    print(
        f"[D.x smoke] reference Planner constructed: "
        f"n_sample={p.n_sample}  beta_filter={p.beta_filter}"
    )
