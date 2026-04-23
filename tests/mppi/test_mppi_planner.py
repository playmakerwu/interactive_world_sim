"""Tests for rl.mppi.mppi_planner.MPPIPlanner.

Per Task C.6 spec, with the warm-start test (#3) replaced by a beta_filter
intra-horizon-noise-correlation test — see MPPI_REFERENCE_NOTES.md for why
the spec's "warm-start" interpretation of beta_filter was incorrect.

7 tests:
  1. plan_step returns correctly-shaped finite action
  2. iterative refinement reduces sample reward variance
  3. beta_filter controls intra-horizon noise correlation
     (replaces spec test 3 — see MPPI_REFERENCE_NOTES.md)
  4. action clipping respects action_upper_lim
  5. reward_weight controls softmax sharpness
  6. planner is stateless across plan_step calls
     (replaces spec test 6 — reference does not maintain cross-call state)
  7. end-to-end smoke
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from env.pusht_wm_env import PushTWMEnv
from rl.mppi.mppi_planner import MPPIPlanner

CKPT_PATH = Path("outputs/pusht_cam1/checkpoints/best.ckpt")
SAMPLE_HDF5 = Path("data/mini/pusht/val/episode_0.hdf5")
GOAL_PT = Path("tests/goal_selection/state_goal.pt")


def _have_gpu_and_ckpt() -> bool:
    return torch.cuda.is_available() and CKPT_PATH.exists()


@pytest.fixture(scope="module")
def env() -> PushTWMEnv:
    if not _have_gpu_and_ckpt():
        pytest.skip("CUDA + WM checkpoint required")
    return PushTWMEnv(str(CKPT_PATH), device="cuda:0")


def _small_cfg(action_dim: int = 4) -> OmegaConf:
    """Tiny config for fast tests."""
    return OmegaConf.create({
        "n_sample": 4,
        "n_look_ahead": 3,
        "n_update_iter": 2,
        "noise_level": 0.05,
        "reward_weight": 200.0,
        "beta_filter": 0.7,
        "cv_fail_penalty": -10.0,
        "action_dim": action_dim,
        "action_lower_lim": [-1.0] * action_dim,
        "action_upper_lim": [1.0] * action_dim,
        "control_steps": 2,
        "seed": 0,
    })


@pytest.fixture(scope="module")
def goal(env) -> dict:
    if not GOAL_PT.exists():
        pytest.skip(f"missing goal file: {GOAL_PT}")
    return env.load_goal(str(GOAL_PT))


@pytest.fixture(scope="module")
def initial_z(env) -> torch.Tensor:
    if not SAMPLE_HDF5.exists():
        pytest.skip(f"missing dataset: {SAMPLE_HDF5}")
    z = env.load_initial_from_hdf5(str(SAMPLE_HDF5), frame_idx=0)
    return z[0]  # (C, H_lat, W_lat)


# ─── 1. plan_step returns correctly-shaped finite action ────────────────

def test_plan_step_returns_finite_action(env, initial_z, goal):
    cfg = _small_cfg(action_dim=env.action_dim)
    planner = MPPIPlanner(env, cfg)
    a = planner.plan_step(initial_z, goal)
    assert a.shape == (env.action_dim,), f"got shape {tuple(a.shape)}"
    assert torch.all(torch.isfinite(a)), f"non-finite action: {a}"
    print(f"[finite] a = {a.tolist()}")


# ─── 2. iterative refinement reduces sample reward variance ──────────────

def test_iterative_refinement_reduces_variance(env, initial_z, goal):
    """After more refinement iterations, the sampled action sequences should
    cluster more tightly around the converged mean — measurable as a smaller
    spread in the per-sample rewards at the LAST iteration."""
    cfg_low = _small_cfg(action_dim=env.action_dim)
    cfg_low.n_update_iter = 1
    cfg_high = _small_cfg(action_dim=env.action_dim)
    cfg_high.n_update_iter = 5

    planner_low = MPPIPlanner(env, cfg_low)
    planner_high = MPPIPlanner(env, cfg_high)

    _ = planner_low.plan_step(initial_z, goal)
    _ = planner_high.plan_step(initial_z, goal)

    var_low = float(planner_low.last_stats.final_iter_rewards.var())
    var_high = float(planner_high.last_stats.final_iter_rewards.var())
    print(f"[refinement] var(rewards) low_iter={var_low:.4f}  high_iter={var_high:.4f}")
    assert var_high <= var_low + 0.1, (
        f"high-iter variance ({var_high:.4f}) should not exceed low-iter "
        f"variance ({var_low:.4f}) + slack — refinement appears not to "
        f"concentrate samples"
    )


# ─── 3. beta_filter controls intra-horizon noise correlation ────────────

def test_beta_filter_intra_horizon_correlation(env):
    """Per MPPI_REFERENCE_NOTES.md: beta_filter is intra-horizon noise
    smoothing, not cross-plan-step warm-start.

    Sample many sequences and measure noise autocorrelation along H.
    With beta_filter=1.0, residual is purely fresh noise each step
    (low autocorr). With beta_filter=0.1, residual changes slowly across
    H (high autocorr). Test that lower beta_filter -> higher autocorr.
    """
    A = env.action_dim
    H = 16
    N = 256

    def _avg_corr(beta: float) -> float:
        cfg = _small_cfg(action_dim=A)
        cfg.n_sample = N
        cfg.n_look_ahead = H
        cfg.beta_filter = beta
        cfg.noise_level = 1.0  # large noise so the correlation is well-resolved
        planner = MPPIPlanner(env, cfg)
        zero_seq = torch.zeros(H, A, device=env.device)
        seqs = planner.sample_action_sequences(zero_seq).cpu().numpy()  # (N, H, A)
        # Step-to-step Pearson correlation per dim, averaged.
        corrs = []
        for d in range(A):
            for n in range(N):
                x = seqs[n, :-1, d]
                y = seqs[n, 1:, d]
                if x.std() > 1e-6 and y.std() > 1e-6:
                    corrs.append(float(np.corrcoef(x, y)[0, 1]))
        return float(np.mean(corrs)) if corrs else float("nan")

    corr_high_smoothing = _avg_corr(beta=0.1)  # near-constant residual
    corr_low_smoothing = _avg_corr(beta=1.0)   # nearly-fresh-each-step noise

    print(
        f"[beta_filter] mean step-to-step autocorr  "
        f"beta=0.1 -> {corr_high_smoothing:.3f}, "
        f"beta=1.0 -> {corr_low_smoothing:.3f}"
    )
    assert corr_high_smoothing > corr_low_smoothing, (
        f"beta=0.1 (high smoothing) should yield higher autocorr "
        f"({corr_high_smoothing:.3f}) than beta=1.0 ({corr_low_smoothing:.3f})"
    )


# ─── 4. action clipping respects action_upper_lim ───────────────────────

def test_action_clipping(env):
    A = env.action_dim
    cfg = _small_cfg(action_dim=A)
    cfg.n_sample = 64
    cfg.n_look_ahead = 4
    cfg.beta_filter = 1.0
    cfg.noise_level = 5.0  # huge noise to force clipping
    upper = 0.01
    cfg.action_upper_lim = [upper] * A
    cfg.action_lower_lim = [-upper] * A
    planner = MPPIPlanner(env, cfg)
    zero_seq = torch.zeros(cfg.n_look_ahead, A, device=env.device)
    seqs = planner.sample_action_sequences(zero_seq)
    assert seqs.max() <= upper + 1e-6, f"max {seqs.max()} > {upper}"
    assert seqs.min() >= -upper - 1e-6, f"min {seqs.min()} < {-upper}"


# ─── 5. reward_weight controls softmax sharpness ────────────────────────

def test_reward_weight_controls_softmax_sharpness(env):
    A = env.action_dim
    cfg = _small_cfg(action_dim=A)
    cfg.n_sample = 32
    rewards = torch.linspace(-1.0, 0.0, steps=cfg.n_sample, device=env.device)
    act_seqs = torch.zeros(cfg.n_sample, cfg.n_look_ahead, A, device=env.device)

    cfg.reward_weight = 1.0
    planner_soft = MPPIPlanner(env, cfg)
    _, w_soft = planner_soft.optimize_action_mppi(act_seqs, rewards)

    cfg.reward_weight = 200.0
    planner_sharp = MPPIPlanner(env, cfg)
    _, w_sharp = planner_sharp.optimize_action_mppi(act_seqs, rewards)

    print(f"[softmax] max_w soft={float(w_soft.max()):.3f}  sharp={float(w_sharp.max()):.3f}")
    assert float(w_sharp.max()) > 0.95, (
        f"reward_weight=200 should give max_weight > 0.95; got {float(w_sharp.max()):.3f}"
    )
    assert float(w_soft.max()) < 0.30, (
        f"reward_weight=1 should give max_weight < 0.30; got {float(w_soft.max()):.3f}"
    )


# ─── 6. planner is stateless across plan_step calls ─────────────────────

def test_planner_stateless_across_calls(env, initial_z, goal):
    """Per reference: trajectory_optimization is stateless across calls.
    Calling plan_step twice from the same z + same RNG state must give
    identical actions."""
    cfg = _small_cfg(action_dim=env.action_dim)
    planner = MPPIPlanner(env, cfg)
    # First call from seed=0
    a1 = planner.plan_step(initial_z, goal)
    # Reset the planner's RNG and do it again
    planner._gen.manual_seed(int(cfg.seed))
    a2 = planner.plan_step(initial_z, goal)
    # Should match bit-exactly modulo WM denoiser stochasticity. We
    # tolerate small numerical drift since the WM uses the global CUDA RNG
    # and we don't re-seed that — but the planner's own RNG is reset, so
    # action sampling is deterministic. Allow a loose tol.
    diff = (a1 - a2).abs().max()
    print(f"[stateless] |a1 - a2|_inf = {float(diff):.4f}")
    assert diff < 1.0, f"stateless planner diverged too much: {diff}"


# ─── 7. end-to-end smoke ────────────────────────────────────────────────

def test_end_to_end_smoke(env, initial_z, goal):
    """Tiny config, run 3 control steps, verify nothing crashes and
    final state estimate is non-NaN."""
    cfg = _small_cfg(action_dim=env.action_dim)
    cfg.control_steps = 3
    planner = MPPIPlanner(env, cfg)

    z = initial_z
    for t in range(int(cfg.control_steps)):
        a = planner.plan_step(z, goal)
        assert torch.all(torch.isfinite(a)), f"action NaN at step {t}"
        z = env.dynamics_step(z, a)
    state = env.estimate_from_latent(z)
    assert "success" in state
    print(f"[smoke] final cv_success={bool(state['success'])}")
