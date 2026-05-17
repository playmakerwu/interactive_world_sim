"""Smoke test: yiru's production-faithful MPPI planner imports + constructs
+ accepts a 1-niter step over a mock env.

This test does NOT load the world model or any checkpoint — it stubs out
the env's snapshot/restore/step_batch interface with deterministic
synthetic outputs so the test runs in <1 s on CPU and validates the
algorithmic plumbing (sample → evaluate → optimize → reward → softmax).

Run with:
    pytest tests/mppi/test_yiru_smoke.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def mock_env():
    """A minimal mock WorldModelEnv exposing exactly the methods the
    new Planner needs:
      - device
      - snapshot() / restore()
      - step_batch(actions) → BatchedObservation(latents, rgbs)
      - render() — used by the closed-loop driver but not by Planner
    """
    class _Snapshot:
        def __init__(self):
            self.action_window = torch.zeros(10, 4, dtype=torch.float32)

    class _BatchedObs:
        def __init__(self, latents, rgbs):
            self.latents = latents
            self.rgbs = rgbs

    class _MockEnv:
        device = torch.device("cpu")

        def snapshot(self):
            return _Snapshot()

        def restore(self, state):
            del state

        def step_batch(self, actions, decode_batch_size=16):
            del decode_batch_size
            actions_t = torch.as_tensor(actions, dtype=torch.float32)
            K, H, _ = actions_t.shape
            latents = torch.zeros(K, H, 8, 16, 16, dtype=torch.float32)
            # Decode to a 128x128 pink image. The CV detector will fail
            # (no T-block) — reward path falls through to cv_fail_penalty.
            rgbs = np.zeros((K, H, 3, 128, 128), dtype=np.uint8)
            rgbs[..., 0, :, :] = 200  # mostly red — fails the "real" preset
            return _BatchedObs(latents, rgbs)

    return _MockEnv()


def test_import_module():
    """The new planner module imports cleanly."""
    from interactive_world_sim_mppi import (  # noqa: F401
        Config,
        GoalPose,
        MPPIPlanner,
        Planner,
        detect_goal_state_from_episode,
    )


def test_config_defaults_match_production():
    """Config defaults track production yaml field-for-field."""
    from interactive_world_sim_mppi import Config

    cfg = Config()
    assert cfg.n_sample == 100
    assert cfg.n_look_ahead == 10
    assert cfg.n_update_iter == 5  # NOT 50 (yiru's old default)
    assert cfg.noise_level == 0.05
    assert cfg.reward_weight == 200.0
    assert cfg.beta_filter == 0.7
    assert cfg.cv_fail_penalty == -10.0  # NOT -1000 (yiru's old default)
    assert cfg.sample_delta_clip is True
    assert tuple(cfg.per_step_delta_lim) == (0.0975, 0.0919, 0.0760, 0.0909)
    assert cfg.cv_processing_resolution == 128  # NOT 512 (yiru's old default)
    assert cfg.seed == 0
    assert cfg.step_each_iter == 1
    assert cfg.control_steps == 50
    assert cfg.cv_n_workers == 16
    assert cfg.delta_mode is False
    assert cfg.waypoints_n is None  # default vanilla; no waypoints


def test_planner_construct(mock_env):
    """Planner constructs against a mock env."""
    from interactive_world_sim_mppi import Config, Planner

    cfg = Config(device="cpu", n_sample=4, n_update_iter=1)
    planner = Planner(mock_env, cfg)
    assert planner.image_diagonal > 180.0  # 128*sqrt(2) ≈ 181.02
    assert planner.image_diagonal < 182.0
    assert planner._gen is not None
    # Action limits live on the planner.
    assert tuple(planner.action_lower_lim.tolist()) == (-1.0, -1.0, -1.0, -1.0)
    assert tuple(planner.action_upper_lim.tolist()) == (1.0, 1.0, 1.0, 1.0)


def test_sample_action_sequences_shape_and_clamp(mock_env):
    """Vanilla sampler produces (N, H, A) with values clamped to the cube."""
    from interactive_world_sim_mppi import Config, Planner

    cfg = Config(
        device="cpu", n_sample=8, n_update_iter=1,
        sample_delta_clip=False,  # disable to test the raw cube clamp
    )
    planner = Planner(mock_env, cfg)
    act_seq = torch.zeros(cfg.n_look_ahead, cfg.action_dim, dtype=torch.float32)
    samples = planner.sample_action_sequences(act_seq)
    assert samples.shape == (cfg.n_sample, cfg.n_look_ahead, cfg.action_dim)
    assert (samples >= planner.action_lower_lim).all()
    assert (samples <= planner.action_upper_lim).all()


def test_softmax_max_subtract_no_underflow(mock_env):
    """optimize_action_mppi uses max-subtract; large negative rewards
    don't underflow softmax."""
    from interactive_world_sim_mppi import Config, Planner

    cfg = Config(device="cpu", n_sample=4, reward_weight=200.0)
    planner = Planner(mock_env, cfg)
    # Synthetic: H=10, A=4, N=4, rewards = [-1000, -2000, -3000, -4000]
    act_seqs = torch.randn(4, 10, 4, dtype=torch.float32)
    rewards = torch.tensor([-1000.0, -2000.0, -3000.0, -4000.0])
    new_mean, weights = planner.optimize_action_mppi(act_seqs, rewards)
    # Max-subtract keeps the softmax non-degenerate (sum exactly 1).
    assert torch.isfinite(weights).all()
    assert abs(weights.sum().item() - 1.0) < 1e-5
    # Best sample (rewards[0]) gets the most weight.
    assert weights[0] == weights.max()


def test_plan_step_runs_one_iter(mock_env):
    """A full plan_step runs end-to-end on the mock env with 1 niter."""
    from interactive_world_sim_mppi import Config, Planner

    cfg = Config(
        device="cpu",
        n_sample=4,
        n_update_iter=1,
        sample_delta_clip=True,
        cv_n_workers=0,  # sequential — avoids spawning a pool in the test
    )
    planner = Planner(mock_env, cfg)
    goal_state = {
        "cx": 64.0, "cy": 64.0,
        "sin_theta": 0.0, "cos_theta": 1.0, "theta_deg": 0.0,
    }
    # sample_delta_clip=True requires an anchor; supply zeros (current
    # action_window default).
    anchor = torch.zeros(cfg.action_dim, dtype=torch.float32)
    action = planner.plan_step(
        z_current_unused=None,
        goal_state=goal_state,
        init_act_seq=None,
        return_iteration_log=False,
        anchor=anchor,
    )
    assert action.shape == (cfg.action_dim,)
    assert torch.isfinite(action).all()
    # Full plan landed on last_stats.
    assert planner.last_stats is not None
    assert planner.last_stats.act_seq.shape == (cfg.n_look_ahead, cfg.action_dim)
    # CV failed on the synthetic frames → reward should be cv_fail_penalty.
    assert planner.last_stats.n_cv_failures_final_iter == cfg.n_sample


def test_audit_log_emits_entries(mock_env):
    """audit_log_enabled=True populates the audit log with the right schema."""
    from interactive_world_sim_mppi import Config, Planner

    cfg = Config(
        device="cpu",
        n_sample=4, n_update_iter=2,
        sample_delta_clip=False,
        cv_n_workers=0,
        audit_log_enabled=True,
    )
    planner = Planner(mock_env, cfg)
    goal_state = {
        "cx": 64.0, "cy": 64.0,
        "sin_theta": 0.0, "cos_theta": 1.0, "theta_deg": 0.0,
    }
    planner.plan_step(
        z_current_unused=None, goal_state=goal_state,
        return_iteration_log=False, anchor=None,
    )
    audit = planner.get_audit_log()
    # 1 pre-loop snapshot (niter=-1) + n_update_iter entries
    assert len(audit) == 1 + cfg.n_update_iter
    assert audit[0]["niter"] == -1
    assert audit[1]["niter"] == 0
    assert audit[-1]["niter"] == cfg.n_update_iter - 1
    for entry in audit:
        assert "plan_call_idx" in entry
        assert "mean" in entry
