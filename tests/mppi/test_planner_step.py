"""Step 4 tests for MPPIPlanner.plan_step.

Four tests:
  1. plan_step on a random latent returns a sane-shaped, finite action.
  2. plan_step from z_goal — the best-scoring trajectory should have
     reward close to zero (we are already at the goal).
  3. Repeatability — same seed produces the same action.
  4. Stats capture — self.last_stats is fully populated and shapes are
     correct (the viz + Step 5 recorder depend on this).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rl.labeling.cv_labeler import CVLabeler
from rl.mppi.planner import MPPIPlanner

CKPT_PATH = Path("outputs/pusht_cam1/checkpoints/best.ckpt")
Z_GOAL_PATH = Path("tests/goal_selection/z_goal.pt")
STATE_GOAL_PATH = Path("tests/goal_selection/state_goal.pt")
ACTION_DIM = 4
RES = 128

# Local fits N=16; cloud uses N=128 (MPPI_NOTES §Step 3).
N_LOCAL = 16
H_LOCAL = 10


def _have_gpu_and_ckpt() -> bool:
    return torch.cuda.is_available() and CKPT_PATH.exists()


@pytest.fixture(scope="module")
def state_goal():
    g = torch.load(STATE_GOAL_PATH, map_location="cpu")
    return {
        "cx": g["cx"], "cy": g["cy"],
        "sin_theta": g["sin_theta"], "cos_theta": g["cos_theta"],
    }


@pytest.fixture(scope="module")
def wm():
    if not _have_gpu_and_ckpt():
        pytest.skip("CUDA + WM checkpoint required")
    from rl.models.world_model import DifferentiableDynamics
    return DifferentiableDynamics(str(CKPT_PATH), device="cuda:0")


@pytest.fixture(scope="module")
def planner(wm, state_goal):
    labeler = CVLabeler(preset="REAL", resolution=RES)
    return MPPIPlanner(
        wm, state_goal,
        N=N_LOCAL, H=H_LOCAL, sigma=0.1, temperature=1.0,
        action_dim=ACTION_DIM, resolution=RES,
        labeler=labeler, device="cuda:0",
    )


@pytest.fixture(scope="module")
def z_goal():
    return torch.load(Z_GOAL_PATH, map_location="cuda:0")


def test_plan_step_returns_finite_action(planner, z_goal):
    """Action should be shape (A,) and free of NaN/Inf."""
    a_star = planner.plan_step(z_goal, seed=0)
    assert a_star.shape == (ACTION_DIM,)
    assert torch.all(torch.isfinite(a_star))
    print(f"[finite] a_star = {a_star.tolist()}")


def test_plan_step_from_goal_has_best_reward_near_zero(planner, z_goal):
    """If we are already at the goal, the best trajectory should score
    close to zero — the reward function and pipeline work."""
    planner.plan_step(z_goal, seed=0)
    stats = planner.last_stats
    assert stats is not None
    # At least one rollout should score very close to zero. Tolerance
    # loose because the decoder is stochastic.
    best = stats.rewards.max()
    assert best > -0.05, f"best reward from z_goal was {best:.4f}"
    mean = stats.rewards.mean()
    print(
        f"[at-goal] best={best:.4f}  mean={mean:.4f}  "
        f"cv_fail={stats.cv_fail_count}/{planner.N}"
    )


def test_plan_step_is_reproducible(planner, z_goal):
    """Same seed must produce the same a_star (bit-exact). This is
    load-bearing for debugging Step 5 trajectories."""
    a1 = planner.plan_step(z_goal, seed=42)
    a2 = planner.plan_step(z_goal, seed=42)
    torch.testing.assert_close(a1, a2, rtol=0, atol=0)


def test_last_stats_is_fully_populated(planner, z_goal):
    """Viz and Step 5 recorder depend on these fields — verify shapes."""
    planner.plan_step(z_goal, seed=0)
    s = planner.last_stats
    assert s is not None
    assert s.a_star.shape == (ACTION_DIM,)
    assert s.a_naive_mean.shape == (ACTION_DIM,)
    assert s.rewards.shape == (planner.N,)
    assert s.weights.shape == (planner.N,)
    assert len(s.labels) == planner.N
    assert s.actions is not None and s.actions.shape == (planner.N, planner.H, ACTION_DIM)
    assert s.final_latents is not None and s.final_latents.shape[0] == planner.N
    # weights sum to 1
    np.testing.assert_allclose(s.weights.sum(), 1.0, atol=1e-5)
    # cv_fail_count matches labels
    assert s.cv_fail_count == sum(1 for lbl in s.labels if not lbl.success)
