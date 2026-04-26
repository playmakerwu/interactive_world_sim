"""Tests for MPPI per-iteration reward logging and visualization artifacts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from rl.mppi.mppi_planner import MPPIPlanner

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run_mppi_v2.py"
DEVICE = "cpu"


class MockEnv:
    """Small deterministic env implementing the MPPIPlanner interface."""

    def __init__(self, action_dim: int = 4, dt: float = 0.1):
        self.action_dim = action_dim
        self.device = DEVICE
        self.image_diagonal = 1.0
        self.dt = dt

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        was_unbatched = z0.dim() == 3
        if was_unbatched:
            z0 = z0.unsqueeze(0)
            actions = actions.unsqueeze(0)
        z = z0.squeeze(-1).squeeze(-1)
        zs = [z]
        for h in range(actions.shape[1]):
            z = z + actions[:, h] * self.dt
            zs.append(z)
        traj = torch.stack(zs, dim=1).unsqueeze(-1).unsqueeze(-1)
        return traj if not was_unbatched else traj[0]

    def dynamics_step(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        was_batched = z.dim() == 4
        if not was_batched:
            z = z.unsqueeze(0)
            action = action.unsqueeze(0)
        z_next = z.squeeze(-1).squeeze(-1) + action * self.dt
        z_next = z_next.unsqueeze(-1).unsqueeze(-1)
        return z_next if was_batched else z_next[0]

    def estimate_from_latent(self, z: torch.Tensor) -> dict:
        z_flat = z.squeeze(-1).squeeze(-1)
        if z_flat.dim() == 1:
            success = torch.tensor(True)
        else:
            success = torch.ones(z_flat.shape[0], dtype=torch.bool)
        return {"_state": z_flat, "success": success}

    def compute_reward(
        self,
        state: dict,
        goal: dict,
        image_diagonal: float | None = None,
        cv_fail_penalty: float = -10.0,
    ) -> torch.Tensor:
        del image_diagonal, cv_fail_penalty
        return -(state["_state"] - goal["_state"]).norm(dim=-1)


class FixedRewardPlanner(MPPIPlanner):
    """Planner whose single iteration sees a known reward vector."""

    def __init__(self, env: MockEnv, cfg, rewards: torch.Tensor):
        super().__init__(env, cfg)
        self._fixed_rewards = rewards.float()

    def sample_action_sequences(self, act_seq: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            int(self.cfg.n_sample),
            int(self.cfg.n_look_ahead),
            int(self.cfg.action_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def evaluate_trajectories(
        self,
        z_current: torch.Tensor,
        act_seqs: torch.Tensor,
        goal_state: dict,
    ) -> tuple[torch.Tensor, dict]:
        del z_current, act_seqs, goal_state
        state = {"success": torch.ones(len(self._fixed_rewards), dtype=torch.bool)}
        return self._fixed_rewards.clone(), state


def _cfg(**overrides):
    cfg = OmegaConf.create({
        "n_sample": 4,
        "n_look_ahead": 3,
        "n_update_iter": 2,
        "noise_level": 0.05,
        "reward_weight": 10.0,
        "beta_filter": 0.7,
        "cv_fail_penalty": -10.0,
        "action_dim": 4,
        "action_lower_lim": [-1.0] * 4,
        "action_upper_lim": [1.0] * 4,
        "control_steps": 2,
        "seed": 0,
    })
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _z0(action_dim: int = 4) -> torch.Tensor:
    return torch.zeros(action_dim, 1, 1)


def _goal(action_dim: int = 4) -> dict:
    goal_vec = torch.ones(action_dim)
    return {"_state": goal_vec, "cx": 1.0, "cy": 1.0}


def _load_runner_helper(name: str):
    spec = importlib.util.spec_from_file_location("run_mppi_v2_for_iteration_tests", str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_mppi_v2_for_iteration_tests"] = module
    spec.loader.exec_module(module)
    return getattr(module, name)


def test_iteration_log_structure():
    env = MockEnv()
    cfg = _cfg(n_update_iter=3, n_sample=5)
    planner = MPPIPlanner(env, cfg)

    action, log = planner.plan_step(_z0(), _goal(), return_iteration_log=True)

    assert action.shape == (env.action_dim,)
    assert len(log) == int(cfg.n_update_iter)
    scalar_or_tensor_keys = {
        "iter",
        "rewards_all",
        "weights_all",
        "reward_max",
        "reward_min",
        "reward_mean",
        "reward_std",
        "reward_softmax_weighted",
    }
    optional_per_sample_keys = {
        "sample_cx",
        "sample_cy",
        "sample_sin_theta",
        "sample_cos_theta",
        "sample_success",
    }
    optional_top_k_keys = {
        "top_k_intermediate_cx",
        "top_k_intermediate_cy",
        "top_k_intermediate_success",
        "top_k_indices",
        "top_k_rewards",
    }
    required = scalar_or_tensor_keys | optional_per_sample_keys | optional_top_k_keys
    for iter_idx, rec in enumerate(log):
        assert set(rec) == required
        assert rec["iter"] == iter_idx
        assert rec["rewards_all"].shape == (int(cfg.n_sample),)
        assert rec["weights_all"].shape == (int(cfg.n_sample),)
        assert rec["rewards_all"].device.type == "cpu"
        assert rec["weights_all"].device.type == "cpu"
        assert torch.all(torch.isfinite(rec["rewards_all"]))
        assert torch.all(torch.isfinite(rec["weights_all"]))
        assert float(rec["weights_all"].sum()) == pytest.approx(1.0, abs=1e-5)
        for key in scalar_or_tensor_keys - {"iter", "rewards_all", "weights_all"}:
            assert torch.isfinite(torch.tensor(rec[key]))
        # MockEnv returns a {_state, success} dict only — no cx/cy/sin/cos —
        # so per-sample CV fields should be None for this test fixture.
        for key in optional_per_sample_keys:
            assert rec[key] is None, (
                f"MockEnv state lacks {key}; planner should record None"
            )
        # Top-K intermediate fields are only patched on the last iter when
        # the env can provide CV; MockEnv can't, so all iterations stay None.
        for key in optional_top_k_keys:
            assert rec[key] is None, (
                f"MockEnv can't provide CV; top_k field {key} should be None"
            )


class CVMockEnv(MockEnv):
    """MockEnv variant whose ``estimate_from_latent`` also returns cx/cy/sin/cos.

    Lets us test the planner's per-sample CV capture in isolation, without
    needing the WM or the real CVLabeler. Maps the first 4 latent dims to
    ``[cx, cy, sin_theta, cos_theta]``.
    """

    def estimate_from_latent(self, z: torch.Tensor) -> dict:
        z_flat = z.squeeze(-1).squeeze(-1)
        single = z_flat.dim() == 1
        if single:
            z_flat = z_flat.unsqueeze(0)
        N, D = z_flat.shape
        assert D >= 4, "CVMockEnv needs >=4 action dims to fake CV state"
        out = {
            "_state": z_flat if not single else z_flat[0],
            "cx": z_flat[:, 0].clone(),
            "cy": z_flat[:, 1].clone(),
            "sin_theta": z_flat[:, 2].clone(),
            "cos_theta": z_flat[:, 3].clone(),
            "success": torch.ones(N, dtype=torch.bool),
        }
        if single:
            out = {k: (v[0] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                   for k, v in out.items()}
            out["_state"] = z_flat[0]
            out["success"] = torch.tensor(True)
        return out


def test_iteration_log_captures_per_sample_cv_when_env_provides_it():
    env = CVMockEnv()
    cfg = _cfg(n_update_iter=2, n_sample=6)
    planner = MPPIPlanner(env, cfg)

    _action, log = planner.plan_step(_z0(), _goal(), return_iteration_log=True)

    for rec in log:
        for key in ("sample_cx", "sample_cy", "sample_sin_theta", "sample_cos_theta"):
            assert isinstance(rec[key], torch.Tensor)
            assert rec[key].shape == (int(cfg.n_sample),)
            assert rec[key].device.type == "cpu"
            assert torch.all(torch.isfinite(rec[key]))
        assert isinstance(rec["sample_success"], torch.Tensor)
        assert rec["sample_success"].shape == (int(cfg.n_sample),)
        assert rec["sample_success"].dtype == torch.bool


def test_iteration_log_last_iter_captures_top_k_intermediate_trajectories():
    """Last iteration of each plan_step gets top-K intermediate CV polylines."""
    env = CVMockEnv()
    cfg = _cfg(n_update_iter=3, n_sample=12, n_look_ahead=4)
    planner = MPPIPlanner(env, cfg)

    _action, log = planner.plan_step(_z0(), _goal(), return_iteration_log=True)

    K_expected = min(10, int(cfg.n_sample))
    Hp1 = int(cfg.n_look_ahead) + 1

    # Earlier iterations: top_k_* not patched -> None
    for rec in log[:-1]:
        for key in ("top_k_intermediate_cx", "top_k_intermediate_cy",
                    "top_k_intermediate_success", "top_k_indices", "top_k_rewards"):
            assert rec[key] is None, f"non-last iter should not carry {key}"

    # Last iteration: shapes and ordering correct
    last = log[-1]
    assert isinstance(last["top_k_intermediate_cx"], torch.Tensor)
    assert last["top_k_intermediate_cx"].shape == (K_expected, Hp1)
    assert last["top_k_intermediate_cy"].shape == (K_expected, Hp1)
    assert last["top_k_intermediate_success"].shape == (K_expected, Hp1)
    assert last["top_k_intermediate_success"].dtype == torch.bool
    assert last["top_k_indices"].shape == (K_expected,)
    assert last["top_k_indices"].dtype == torch.long
    assert last["top_k_rewards"].shape == (K_expected,)
    # Top-K rewards must be non-increasing (descending sort)
    rewards = last["top_k_rewards"]
    assert torch.all(rewards[:-1] >= rewards[1:]), (
        "top_k_rewards must be in descending order"
    )


def test_iteration_log_stats_math_known_rewards():
    rewards = torch.tensor([0.1, 0.5, 0.3, -0.2])
    cfg = _cfg(n_update_iter=1, n_sample=len(rewards), reward_weight=2.5)
    planner = FixedRewardPlanner(MockEnv(), cfg, rewards)

    _action, log = planner.plan_step(_z0(), _goal(), return_iteration_log=True)
    rec = log[0]

    scaled = rewards * float(cfg.reward_weight)
    expected_weights = F.softmax(scaled - scaled.max(), dim=0)
    expected_weighted = (expected_weights * rewards).sum()

    assert torch.allclose(rec["rewards_all"], rewards)
    assert torch.allclose(rec["weights_all"], expected_weights)
    assert rec["reward_softmax_weighted"] == pytest.approx(float(expected_weighted))
    assert rec["reward_max"] == pytest.approx(float(rewards.max()))
    assert rec["reward_min"] == pytest.approx(float(rewards.min()))
    assert rec["reward_mean"] == pytest.approx(float(rewards.mean()))
    assert rec["reward_std"] == pytest.approx(float(rewards.std()))


def test_plan_step_default_return_stays_backward_compatible():
    planner = MPPIPlanner(MockEnv(), _cfg())

    action = planner.plan_step(_z0(), _goal())

    assert isinstance(action, torch.Tensor)
    assert action.shape == (4,)
    assert planner.last_stats is not None
    assert len(planner.last_stats.iteration_log) == int(planner.cfg.n_update_iter)


def test_iteration_visualization_files_exist_and_open(tmp_path):
    env = MockEnv()
    cfg = _cfg(n_update_iter=2, n_sample=4, control_steps=2)
    planner = MPPIPlanner(env, cfg)
    goal = _goal()

    z = _z0()
    iteration_logs = []
    per_step = [{
        "t": 0,
        "reward": -1.0,
        "cv_success": True,
        "cx": 0.0,
        "cy": 0.0,
        "theta_deg": 0.0,
    }]
    for t in range(int(cfg.control_steps)):
        action, log = planner.plan_step(z, goal, return_iteration_log=True)
        iteration_logs.append(log)
        z = env.dynamics_step(z, action)
        state = env.estimate_from_latent(z)
        reward = env.compute_reward(state, goal)
        z_flat = z.squeeze(-1).squeeze(-1)
        per_step.append({
            "t": t + 1,
            "reward": float(reward),
            "cv_success": True,
            "cx": float(z_flat[0]),
            "cy": float(z_flat[1]),
            "theta_deg": 0.0,
        })

    write_viz = _load_runner_helper("_write_iteration_visualizations")
    paths = write_viz(iteration_logs, per_step, goal, cfg, tmp_path)

    plan_step_png = tmp_path / "iter_viz" / "plan_step_000.png"
    heatmap_png = tmp_path / "iter_reward_heatmap.png"
    assert paths["heatmap"] == str(heatmap_png)
    for png in (plan_step_png, heatmap_png):
        assert png.exists()
        assert png.stat().st_size > 0
        with Image.open(png) as img:
            img.verify()
