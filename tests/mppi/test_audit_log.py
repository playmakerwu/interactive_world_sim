"""Tests for yaml-gated audit logging in MPPIPlanner.

Three new yaml fields control the feature (all default to False):
- ``audit_log_enabled``: master switch
- ``audit_log_record_samples``: include (N, H, A) per-niter samples + rewards
- ``audit_log_path``: dump location (consumed by the runner, not the planner)

Tests exercise the planner side; runner-side path resolution is
verified by inspection (one short ``getattr`` call in ``_run_episode``).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rl.mppi.mppi_planner import MPPIPlanner

DEVICE = "cpu"


class MockEnv:
    """Same lightweight env stub as test_iteration_log.MockEnv."""

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


def _base_cfg(**overrides) -> OmegaConf:
    cfg = OmegaConf.create({
        "n_sample": 6,
        "n_look_ahead": 4,
        "n_update_iter": 3,
        "noise_level": 0.05,
        "reward_weight": 200.0,
        "beta_filter": 0.7,
        "cv_fail_penalty": -10.0,
        "action_dim": 4,
        "action_lower_lim": [-1.0, -1.0, -1.0, -1.0],
        "action_upper_lim": [1.0, 1.0, 1.0, 1.0],
        "seed": 0,
    })
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    return cfg


def _z0() -> torch.Tensor:
    return torch.zeros(4, 1, 1)


def _goal() -> dict:
    return {"_state": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "cx": 0.0, "cy": 0.0, "sin_theta": 0.0, "cos_theta": 1.0}


def test_audit_log_disabled_by_default() -> None:
    """No audit fields in cfg ⇒ get_audit_log() empty after a plan call."""
    cfg = _base_cfg()  # no audit_log_* fields
    p = MPPIPlanner(MockEnv(), cfg)
    assert p.get_audit_log() == []
    p.trajectory_optimization(_z0(), _goal())
    assert p.get_audit_log() == [], (
        "feature disabled by default — log must remain empty"
    )


def test_audit_log_explicitly_disabled() -> None:
    """audit_log_enabled=False ⇒ log empty (same as default)."""
    cfg = _base_cfg(audit_log_enabled=False, audit_log_record_samples=True)
    p = MPPIPlanner(MockEnv(), cfg)
    p.trajectory_optimization(_z0(), _goal())
    assert p.get_audit_log() == []


def test_audit_log_records_per_niter_with_init() -> None:
    """audit_log_enabled=True ⇒ log = init + n_update_iter entries."""
    cfg = _base_cfg(audit_log_enabled=True)
    p = MPPIPlanner(MockEnv(), cfg)
    p.trajectory_optimization(_z0(), _goal())
    log = p.get_audit_log()
    n_iter = int(cfg.n_update_iter)
    assert len(log) == n_iter + 1, (
        f"expected {n_iter + 1} entries (1 init + {n_iter} iters); got {len(log)}"
    )
    # init entry
    assert log[0]["niter"] == -1
    assert log[0]["best_sample_reward"] is None
    assert "samples" not in log[0]
    # subsequent entries
    for i, entry in enumerate(log[1:]):
        assert entry["niter"] == i
        assert entry["plan_call_idx"] == 0
        assert isinstance(entry["best_sample_reward"], float)
        assert "samples" not in entry, (
            "record_samples=False ⇒ samples must NOT be present"
        )
        # mean shape (H, A)
        mean = entry["mean"]
        assert len(mean) == int(cfg.n_look_ahead)
        assert len(mean[0]) == int(cfg.action_dim)


def test_audit_log_record_samples_includes_samples() -> None:
    """record_samples=True ⇒ in-loop entries have ``samples`` and ``sample_rewards``."""
    cfg = _base_cfg(audit_log_enabled=True, audit_log_record_samples=True)
    p = MPPIPlanner(MockEnv(), cfg)
    p.trajectory_optimization(_z0(), _goal())
    log = p.get_audit_log()

    N = int(cfg.n_sample)
    H = int(cfg.n_look_ahead)
    A = int(cfg.action_dim)

    # init entry never has samples (no sampling has happened yet)
    assert "samples" not in log[0]

    for entry in log[1:]:
        assert "samples" in entry, "record_samples=True ⇒ samples expected"
        assert "sample_rewards" in entry
        samples = entry["samples"]
        rewards = entry["sample_rewards"]
        assert len(samples) == N
        assert len(samples[0]) == H
        assert len(samples[0][0]) == A
        assert len(rewards) == N


def test_audit_log_increments_plan_call_idx_across_calls() -> None:
    """plan_call_idx must increment per trajectory_optimization call."""
    cfg = _base_cfg(audit_log_enabled=True)
    p = MPPIPlanner(MockEnv(), cfg)
    p.trajectory_optimization(_z0(), _goal())
    p.trajectory_optimization(_z0(), _goal())
    log = p.get_audit_log()
    n_iter = int(cfg.n_update_iter)
    expected = 2 * (n_iter + 1)
    assert len(log) == expected, (
        f"two calls → {expected} entries; got {len(log)}"
    )
    # First call's entries all have plan_call_idx=0; second's all =1
    half = n_iter + 1
    assert all(e["plan_call_idx"] == 0 for e in log[:half])
    assert all(e["plan_call_idx"] == 1 for e in log[half:])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
