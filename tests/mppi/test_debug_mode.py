"""Tests for yaml-gated comprehensive debug_mode in MPPIPlanner.

Six new yaml fields control the feature (all default False / null):
- ``debug_mode``: master switch; auto-promotes ``audit_log_*`` to True
- ``debug_dump_dir``: target dir (None ⇒ runner default <output_dir>/debug/)
- ``debug_dump_per_sample_rollout``: per-niter (N, H+1, C, H_lat, W_lat) latents
- ``debug_dump_per_h_cv``: per-niter (H+1, 5) mean-rollout CV pose
- ``debug_dump_decoded_full``: Tier 2 (NOT YET IMPLEMENTED — flag reserved)
- ``debug_storage_warning_gb``: runner-side warning threshold

These tests exercise the planner side; runner-side init / executed_step /
manifest dumps are smoke-tested in a dedicated validation run rather than
mocked here (would require loading the WM checkpoint).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rl.mppi.mppi_planner import MPPIPlanner

DEVICE = "cpu"


class MockEnv:
    """Same lightweight env stub used in test_audit_log + test_iteration_log."""

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
            shape = ()
        else:
            success = torch.ones(z_flat.shape[0], dtype=torch.bool)
            shape = (z_flat.shape[0],)
        out = {
            "_state": z_flat,
            "success": success,
            "cx": torch.zeros(shape, dtype=torch.float32),
            "cy": torch.zeros(shape, dtype=torch.float32),
            "theta_deg": torch.zeros(shape, dtype=torch.float32),
            "icp_residual": torch.zeros(shape, dtype=torch.float32),
        }
        return out

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
        "n_update_iter": 2,
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


def test_debug_mode_disabled_default(tmp_path: Path) -> None:
    """No debug_mode field ⇒ no debug dir written even when set."""
    cfg = _base_cfg()  # no debug_mode field at all
    p = MPPIPlanner(MockEnv(), cfg)
    p.set_debug_dir(tmp_path)
    p.trajectory_optimization(_z0(), _goal())
    # debug_mode is off so the planner should not write to the debug dir
    written = list(tmp_path.rglob("*.npz"))
    assert written == [], (
        f"debug_mode disabled by default — no npz should appear; got {written}"
    )


def test_debug_mode_enabled_writes_per_niter(tmp_path: Path) -> None:
    """debug_mode=True + set_debug_dir ⇒ one npz per niter under plan_NNN/."""
    cfg = _base_cfg(debug_mode=True)
    p = MPPIPlanner(MockEnv(), cfg)
    p.set_debug_dir(tmp_path)
    p.trajectory_optimization(_z0(), _goal())
    n_iter = int(cfg.n_update_iter)
    plan_dir = tmp_path / "plan_000"
    assert plan_dir.exists()
    files = sorted(plan_dir.glob("niter_*.npz"))
    assert len(files) == n_iter, f"expected {n_iter} niter files; got {len(files)}"
    data = np.load(files[0])
    keys = set(data.files)
    expected = {"samples", "sample_rewards", "softmax_weights", "mean",
                "best_sample_idx", "best_sample_reward"}
    assert expected.issubset(keys), f"missing keys: {expected - keys}"
    # shapes
    N = int(cfg.n_sample)
    H = int(cfg.n_look_ahead)
    A = int(cfg.action_dim)
    assert data["samples"].shape == (N, H, A)
    assert data["sample_rewards"].shape == (N,)
    assert data["softmax_weights"].shape == (N,)
    assert data["mean"].shape == (H, A)


def test_debug_per_sample_rollout_off_by_default(tmp_path: Path) -> None:
    """debug_mode=True but debug_dump_per_sample_rollout omitted ⇒
    per_sample_rollout_latents key absent from npz."""
    cfg = _base_cfg(debug_mode=True)
    p = MPPIPlanner(MockEnv(), cfg)
    p.set_debug_dir(tmp_path)
    p.trajectory_optimization(_z0(), _goal())
    data = np.load(tmp_path / "plan_000" / "niter_00.npz")
    assert "per_sample_rollout_latents" not in data.files, (
        "per-sample rollout latents must be opt-in"
    )


def test_debug_per_sample_rollout_on_writes_latents(tmp_path: Path) -> None:
    """debug_dump_per_sample_rollout=True ⇒ key present, correct shape."""
    cfg = _base_cfg(debug_mode=True, debug_dump_per_sample_rollout=True)
    p = MPPIPlanner(MockEnv(), cfg)
    p.set_debug_dir(tmp_path)
    p.trajectory_optimization(_z0(), _goal())
    data = np.load(tmp_path / "plan_000" / "niter_00.npz")
    assert "per_sample_rollout_latents" in data.files
    arr = data["per_sample_rollout_latents"]
    # MockEnv produces (N, H+1, latent_dims...). Latent has shape (4,1,1)
    # → flatten test: just check leading 2 dims.
    N = int(cfg.n_sample)
    H = int(cfg.n_look_ahead)
    assert arr.shape[0] == N
    assert arr.shape[1] == H + 1


def test_debug_per_h_cv_on_writes_mean_cv(tmp_path: Path) -> None:
    """debug_dump_per_h_cv=True ⇒ mean_cv_per_h key present, shape (H+1, 5)."""
    cfg = _base_cfg(debug_mode=True, debug_dump_per_h_cv=True)
    p = MPPIPlanner(MockEnv(), cfg)
    p.set_debug_dir(tmp_path)
    p.trajectory_optimization(_z0(), _goal())
    data = np.load(tmp_path / "plan_000" / "niter_00.npz")
    assert "mean_cv_per_h" in data.files, (
        "per-H CV must be present when flag set"
    )
    cv = data["mean_cv_per_h"]
    H = int(cfg.n_look_ahead)
    assert cv.shape == (H + 1, 5), f"expected ({H + 1}, 5); got {cv.shape}"


def test_debug_audit_log_auto_promoted(tmp_path: Path) -> None:
    """debug_mode=True with audit_log_enabled left False ⇒ audit log
    still populated (auto-promotion)."""
    cfg = _base_cfg(
        debug_mode=True,
        audit_log_enabled=False,
        audit_log_record_samples=False,
    )
    p = MPPIPlanner(MockEnv(), cfg)
    p.set_debug_dir(tmp_path)
    p.trajectory_optimization(_z0(), _goal())
    log = p.get_audit_log()
    n_iter = int(cfg.n_update_iter)
    assert len(log) == n_iter + 1, (
        f"debug_mode must auto-promote audit_log; expected "
        f"{n_iter + 1} entries, got {len(log)}"
    )
    # in-loop entries should carry samples + softmax_weights (debug extras)
    for entry in log[1:]:
        assert "samples" in entry, "auto-promoted audit must record samples"
        assert "softmax_weights" in entry, "debug entries must carry weights"


def test_debug_dir_none_no_writes(tmp_path: Path) -> None:
    """debug_mode=True but set_debug_dir(None) ⇒ no files written
    (escape hatch for in-process callers that don't want disk i/o)."""
    cfg = _base_cfg(debug_mode=True)
    p = MPPIPlanner(MockEnv(), cfg)
    # do NOT call set_debug_dir — default is None
    p.trajectory_optimization(_z0(), _goal())
    # audit log still gets entries (promoted), but no files in tmp_path
    assert list(tmp_path.rglob("*.npz")) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
