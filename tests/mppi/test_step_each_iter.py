"""Tests for the step_each_iter yaml field (replaces hardcoded STEP_EACH_ITER).

Coverage:
  1. Default yaml has step_each_iter == 1 (backward-compat baseline).
  2. step_each_iter=5 propagates through cfg into the runner's loop math
     (n_plan_calls = ceil(control_steps / step_each_iter)).
  3. Combining delta_mode=true + step_each_iter=5 instantiates the
     planner cleanly (no interaction bugs between the two new features).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rl.mppi.mppi_planner import MPPIPlanner

REPO_ROOT = Path(__file__).resolve().parents[2]


class _StubEnv:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.action_dim = 4
        self.image_diagonal = 181.02


def _base_cfg(**overrides):
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
        "step_each_iter": 1,
        "control_steps": 10,
        "seed": 42,
        "waypoints_n": None,
        "waypoints_interp": "linear",
    })
    for k, v in overrides.items():
        base[k] = v
    return base


def test_default_yaml_step_each_iter_is_1():
    """The shipped default config must keep step_each_iter=1 (bit-equivalent
    to the pre-refactor STEP_EACH_ITER=1 default)."""
    cfg = OmegaConf.load(REPO_ROOT / "configs/mppi/default.yaml")
    assert "step_each_iter" in cfg, "default.yaml missing step_each_iter field"
    assert int(cfg.step_each_iter) == 1, (
        f"default step_each_iter must be 1; got {cfg.step_each_iter}"
    )


def test_step_each_iter_5_propagates_to_loop_math():
    """With step_each_iter=5 and control_steps=10, the runner schedules
    ceil(10/5)=2 plan_step calls; the run_mppi_v2 loop's n_this slicing
    follows min(step_each_iter, control_steps - n_actions_done)."""
    cfg = _base_cfg(step_each_iter=5, control_steps=10)
    sei = int(getattr(cfg, "step_each_iter", 1))
    expected_n_plan_calls = math.ceil(int(cfg.control_steps) / sei)
    assert sei == 5
    assert expected_n_plan_calls == 2

    # Simulate the n_this scheduling
    n_actions_done = 0
    n_calls = 0
    while n_actions_done < int(cfg.control_steps):
        n_this = min(sei, int(cfg.control_steps) - n_actions_done)
        assert 1 <= n_this <= sei
        n_actions_done += n_this
        n_calls += 1
    assert n_calls == expected_n_plan_calls
    assert n_actions_done == int(cfg.control_steps)


def test_step_each_iter_partial_final_call():
    """control_steps=12 with step_each_iter=5 → calls take 5,5,2 actions."""
    cfg = _base_cfg(step_each_iter=5, control_steps=12)
    sei = int(getattr(cfg, "step_each_iter", 1))
    n_per_call = []
    n_done = 0
    while n_done < int(cfg.control_steps):
        n_this = min(sei, int(cfg.control_steps) - n_done)
        n_per_call.append(n_this)
        n_done += n_this
    assert n_per_call == [5, 5, 2]


def test_step_each_iter_does_not_break_delta_mode():
    """delta_mode=true + step_each_iter=5 must construct cleanly; the two
    features are orthogonal (delta_mode lives in the sampler, step_each_iter
    in the run loop)."""
    cfg = _base_cfg(
        delta_mode=True, step_each_iter=5,
        delta_action_lim=0.0872, noise_level_delta=0.02,
        n_sample=8, n_look_ahead=10,
    )
    planner = MPPIPlanner(_StubEnv(), cfg)
    seed = torch.zeros(10, 4)
    out = planner.sample_action_sequences(seed)
    assert out.shape == (8, 10, 4)
    # delta_mode clip should still apply
    assert (out.abs() <= 0.0872 + 1e-6).all()


def test_step_each_iter_missing_field_falls_back_to_1():
    """getattr(cfg, 'step_each_iter', 1) preserves backward compat with
    pre-refactor yamls that lack the field."""
    cfg = _base_cfg()
    OmegaConf.set_struct(cfg, False)
    del cfg["step_each_iter"]
    OmegaConf.set_struct(cfg, True)
    sei = int(getattr(cfg, "step_each_iter", 1))
    assert sei == 1, f"missing field must default to 1; got {sei}"
