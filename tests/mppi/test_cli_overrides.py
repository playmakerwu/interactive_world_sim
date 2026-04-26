"""Unit tests for scripts.run_mppi_v2._apply_cli_overrides.

The helper applies CLI overrides to the loaded OmegaConf and returns a
config_deviation dict. Tests exercise the helper directly with
SimpleNamespace args, so they don't need the WM, CUDA, or any cv2 I/O.

Rationale for split: the full driver (run_mppi_v2.main) needs the WM
+ GPU + goal state + hdf5 just to parse a config, so end-to-end tests
are impractical for CI. The override logic is small and pure; isolating
it keeps test runtime at ~1 ms.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_mppi_v2.py"


def _load_apply_cli_overrides():
    """Import _apply_cli_overrides from run_mppi_v2.py without running main().

    Regular `import scripts.run_mppi_v2` would execute top-level cv2 +
    omegaconf imports; those are fine but the module is not on sys.path
    as a package. Use importlib.util to get just the helper.
    """
    spec = importlib.util.spec_from_file_location("run_mppi_v2", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_mppi_v2"] = module
    spec.loader.exec_module(module)
    return module._apply_cli_overrides


def _default_cfg():
    return OmegaConf.create({
        "n_sample": 100,
        "n_look_ahead": 10,
        "n_update_iter": 5,
        "noise_level": 0.05,
        "reward_weight": 200.0,
        "beta_filter": 0.7,
        "cv_fail_penalty": -10.0,
        "action_dim": 4,
        "action_lower_lim": [-1.0] * 4,
        "action_upper_lim": [1.0] * 4,
        "control_steps": 50,
        "seed": 0,
    })


def _default_args(**overrides):
    args = SimpleNamespace(
        n_sample=None,
        n_update_iter=None,
        override_reason=None,
        expected_impact=None,
        control_steps=None,
        seed=None,
        decode_batch_size=None,
        horizon=None,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


# ─── 1. no overrides → cfg unchanged, deviation empty ───────────────────

def test_no_overrides_keeps_cfg_defaults():
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()
    dev = apply(cfg, _default_args())
    assert int(cfg.control_steps) == 50
    assert int(cfg.seed) == 0
    assert int(cfg.n_sample) == 100
    assert dev["changed"] == []
    assert dev["reason"] is None


# ─── 2. --control_steps propagates ──────────────────────────────────────

def test_control_steps_override_takes_effect():
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()
    dev = apply(cfg, _default_args(control_steps=7))
    assert int(cfg.control_steps) == 7
    assert int(cfg.seed) == 0
    assert int(cfg.n_sample) == 100
    # control_steps is NOT an algorithm deviation
    assert dev["changed"] == []


# ─── 3. --seed propagates ──────────────────────────────────────────────

def test_seed_override_takes_effect():
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()
    dev = apply(cfg, _default_args(seed=42))
    assert int(cfg.seed) == 42
    # seed is NOT an algorithm deviation
    assert dev["changed"] == []


# ─── 4. --decode_batch_size propagates without deviation ───────────────

def test_decode_batch_size_override_takes_effect():
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()
    dev = apply(cfg, _default_args(decode_batch_size=8))
    assert int(cfg.decode_batch_size) == 8
    # decode_batch_size is a memory-only knob, not an algorithm deviation.
    assert dev["changed"] == []


# ─── 5. --n_sample without --override_reason raises ─────────────────────

def test_n_sample_override_requires_reason():
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()
    with pytest.raises(SystemExit):
        apply(cfg, _default_args(n_sample=16))


# ─── 6. --n_sample + --override_reason works and records deviation ─────

def test_n_sample_override_with_reason_recorded():
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()
    dev = apply(cfg, _default_args(
        n_sample=16,
        override_reason="local VRAM budget",
    ))
    assert int(cfg.n_sample) == 16
    assert dev["changed"] == ["n_sample: 100 -> 16"]
    assert dev["reason"] == "local VRAM budget"
    assert dev["expected_impact_quantified"] is not None


# ─── 7. --n_update_iter + --override_reason works and records deviation ─

def test_n_update_iter_override_with_reason_recorded():
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()
    dev = apply(cfg, _default_args(
        n_update_iter=3,
        override_reason="visualization smoke test",
    ))
    assert int(cfg.n_update_iter) == 3
    assert dev["changed"] == ["n_update_iter: 5 -> 3"]
    assert dev["reason"] == "visualization smoke test"


# ─── 8. all five overrides together ────────────────────────────────────

def test_all_five_overrides_compose():
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()
    dev = apply(cfg, _default_args(
        control_steps=100,
        seed=7,
        n_sample=16,
        n_update_iter=3,
        decode_batch_size=8,
        override_reason="local VRAM budget",
    ))
    assert int(cfg.control_steps) == 100
    assert int(cfg.seed) == 7
    assert int(cfg.n_sample) == 16
    assert int(cfg.n_update_iter) == 3
    assert int(cfg.decode_batch_size) == 8
    assert dev["changed"] == ["n_sample: 100 -> 16", "n_update_iter: 5 -> 3"]


# ─── 9. --horizon override propagates with audit guard ─────────────────

def test_horizon_override_propagates():
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()
    dev = apply(cfg, _default_args(horizon=20, override_reason="long horizon ablation"))
    assert int(cfg.n_look_ahead) == 20
    assert dev["changed"] == ["n_look_ahead: 10 -> 20"]
    assert dev["reason"] == "long horizon ablation"


def test_horizon_override_audit_guard_without_reason():
    """--horizon != default must require --override_reason (algorithm
    deviation, same audit guard as --n_sample / --n_update_iter)."""
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()
    with pytest.raises(SystemExit):
        apply(cfg, _default_args(horizon=20))


def test_horizon_override_above_max_raises_value_error():
    """--horizon 51 raises ValueError BEFORE the audit-guard check, so
    the user gets a clear message about the cap rather than a misleading
    'missing override_reason' error."""
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()
    with pytest.raises(ValueError, match=r"\[1, 50\]"):
        apply(cfg, _default_args(horizon=51, override_reason="trying very long horizon"))


def test_horizon_override_below_min_raises_value_error():
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()
    with pytest.raises(ValueError, match=r"\[1, 50\]"):
        apply(cfg, _default_args(horizon=0, override_reason="boundary"))


def test_horizon_equal_to_default_records_no_deviation():
    """--horizon 10 (== config default) is a no-op; no audit guard fires."""
    apply = _load_apply_cli_overrides()
    cfg = _default_cfg()  # default has n_look_ahead=10
    dev = apply(cfg, _default_args(horizon=10))
    assert int(cfg.n_look_ahead) == 10
    assert dev["changed"] == []
