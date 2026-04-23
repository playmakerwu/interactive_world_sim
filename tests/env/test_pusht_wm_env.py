"""Tests for env.PushTWMEnv.

Six tests per Task B spec:
  1. encode/decode roundtrip is visually coherent
  2. rollout is batched-consistent
  3. estimate_state produces expected keys
  4. estimate_from_latent matches decode + estimate_state
  5. compute_reward is correct
  6. load_initial_from_hdf5 enforces camera_1_color (regression guard)
"""

from __future__ import annotations

import inspect
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from env.pusht_wm_env import PUSHT_CAMERA_KEY, PushTWMEnv

CKPT_PATH = Path("outputs/pusht_cam1/checkpoints/best.ckpt")
SAMPLE_HDF5 = Path("data/mini/pusht/train/episode_3.hdf5")
GOAL_PT = Path("tests/goal_selection/state_goal.pt")
RES = 128


def _have_gpu_and_ckpt() -> bool:
    return torch.cuda.is_available() and CKPT_PATH.exists()


@pytest.fixture(scope="module")
def env() -> PushTWMEnv:
    if not _have_gpu_and_ckpt():
        pytest.skip("CUDA + WM checkpoint required")
    return PushTWMEnv(str(CKPT_PATH), device="cuda:0", resolution=RES)


@pytest.fixture(scope="module")
def sample_rgb_chw() -> torch.Tensor:
    """One real frame from data/mini/pusht/train/episode_3 via camera_1_color,
    pre-processed to (3, 128, 128) float [0, 1]."""
    if not SAMPLE_HDF5.exists():
        pytest.skip(f"missing dataset: {SAMPLE_HDF5}")
    import cv2
    with h5py.File(str(SAMPLE_HDF5), "r") as f:
        raw = f[f"obs/images/{PUSHT_CAMERA_KEY}"][50]
    h, w = raw.shape[:2]
    s = min(h, w)
    cropped = raw[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    resized = cv2.resize(cropped, (RES, RES), interpolation=cv2.INTER_AREA)
    pre = resized.astype(np.float32) / 255.0
    return torch.from_numpy(pre).permute(2, 0, 1)  # (3, 128, 128)


# ─── 1. encode / decode roundtrip is visually coherent ─────────────────

def test_encode_decode_roundtrip_coherent(env, sample_rgb_chw):
    z = env.encode(sample_rgb_chw)
    assert z.dim() == 3, f"expected (C, H, W), got {tuple(z.shape)}"
    assert z.shape[1:] == (32, 32), f"unexpected latent grid: {tuple(z.shape)}"
    norm = float(z.flatten().norm())
    assert abs(norm - 32.0) < 0.5, f"latent norm {norm} not ≈ 32"

    rgb_back = env.decode(z)
    assert rgb_back.shape == (3, RES, RES)
    assert rgb_back.dtype == torch.float32
    assert float(rgb_back.min()) >= 0.0 and float(rgb_back.max()) <= 1.0

    diff = (rgb_back.cpu() - sample_rgb_chw).abs().mean()
    assert diff < 0.2, f"decoded mean abs diff {diff:.3f} too large (expect < 0.2)"


# ─── 2. rollout is batched-consistent ──────────────────────────────────

def test_rollout_batched_consistent(env, sample_rgb_chw):
    z0 = env.encode(sample_rgb_chw)  # (C, H, W)
    z0_batch = z0.unsqueeze(0).expand(4, -1, -1, -1).contiguous()  # (4, C, H, W)

    H = 5
    actions = torch.zeros(4, H, env.action_dim, device="cuda:0")
    actions[0] = 0.0
    actions[1] = 0.0
    torch.manual_seed(0)
    actions[2] = torch.randn(H, env.action_dim, device="cuda:0") * 0.05
    actions[3] = torch.randn(H, env.action_dim, device="cuda:0") * 0.05

    traj = env.rollout(z0_batch, actions)
    assert traj.shape == (4, H + 1, *z0.shape), f"unexpected traj shape {tuple(traj.shape)}"

    # First time-step is the unmodified z0 for every sample (no action applied yet).
    for k in range(4):
        torch.testing.assert_close(traj[k, 0].cpu(), z0.cpu(), rtol=0, atol=0)

    # Random-action rollouts should diverge from zero-action rollouts.
    diff_zero_zero = (traj[0, -1] - traj[1, -1]).flatten().norm()
    diff_zero_rand = (traj[0, -1] - traj[2, -1]).flatten().norm()
    assert diff_zero_rand > diff_zero_zero, (
        f"random-action divergence ({diff_zero_rand:.3f}) should exceed "
        f"zero-action stochasticity ({diff_zero_zero:.3f})"
    )


# ─── 3. estimate_state returns the expected keys ───────────────────────

def test_estimate_state_keys(env, sample_rgb_chw):
    state = env.estimate_state(sample_rgb_chw)
    expected_keys = {
        "cx", "cy", "sin_theta", "cos_theta", "theta_deg",
        "success", "contour_area", "icp_residual",
    }
    assert set(state.keys()) == expected_keys, (
        f"missing/extra keys: {set(state.keys()) ^ expected_keys}"
    )
    assert state["success"].dtype == torch.bool
    # All numeric keys are scalar tensors for this single-frame call.
    for k in expected_keys - {"success"}:
        assert state[k].dim() == 0, f"{k} should be scalar; got {state[k].shape}"


# ─── 4. estimate_from_latent ≈ decode + estimate_state ─────────────────

def test_estimate_from_latent_matches_decode_then_estimate(env, sample_rgb_chw):
    z = env.encode(sample_rgb_chw)
    state_a = env.estimate_from_latent(z)
    rgb = env.decode(z)
    state_b = env.estimate_state(rgb)
    assert set(state_a.keys()) == set(state_b.keys())
    if state_a["success"].item() and state_b["success"].item():
        # Decoder is deterministic given fixed seed; with no seeding these
        # may differ slightly due to denoise stochasticity. Allow loose tol.
        assert abs(float(state_a["cx"]) - float(state_b["cx"])) < 5.0
        assert abs(float(state_a["cy"]) - float(state_b["cy"])) < 5.0


# ─── 5. compute_reward correctness ─────────────────────────────────────

def test_compute_reward_at_goal_is_zero(env):
    goal = {"cx": 64.0, "cy": 64.0, "sin_theta": 0.0, "cos_theta": 1.0}
    state = {
        "cx": torch.tensor(64.0),
        "cy": torch.tensor(64.0),
        "sin_theta": torch.tensor(0.0),
        "cos_theta": torch.tensor(1.0),
        "theta_deg": torch.tensor(0.0),
        "success": torch.tensor(True),
        "contour_area": torch.tensor(0.0),
        "icp_residual": torch.tensor(0.0),
    }
    r = env.compute_reward(state, goal, image_diagonal=181.0)
    assert abs(float(r)) < 1e-5, f"reward at goal should be 0, got {r}"


def test_compute_reward_cv_fail_returns_penalty(env):
    goal = {"cx": 64.0, "cy": 64.0, "sin_theta": 0.0, "cos_theta": 1.0}
    state = {
        "cx": torch.tensor(float("nan")),
        "cy": torch.tensor(float("nan")),
        "sin_theta": torch.tensor(float("nan")),
        "cos_theta": torch.tensor(float("nan")),
        "theta_deg": torch.tensor(float("nan")),
        "success": torch.tensor(False),
        "contour_area": torch.tensor(float("nan")),
        "icp_residual": torch.tensor(float("nan")),
    }
    r = env.compute_reward(state, goal, image_diagonal=181.0, cv_fail_penalty=-10.0)
    assert float(r) == -10.0


def test_compute_reward_far_position(env):
    goal = {"cx": 64.0, "cy": 64.0, "sin_theta": 0.0, "cos_theta": 1.0}
    state = {
        "cx": torch.tensor(64.0 + 50.0),
        "cy": torch.tensor(64.0),
        "sin_theta": torch.tensor(0.0),
        "cos_theta": torch.tensor(1.0),
        "theta_deg": torch.tensor(0.0),
        "success": torch.tensor(True),
        "contour_area": torch.tensor(0.0),
        "icp_residual": torch.tensor(0.0),
    }
    r = env.compute_reward(state, goal, image_diagonal=181.0)
    expected = -50.0 / 181.0
    assert abs(float(r) - expected) < 1e-4, f"got {r}, expected ≈ {expected:.4f}"


def test_compute_reward_180_flip(env):
    goal = {"cx": 64.0, "cy": 64.0, "sin_theta": 0.0, "cos_theta": 1.0}
    state = {
        "cx": torch.tensor(64.0),
        "cy": torch.tensor(64.0),
        "sin_theta": torch.tensor(0.0),
        "cos_theta": torch.tensor(-1.0),
        "theta_deg": torch.tensor(180.0),
        "success": torch.tensor(True),
        "contour_area": torch.tensor(0.0),
        "icp_residual": torch.tensor(0.0),
    }
    r = env.compute_reward(state, goal, image_diagonal=181.0)
    assert abs(float(r) - (-2.0)) < 1e-4, f"180-flip reward should be -2, got {r}"


# ─── 6. load_initial_from_hdf5 enforces camera_1_color ─────────────────

def test_load_initial_uses_camera_1(env):
    """Regression guard against the camera bug. We assert via two
    independent checks: (a) the module-level constant, (b) the source
    text of the function (ensures no future edit silently changes it)."""
    assert PUSHT_CAMERA_KEY == "camera_1_color"
    src = inspect.getsource(env.load_initial_from_hdf5)
    assert "PUSHT_CAMERA_KEY" in src, (
        "load_initial_from_hdf5 should use the module-level constant, "
        "not a hardcoded string — that's how the camera bug originally crept in"
    )
    assert "camera_0" not in src, "camera_0 must not appear in load_initial_from_hdf5"
