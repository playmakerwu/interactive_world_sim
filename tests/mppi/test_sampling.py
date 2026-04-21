"""Step 3 tests for batched reward.

Three tests:
  1. batched_state_reward returns the expected shape and finite values
     on 16 real-ish RGB inputs (encoded-then-decoded frames).
  2. batched_state_reward yields the same per-sample rewards as looping
     state_reward (consistency / no shape reorder bug).
  3. score_latents end-to-end: (N, C, H, W) latents -> (N,) rewards.

The CV-path pieces (resolution mismatch error, CV-fail penalty) are
covered in test_reward.py; these tests focus on the batching harness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rl.labeling.cv_labeler import CVLabeler
from rl.mppi.reward import batched_state_reward, score_latents, state_reward

CKPT_PATH = Path("outputs/pusht_cam1/checkpoints/best.ckpt")
Z_GOAL_PATH = Path("tests/goal_selection/z_goal.pt")
STATE_GOAL_PATH = Path("tests/goal_selection/state_goal.pt")
RES = 128


def _have_gpu_and_ckpt() -> bool:
    return torch.cuda.is_available() and CKPT_PATH.exists()


@pytest.fixture(scope="module")
def state_goal():
    g = torch.load(STATE_GOAL_PATH, map_location="cpu")
    return {
        "cx": g["cx"],
        "cy": g["cy"],
        "sin_theta": g["sin_theta"],
        "cos_theta": g["cos_theta"],
    }


@pytest.fixture(scope="module")
def labeler():
    return CVLabeler(preset="REAL", resolution=RES)


@pytest.fixture(scope="module")
def wm():
    if not _have_gpu_and_ckpt():
        pytest.skip("CUDA + WM checkpoint required")
    from rl.models.world_model import DifferentiableDynamics
    return DifferentiableDynamics(str(CKPT_PATH), device="cuda:0")


@pytest.fixture(scope="module")
def goal_rgb_batch(wm):
    """Produce N=16 real-ish RGB frames by decoding z_goal through the
    WM 16 times (the decoder is stochastic, so we get N different RGBs
    from the same latent without having to load a dataset)."""
    z = torch.load(Z_GOAL_PATH, map_location="cuda:0")
    z_batch = z.expand(16, -1, -1, -1).contiguous()
    with torch.no_grad():
        rgb = wm.decode(z_batch, resolution=RES)  # (16, 3, 128, 128) in [0, 1]
    arr = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return arr.transpose(0, 2, 3, 1)  # (16, 128, 128, 3)


def test_batched_reward_shape_and_finite(goal_rgb_batch, state_goal, labeler):
    rewards, labels = batched_state_reward(
        goal_rgb_batch, state_goal, labeler=labeler,
    )
    assert rewards.shape == (16,)
    assert rewards.dtype == np.float64
    assert np.all(np.isfinite(rewards))
    assert np.all(~np.isnan(rewards))
    assert len(labels) == 16
    print(
        f"[shape+finite] rewards mean={rewards.mean():.4f}, "
        f"min={rewards.min():.4f}, max={rewards.max():.4f}"
    )


def test_batched_reward_matches_single_loop(goal_rgb_batch, state_goal, labeler):
    """Looping state_reward on each frame must give the same values as
    calling batched_state_reward once."""
    rewards_batched, _ = batched_state_reward(
        goal_rgb_batch, state_goal, labeler=labeler,
    )
    rewards_loop = np.array(
        [state_reward(frame, state_goal, labeler=labeler)[0]
         for frame in goal_rgb_batch],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(rewards_batched, rewards_loop)


def test_score_latents_end_to_end(wm, state_goal, labeler):
    """z -> decode -> CV -> reward should produce finite (N,) rewards."""
    z = torch.load(Z_GOAL_PATH, map_location="cuda:0")
    z_batch = z.expand(8, -1, -1, -1).contiguous()
    rewards, labels = score_latents(
        z_batch, state_goal, wm, labeler=labeler, resolution=RES,
    )
    assert rewards.shape == (8,)
    assert np.all(np.isfinite(rewards))
    # All 8 are decodes of z_goal, so all should be close to zero.
    assert rewards.mean() > -0.2, f"mean reward on z_goal too low: {rewards.mean():.4f}"
    assert all(lbl.success for lbl in labels)
    print(
        f"[score_latents] rewards: mean={rewards.mean():.4f} "
        f"std={rewards.std():.4f} min={rewards.min():.4f} max={rewards.max():.4f}"
    )


def test_batched_reward_accepts_torch_tensor(wm, state_goal, labeler):
    """The decoder returns (N, 3, H, W) torch tensors in [0, 1]. The
    batched API must accept this directly — no callers should have to
    convert to numpy first."""
    z = torch.load(Z_GOAL_PATH, map_location="cuda:0")
    z_batch = z.expand(4, -1, -1, -1).contiguous()
    with torch.no_grad():
        rgb = wm.decode(z_batch, resolution=RES)
    assert rgb.shape == (4, 3, RES, RES)
    assert rgb.dtype == torch.float32
    rewards, _ = batched_state_reward(rgb, state_goal, labeler=labeler)
    assert rewards.shape == (4,)
    assert np.all(np.isfinite(rewards))
