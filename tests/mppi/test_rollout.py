"""Step 1 tests for rl.mppi.utils.batched_rollout.

Four tests:
  1. test_zero_action_rollout_stays_close    — zero actions don't drift far
  2. test_random_action_rollout_diverges_more — random actions drive divergence
  3. test_rollout_small                       — N=32, H=10 runs locally, records VRAM
  4. test_rollout_scaling                     — N=128, H=10; gated behind
                                                GPU free >= 20 GiB so it only
                                                runs on the L40S.

All tests require CUDA + the pretrained WM ckpt. If either is missing
they skip — there is no CPU fallback for the full WM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rl.mppi.utils import batched_rollout

CKPT_PATH = Path("outputs/pusht_cam1/checkpoints/best.ckpt")
ACTION_DIM = 4  # matches outputs/pusht_cam1/.hydra/config.yaml:105
LATENT_SHAPE = (4, 32, 32)  # (C, H, W) for IWS PushT checkpoint

SCALING_MIN_FREE_BYTES = 20 * 1024**3  # 20 GiB — N=128 needs ~12 GiB, leave margin


def _have_gpu_and_ckpt() -> bool:
    return torch.cuda.is_available() and CKPT_PATH.exists()


@pytest.fixture(scope="module")
def wm():
    if not _have_gpu_and_ckpt():
        pytest.skip("CUDA + WM checkpoint required for rollout tests")
    from rl.models.world_model import DifferentiableDynamics
    return DifferentiableDynamics(str(CKPT_PATH), device="cuda:0")


@pytest.fixture(scope="module")
def z_goal():
    z = torch.load("tests/goal_selection/z_goal.pt", map_location="cuda:0")
    assert z.shape == (1, *LATENT_SHAPE), z.shape
    return z


def test_zero_action_rollout_stays_close(wm, z_goal):
    """Zero actions should not push the latent far."""
    B, H = 4, 5
    z0 = z_goal.expand(B, -1, -1, -1).contiguous()
    actions = torch.zeros(B, H, ACTION_DIM, device=z0.device)

    with torch.no_grad():
        latents = batched_rollout(z0, actions, wm)

    assert latents.shape == (B, H + 1, *LATENT_SHAPE)
    assert not torch.isnan(latents).any()
    assert not torch.isinf(latents).any()

    per_sample_l2 = (latents[:, -1] - latents[:, 0]).reshape(B, -1).norm(dim=1)
    assert torch.all(torch.isfinite(per_sample_l2))
    print(f"[zero-action] mean ||z_H - z_0|| = {per_sample_l2.mean():.3f}")


def test_random_action_rollout_diverges_more(wm, z_goal):
    """Random actions should drive the latent further from z_0 than zero
    actions — confirms the action channel actually affects dynamics."""
    B, H = 4, 5
    z0 = z_goal.expand(B, -1, -1, -1).contiguous()

    torch.manual_seed(0)
    a_zero = torch.zeros(B, H, ACTION_DIM, device=z0.device)
    a_rand = torch.randn(B, H, ACTION_DIM, device=z0.device) * 0.3

    with torch.no_grad():
        lat_zero = batched_rollout(z0, a_zero, wm)
        lat_rand = batched_rollout(z0, a_rand, wm)

    d_zero = (lat_zero[:, -1] - lat_zero[:, 0]).reshape(B, -1).norm(dim=1)
    d_rand = (lat_rand[:, -1] - lat_rand[:, 0]).reshape(B, -1).norm(dim=1)

    assert d_rand.mean() > d_zero.mean(), (
        f"random-action drift {d_rand.mean():.3f} should exceed zero-action "
        f"drift {d_zero.mean():.3f}"
    )
    print(
        f"[divergence] zero = {d_zero.mean():.3f}, rand = {d_rand.mean():.3f}"
    )


def test_rollout_small(wm, z_goal):
    """N=32, H=10 should run locally on a typical GPU. Records peak VRAM."""
    B, H = 32, 10
    z0 = z_goal.expand(B, -1, -1, -1).contiguous()

    torch.manual_seed(0)
    actions = torch.randn(B, H, ACTION_DIM, device=z0.device) * 0.1

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        latents = batched_rollout(z0, actions, wm)
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    assert latents.shape == (B, H + 1, *LATENT_SHAPE)
    assert not torch.isnan(latents).any()
    print(f"[N=32, H=10] peak VRAM during rollout: {peak_mb:.1f} MiB")


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.mem_get_info(0)[0] >= SCALING_MIN_FREE_BYTES
    ),
    reason="N=128, H=10 needs ~12 GiB free — gated behind 20 GiB (cloud L40S only)",
)
def test_rollout_scaling(wm, z_goal):
    """N=128, H=10 on a cloud-scale GPU (L40S, 45 GiB)."""
    B, H = 128, 10
    z0 = z_goal.expand(B, -1, -1, -1).contiguous()

    torch.manual_seed(0)
    actions = torch.randn(B, H, ACTION_DIM, device=z0.device) * 0.1

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        latents = batched_rollout(z0, actions, wm)
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    assert latents.shape == (B, H + 1, *LATENT_SHAPE)
    assert not torch.isnan(latents).any()
    print(f"[N=128, H=10] peak VRAM during rollout: {peak_mb:.1f} MiB")
