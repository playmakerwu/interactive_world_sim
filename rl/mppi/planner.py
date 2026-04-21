"""MPPI planner: one planning step = sample -> rollout -> decode -> reward -> softmax.

This implements the v0 algorithm from MPPI_NOTES.md §Algorithm. No warm
start, no trajectory refinement across iterations — those are v1 features
that we will only reach for if the v0 loop converges on PushT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch

from rl.labeling.cv_labeler import CVLabeler, CVLabelResult
from rl.mppi.action_sampling import ActionSampler, GaussianSampler
from rl.mppi.reward import (
    DEFAULT_LARGE_PENALTY,
    IMAGE_DIAGONAL_128,
    batched_state_reward,
)
from rl.mppi.utils import batched_rollout


@dataclass
class PlanStepStats:
    """Debug stats captured on each plan_step call. Used by viz and by
    Step 5 trajectory recording."""

    a_star: torch.Tensor        # (A,) the action that will be executed
    a_naive_mean: torch.Tensor  # (A,) simple mean of first actions (for comparison)
    rewards: np.ndarray         # (N,) per-trajectory reward
    weights: np.ndarray         # (N,) softmax weights
    labels: list[CVLabelResult] = field(default_factory=list)
    actions: torch.Tensor | None = None    # (N, H, A) sampled action sequences
    final_latents: torch.Tensor | None = None  # (N, C, H, W) z_H per sample
    decoded_rgb: np.ndarray | None = None  # (N, H, W, 3) uint8 — optional, for viz
    cv_fail_count: int = 0


class MPPIPlanner:
    """Plan one action at a time via MPPI sampling + softmax.

    Construction arguments configure the sampling + reward behaviour;
    `plan_step` is the per-timestep entry point. Internal debug state
    from the last plan_step is exposed via `self.last_stats` so Step 5
    can record it to disk.
    """

    def __init__(
        self,
        wm,
        state_goal: Mapping[str, float],
        *,
        N: int = 128,
        H: int = 10,
        sigma: float = 0.1,
        temperature: float = 1.0,
        action_dim: int = 4,
        resolution: int = 128,
        image_diagonal: float = IMAGE_DIAGONAL_128,
        large_penalty: float = DEFAULT_LARGE_PENALTY,
        symmetry_aware: bool = False,
        action_sampler: ActionSampler | None = None,
        labeler: CVLabeler | None = None,
        device: str = "cuda:0",
        capture_rgb: bool = False,
    ) -> None:
        self.wm = wm
        self.state_goal = dict(state_goal)
        self.N = N
        self.H = H
        self.sigma = sigma
        self.temperature = temperature
        self.action_dim = action_dim
        self.resolution = resolution
        self.image_diagonal = image_diagonal
        self.large_penalty = large_penalty
        self.symmetry_aware = symmetry_aware
        self.action_sampler: ActionSampler = action_sampler or GaussianSampler(
            sigma=sigma, action_dim=action_dim
        )
        if self.action_sampler.action_dim != action_dim:
            raise ValueError(
                f"action_sampler.action_dim={self.action_sampler.action_dim} "
                f"!= planner action_dim={action_dim}"
            )
        self.labeler = labeler or CVLabeler(preset="REAL", resolution=resolution)
        if self.labeler.resolution != resolution:
            raise ValueError(
                f"labeler resolution {self.labeler.resolution} != {resolution}"
            )
        self.device = device
        self.capture_rgb = capture_rgb  # Step 4/5 viz; adds a (N, H, W, 3) uint8 copy
        self.last_stats: PlanStepStats | None = None

    def _sample_actions(self) -> torch.Tensor:
        return self.action_sampler.sample(self.N, self.H, self.device)

    def plan_step(
        self,
        z_current: torch.Tensor,
        *,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Plan one action from a single current latent.

        Args:
            z_current: (C, H, W) or (1, C, H, W) latent of the current
                world state. Copied N times internally.
            seed: optional seed for action-noise reproducibility. If
                given, a fresh torch.Generator is created — does not
                touch the global RNG.

        Returns:
            a_star: (action_dim,) tensor, the weighted-mean action to
            execute. Side effect: `self.last_stats` is populated with
            the full debug record of this plan step.
        """
        if z_current.dim() == 3:
            z_current = z_current.unsqueeze(0)
        if z_current.dim() != 4 or z_current.shape[0] != 1:
            raise ValueError(
                f"z_current must be (C, H, W) or (1, C, H, W); got {tuple(z_current.shape)}"
            )
        z_current = z_current.to(self.device)

        # 1. Sample N action sequences.
        # NOTE: the WM dynamics inject fresh noise per denoising step using
        # the global CUDA RNG (see world_model.py). A local torch.Generator
        # for action noise alone is therefore not enough to guarantee
        # reproducibility — seed the global RNGs too. Caller is responsible
        # for passing a different seed per plan_step in a planning loop.
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        actions = self._sample_actions()  # (N, H, A)

        # 2. Batched rollout through IWS dynamics.
        z0_batch = z_current.expand(self.N, -1, -1, -1).contiguous()
        with torch.no_grad():
            latents = batched_rollout(z0_batch, actions, self.wm)  # (N, H+1, C, H, W)

        # 3. Decode last-step latents.
        z_last = latents[:, -1].clone()
        del latents
        torch.cuda.empty_cache()  # release rollout scratch (Step 3 finding)
        with torch.no_grad():
            rgb = self.wm.decode(z_last, resolution=self.resolution)  # (N, 3, H, W)

        # 4. Compute reward per trajectory.
        rgb_u8 = (
            rgb.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255
        ).astype(np.uint8)
        rewards, labels = batched_state_reward(
            rgb_u8,
            self.state_goal,
            labeler=self.labeler,
            image_diagonal=self.image_diagonal,
            large_penalty=self.large_penalty,
            symmetry_aware=self.symmetry_aware,
        )
        cv_fail_count = sum(1 for lbl in labels if not lbl.success)

        # 5. Softmax weighting.
        rewards_t = torch.from_numpy(rewards).to(
            device=actions.device, dtype=actions.dtype
        )
        shifted = (rewards_t - rewards_t.max()) / self.temperature
        weights = torch.softmax(shifted, dim=0)  # (N,)
        # Defensive: if all rewards are -large_penalty the softmax is
        # uniform, but if temperature is tiny and numerical issues
        # produced NaN, fall back to uniform.
        if torch.isnan(weights).any():
            weights = torch.full_like(weights, 1.0 / self.N)

        # 6. Weighted mean of the first action of each trajectory.
        first_actions = actions[:, 0]  # (N, A)
        a_star = (weights.unsqueeze(-1) * first_actions).sum(dim=0)  # (A,)
        a_naive_mean = first_actions.mean(dim=0)

        self.last_stats = PlanStepStats(
            a_star=a_star.detach().cpu(),
            a_naive_mean=a_naive_mean.detach().cpu(),
            rewards=rewards,
            weights=weights.detach().cpu().numpy(),
            labels=labels,
            actions=actions.detach().cpu(),
            final_latents=z_last.detach().cpu(),
            decoded_rgb=rgb_u8 if self.capture_rgb else None,
            cv_fail_count=cv_fail_count,
        )

        return a_star
