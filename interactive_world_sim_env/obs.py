"""Observation dataclasses returned by WorldModelEnv.

Per the design:
- Observation: returned by reset(), step(), and observe(). Always carries
  the latest latent and the latent history; rgb is None for reset() /
  step() unless the env was constructed with decode_on_step=True, and is
  ALWAYS populated for observe() (with the preprocessed input RGB —
  not a decoder round-trip).
- BatchedObservation: returned by step_batch(). Carries both latents and
  decoded RGBs because the planner's reward is RGB-based.

All tensor fields are guaranteed detached (requires_grad=False) at the
producer side. Callers must not mutate them; treat as read-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Observation:
    """Observation for single-env reset / step / observe.

    Attributes
    ----------
    latent: (C_latent, H_lat, W_lat) fp32, on env.device.
        The most recent latent — i.e., latent_history[-1].
    latent_history: (T_hist, C_latent, H_lat, W_lat) fp32, on env.device.
        The full sliding window of past latents the dynamics conditions on.
    step: number of step()-or-observe() calls since the most recent
        reset(). 0 right after reset(), regardless of init_window_size.
    rgb: shape consistent across step() and observe() so reward functions
        do not need to branch on the producer:
          - step(): decoder output, HxWx3 uint8 (single-view tasks) or
            dict[obs_key, HxWx3 uint8] (multi-view). Populated only when
            the env was constructed with decode_on_step=True; otherwise None.
          - observe(): the preprocessed input RGB (center-crop + resize to
            resolution x resolution, uint8 HWC, or per-view dict). Always
            populated. NOT a decoder round-trip of the input — the input
            itself is the ground truth at this step.
          - reset(): None.
        Use env.render() to decode on demand from outside step/observe.
        Use info["source"] to disambiguate ("dynamics", "encoder", "reset").
    """

    latent: torch.Tensor
    latent_history: torch.Tensor
    step: int
    rgb: np.ndarray | dict[str, np.ndarray] | None


@dataclass(frozen=True)
class BatchedObservation:
    """Observation returned by step_batch().

    Attributes
    ----------
    latents: (K, H, C_latent, H_lat, W_lat) fp32, on env.device.
        K parallel rollouts of length H. Latents only — no leading start
        frame included (use the env's pre-call snapshot if you need it).
    rgbs: (K, H, 3, H_img, W_img) uint8 numpy.
        Decoded RGB frames for every (k, h) cell. Always populated because
        the planner's reward is RGB-based. Channel-first to match the
        normalizer's convention.
    """

    latents: torch.Tensor
    rgbs: np.ndarray
