"""Batched rollout helper — thin convenience wrapper over DifferentiableDynamics.rollout."""

from __future__ import annotations

import torch


def batched_rollout(
    z0: torch.Tensor,
    actions: torch.Tensor,
    wm,
    hist_context: int = 10,
) -> torch.Tensor:
    """Roll N action sequences of horizon H through the world model.

    Args:
        z0: (B, C, H_lat, W_lat) initial latent (no time dim). If a
            (C, H_lat, W_lat) latent for a single start state is given,
            it is expanded to B copies — the caller is expected to
            provide the already-batched form.
        actions: (B, H, A) action sequences.
        wm: a DifferentiableDynamics instance (or any object that
            exposes a `.rollout(z_init=(B, 1, C, H, W), actions=(B, H, A),
            hist_context=int) -> (B, H+1, C, H, W)` method).
        hist_context: how many past latents the dynamics model conditions
            on at each step. The default 10 matches the checkpoint's
            `n_frames=10` configuration.

    Returns:
        latents: (B, H+1, C, H_lat, W_lat) — z_0 through z_H.
    """
    if z0.dim() != 4:
        raise ValueError(
            f"z0 must be (B, C, H, W) with 4 dims, got shape {tuple(z0.shape)}"
        )
    if actions.dim() != 3:
        raise ValueError(
            f"actions must be (B, H, A) with 3 dims, got shape {tuple(actions.shape)}"
        )
    if z0.shape[0] != actions.shape[0]:
        raise ValueError(
            f"batch dim mismatch: z0 has B={z0.shape[0]}, "
            f"actions has B={actions.shape[0]}"
        )

    z_init = z0.unsqueeze(1)  # (B, 1, C, H, W)
    return wm.rollout(z_init, actions, hist_context=hist_context)
