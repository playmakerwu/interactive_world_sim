"""EnvState — frozen dataclass capturing the full internal state of a
WorldModelEnv. snapshot() returns one of these; restore() consumes one.

Per the design, this carries exactly the two sliding windows plus a step
counter and the task name (for sanity-checking cross-task misuse).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EnvState:
    """Self-contained snapshot of a WorldModelEnv.

    All tensor fields are device-resident, contiguous, requires_grad=False.
    Both snapshot() and restore() deep-clone these on the way in/out so
    callers can mutate either side without aliasing the env's internals.

    Attributes
    ----------
    latent_window: (T_hist, C_latent, H_lat, W_lat) fp32 on `model.device`.
        The sliding window of past latents fed to `dynamics_forward` as
        conditioning context.
    action_window: (T_hist, action_dim) fp32 on `model.device`.
        The same-length sliding window of normalized actions in [-1, 1]
        that were associated with each latent step.
    step_counter: number of step() calls since the most recent reset().
    task: registry key the snapshot was produced from. restore() raises
        if the loaded env's task does not match.
    """

    latent_window: torch.Tensor
    action_window: torch.Tensor
    step_counter: int
    task: str
