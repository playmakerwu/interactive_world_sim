"""Verbatim ports of diffusion-forcing's MPPI primitives, used as the
oracle in tests/mppi/test_consistency_module.py.

Source: ``~/Documents/diffusion-forcing/algorithms/latent_dynamics/planner_v0_0.py``
Functions ported here:
  - ``Planner.sample_action_sequences_default`` (lines 196-256)
  - ``Planner.optimize_action_mppi``           (lines 390-400)
  - inline ``torch.clamp`` semantics from line 250

The reference's actual code reads its hyperparameters from ``self.config``
and uses the global ``torch.randn`` for noise. We accept hyperparameters
as explicit kwargs and accept an optional ``torch.Generator`` so the
tests can compare against a bit-deterministic noise source identical to
the one our planner uses.

The CONTROL FLOW of every line is preserved 1:1. The translation is:

  reference:  noise_sample = torch.normal(0, noise)
  here:       noise_sample = torch.randn(N, A, generator=gen) * noise_level

These produce the same scalar samples in the same shape under the same
RNG state — ``torch.normal(0, std)`` ≡ ``torch.randn(...) * std``.

A separate import-smoke test confirms the actual diffusion-forcing
``Planner`` class can be instantiated with our config schema; that check
guards against the reference repo evolving away from us.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reference_sample_action_sequences(
    act_seq: torch.Tensor,                  # (H, A)
    *,
    n_sample: int,
    beta_filter: float,
    noise_level: float,
    action_lower_lim: torch.Tensor,         # (A,)
    action_upper_lim: torch.Tensor,         # (A,)
    generator: torch.Generator | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Verbatim port of ``Planner.sample_action_sequences_default``.

    Returns ``(n_sample, H, A)``.
    """
    H, A = act_seq.shape
    if device is None:
        device = act_seq.device
    act_seq = act_seq.to(device)
    action_lower_lim = action_lower_lim.to(device)
    action_upper_lim = action_upper_lim.to(device)

    # [n_sample, H, A]
    act_seqs = torch.stack([act_seq.clone()] * n_sample)
    # [n_sample, A]
    act_residual = torch.zeros(
        (n_sample, A), dtype=act_seqs.dtype, device=device,
    )

    for i in range(H):
        # noise std is uniform across dims (the float-noise_level branch of
        # the reference's two-mode noise specification).
        if generator is None:
            noise_sample = torch.randn(
                (n_sample, A), device=device, dtype=act_seqs.dtype,
            ) * noise_level
        else:
            noise_sample = torch.randn(
                (n_sample, A), generator=generator,
                device=device, dtype=act_seqs.dtype,
            ) * noise_level

        act_residual = beta_filter * noise_sample + act_residual * (1.0 - beta_filter)
        act_seqs[:, i] += act_residual
        act_seqs[:, i] = torch.clamp(act_seqs[:, i], action_lower_lim, action_upper_lim)

    return act_seqs


def reference_softmax_weights(
    rewards: torch.Tensor,
    reward_weight: float,
) -> torch.Tensor:
    """Verbatim port of the softmax line in ``optimize_action_mppi``.

    Reference:
        ``softmax_weight = F.softmax(reward_seqs * self.reward_weight, dim=0)``

    Note: reference does NOT subtract the max before softmax. Our planner
    DOES subtract the max for numerical safety. The two are mathematically
    identical (``softmax(x*w) == softmax((x-x.max())*w)`` for any positive
    ``w``). The unstable-form helper here is used for the strict-bit-exact
    comparison; a separate test confirms the stabilised form produces
    identical output for well-conditioned inputs.
    """
    return F.softmax(rewards * reward_weight, dim=0)


def reference_optimize_action_mppi(
    act_seqs: torch.Tensor,                  # (N, H, A)
    rewards: torch.Tensor,                   # (N,)
    *,
    reward_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim port of ``Planner.optimize_action_mppi`` lines 393-400.

    Returns ``(act_seq (H, A), weights (N,))``.
    """
    softmax_weight = reference_softmax_weights(rewards, reward_weight)  # (N,)
    act_seq = torch.sum(
        act_seqs * softmax_weight.unsqueeze(-1).unsqueeze(-1),
        dim=0,
    )
    return act_seq, softmax_weight


def reference_clamp_actions(
    act_seqs: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Reference uses inline ``torch.clamp(act_seqs[:, i], lower, upper)``.

    For batched (N, H, A) tensors, this is equivalent to broadcasting the
    same per-dim bounds across the N and H axes.
    """
    return torch.clamp(act_seqs, lower, upper)
