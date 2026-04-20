"""Latent -> T-block-pose state probe.

MLP per design doc §1.7 (revised): [4096 -> 256 -> 128 -> 4], GELU,
LayerNorm on hidden layers, ~1.08M params. Differentiable so its
output can feed Dreamer's analytic policy gradient at RL time.

Output convention:
- Columns [0, 1]: position (cx, cy). Training targets are in [0, 1]
  (raw pixel / 128) so MSE gradient scale is unit-free; callers scale
  back to pixels via `pixels_from_norm(out)` for visualization / reward.
- Columns [2, 3]: (sin theta, cos theta). Unnormalised at training time
  — the soft unit-norm regulariser in the training loss pushes the
  network toward unit outputs, and post-hoc `normalize_sincos()` is
  applied at inference for numerical safety.

The module itself is JUST the MLP. Normalisation helpers live here as
free functions for callers to apply outside the forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_HIDDEN = (256, 128)


class StateProbe(nn.Module):
    """latent (B, 4, 32, 32) -> state (B, 4) raw output."""

    def __init__(
        self,
        in_ch: int = 4,
        in_hw: int = 32,
        hidden: tuple[int, ...] = DEFAULT_HIDDEN,
        out_dim: int = 4,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.in_hw = in_hw
        self.out_dim = out_dim

        in_dim = in_ch * in_hw * in_hw
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(h))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, C, H, W) float -> (B, 4) raw."""
        B = z.shape[0]
        x = z.reshape(B, -1)
        return self.net(x)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# Utilities (not part of the module — callers apply at train or inference time)
# -----------------------------------------------------------------------------

def pixels_from_norm(pos_norm: torch.Tensor, resolution: int = 128) -> torch.Tensor:
    """Scale normalised position outputs back to pixel coords. Pure op."""
    return pos_norm * resolution


def normalize_sincos(sincos: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Post-hoc L2 normalise (sin, cos) to the unit circle.

    Applied at inference only. During training we rely on the soft
    unit-norm regulariser (see rl/... training script).
    """
    norm = torch.clamp_min(torch.norm(sincos, dim=-1, keepdim=True), eps)
    return sincos / norm


def split_output(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split (B, 4) into (pos (B, 2), sincos (B, 2))."""
    return out[..., :2], out[..., 2:]
