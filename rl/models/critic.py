import copy

import torch
import torch.nn as nn


class Critic(nn.Module):
    """Twin critic (min of two value heads) with EMA target support."""

    def __init__(
        self,
        latent_dim: int = 4096,
        hidden: int = 512,
        n_layers: int = 3,
    ):
        super().__init__()
        self.q1 = self._build_head(latent_dim, hidden, n_layers)
        self.q2 = self._build_head(latent_dim, hidden, n_layers)

    @staticmethod
    def _build_head(in_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(hidden, 1))
        return nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return min of twin critics.  z: (B, C, H, W) or (B, D) → (B,)."""
        if z.ndim == 4:
            z = z.reshape(z.shape[0], -1)
        v1 = self.q1(z).squeeze(-1)
        v2 = self.q2(z).squeeze(-1)
        return torch.min(v1, v2)

    def both(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both critic values.  z: (B, ...) → (B,), (B,)."""
        if z.ndim == 4:
            z = z.reshape(z.shape[0], -1)
        return self.q1(z).squeeze(-1), self.q2(z).squeeze(-1)


def make_target_critic(critic: Critic) -> Critic:
    target = copy.deepcopy(critic)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


@torch.no_grad()
def soft_update(target: Critic, source: Critic, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.lerp_(sp.data, tau)
