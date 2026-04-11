import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 4096,
        action_dim: int = 4,
        hidden: int = 512,
        n_layers: int = 3,
    ):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            in_dim = hidden
        layers += [nn.Linear(hidden, action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def act(self, z: torch.Tensor) -> torch.Tensor:
        """Differentiable action from latent.  z: (B, C, H, W) or (B, D)."""
        if z.ndim == 4:
            z = z.reshape(z.shape[0], -1)
        return self.net(z)  # (B, A) in [-1, 1]

    @torch.no_grad()
    def act_eval(self, z: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation."""
        return self.act(z)
