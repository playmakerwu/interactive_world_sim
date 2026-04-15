"""Discrete actor: categorical distribution over keyboard-style actions.

The actor outputs logits over N discrete action choices. During imagination
(training) we use Gumbel-Softmax straight-through estimator to keep the
action selection differentiable so gradients flow through the dynamics model.
During evaluation we take argmax.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteActor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 4096,
        hidden: int = 512,
        n_layers: int = 3,
        action_table_path: str = "rl/discrete_action_space.json",
    ):
        super().__init__()
        # load action table
        with open(action_table_path) as f:
            spec = json.load(f)
        self.action_names = spec["action_names"]
        self.n_actions = len(self.action_names)
        self.action_dim = spec["action_dim"]

        table = torch.tensor(
            [spec["actions"][name] for name in self.action_names],
            dtype=torch.float32,
        )  # (N, action_dim)
        self.register_buffer("action_table", table)

        # MLP
        layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, self.n_actions))
        self.net = nn.Sequential(*layers)

    def _latent_flat(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 4:
            z = z.reshape(z.shape[0], -1)
        return z

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self._latent_flat(z))  # (B, N)

    def act(
        self, z: torch.Tensor, temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a discrete action via Gumbel-Softmax straight-through.

        Returns:
            action_vec: (B, action_dim) — exact one of the discrete vectors,
                        but with gradients that flow through the soft distribution
            logits:     (B, N)
            one_hot:    (B, N) hard one-hot used for selection
        """
        logits = self.logits(z)
        # straight-through: hard=True returns one-hot in forward but keeps soft
        # gradients during backward.
        one_hot = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        action_vec = one_hot @ self.action_table  # (B, action_dim)
        return action_vec, logits, one_hot

    @torch.no_grad()
    def act_eval(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy argmax action (no sampling).  Returns (action_vec, idx)."""
        logits = self.logits(z)
        idx = logits.argmax(dim=-1)  # (B,)
        action_vec = self.action_table[idx]
        return action_vec, idx

    def get_action_table(self) -> torch.Tensor:
        return self.action_table
