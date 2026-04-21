"""Action samplers for MPPI.

The default `GaussianSampler` is what the v0 MPPI uses (zero-mean
isotropic noise at scale sigma). Step-5 follow-up showed this is
catastrophically out-of-distribution for the IWS world model — demo
actions have non-zero per-dim mean (≈ +0.17) and very smooth
step-to-step deltas (≈ 0.012 vs ~0.14 for σ=0.1 Gaussian), so the
WM's decoder hallucinates and the bimanual arms vanish from the
decoded scene over a handful of steps.

`DemoChunkSampler` fixes the distribution shape by drawing length-H
sequences directly from a real-action bank (the training dataset).
`DemoChunkJitterSampler` adds a small Gaussian perturbation on top
to give MPPI some search diversity around the demonstration manifold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
import torch


class ActionSampler(Protocol):
    """Stateless callable: (N, H, A) tensor of action sequences on `device`."""

    action_dim: int

    def sample(self, N: int, H: int, device: str | torch.device) -> torch.Tensor:
        ...


class GaussianSampler:
    """Independent zero-mean Gaussian per (sample, step, dim).

    This is the v0 sampler. Documented as catastrophically OOD for the
    IWS WM and kept here only for backward-comparable runs (v1/v2/v3).
    """

    def __init__(self, sigma: float, action_dim: int = 4) -> None:
        self.sigma = sigma
        self.action_dim = action_dim

    def sample(self, N: int, H: int, device: str | torch.device) -> torch.Tensor:
        return torch.randn(N, H, self.action_dim, device=device) * self.sigma

    def __repr__(self) -> str:
        return f"GaussianSampler(sigma={self.sigma}, action_dim={self.action_dim})"


def _load_action_bank(train_dir: Path, action_key: str = "action") -> np.ndarray:
    """Stack actions from every train episode into one (T_total, A) array."""
    eps = sorted(train_dir.glob("episode_*.hdf5"))
    if not eps:
        raise FileNotFoundError(f"no episode_*.hdf5 in {train_dir}")
    chunks = []
    for ep_path in eps:
        with h5py.File(ep_path, "r") as f:
            chunks.append(f[action_key][()])
    return np.concatenate(chunks, axis=0).astype(np.float32)


class DemoChunkSampler:
    """Sample length-H windows directly from a real-action bank.

    No perturbation, no rescaling. The action sequences MPPI scores are
    literal slices of demonstration trajectories. Guarantees in-distribution
    inputs for the WM, at the cost of zero exploration around the demo
    manifold.
    """

    def __init__(self, train_dir: Path | str, action_key: str = "action") -> None:
        self.bank = _load_action_bank(Path(train_dir), action_key=action_key)
        self.action_dim = self.bank.shape[1]

    def sample(self, N: int, H: int, device: str | torch.device) -> torch.Tensor:
        T = self.bank.shape[0]
        if H > T:
            raise ValueError(f"H={H} exceeds total demo length {T}")
        starts = np.random.randint(0, T - H + 1, size=N)
        chunks = np.stack([self.bank[s : s + H] for s in starts], axis=0)
        return torch.from_numpy(chunks).to(device)

    def __repr__(self) -> str:
        return (
            f"DemoChunkSampler(bank={self.bank.shape}, action_dim={self.action_dim})"
        )


class DemoChunkJitterSampler(DemoChunkSampler):
    """Demo chunks + small per-step Gaussian jitter.

    Trades a small step out of the training manifold for MPPI search
    diversity. `jitter_sigma` should be much smaller than the demo
    per-step delta L2 (≈ 0.012) to avoid dominating the demo signal —
    pick something like 0.005-0.02 for safe perturbation.
    """

    def __init__(
        self,
        train_dir: Path | str,
        jitter_sigma: float,
        action_key: str = "action",
    ) -> None:
        super().__init__(train_dir, action_key=action_key)
        self.jitter_sigma = jitter_sigma

    def sample(self, N: int, H: int, device: str | torch.device) -> torch.Tensor:
        chunks = super().sample(N, H, device)
        if self.jitter_sigma > 0.0:
            chunks = chunks + torch.randn_like(chunks) * self.jitter_sigma
        return chunks

    def __repr__(self) -> str:
        return (
            f"DemoChunkJitterSampler(bank={self.bank.shape}, "
            f"jitter_sigma={self.jitter_sigma})"
        )
