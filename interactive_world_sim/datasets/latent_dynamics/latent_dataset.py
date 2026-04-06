"""Dataset that loads pre-encoded latent tensors (.pt) for Stage 2 training.

Eliminates encoder forward pass and RGB I/O during training.
Supports trajectory-level bootstrapping via bootstrap_seed.
"""

import copy
import glob
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_range_normalizer_from_stat,
)
from interactive_world_sim.utils.replay_buffer import ReplayBuffer
from interactive_world_sim.utils.sampler import SequenceSampler

from .base_dataset import BaseImageDataset


class LatentReplayBuffer:
    """Minimal replay buffer backed by pre-encoded .pt episode files."""

    def __init__(self, split_dir: str):
        metadata = torch.load(
            os.path.join(split_dir, "metadata.pt"), weights_only=False
        )
        self.episode_ends = metadata["episode_ends"]  # np.int64 array
        self.n_episodes = metadata["n_episodes"]
        self.obs_keys = metadata["obs_keys"]

        # Load all episodes into contiguous arrays
        episode_paths = sorted(
            glob.glob(os.path.join(split_dir, "episode_*.pt")),
            key=lambda p: int(Path(p).stem.split("_")[-1]),
        )
        assert len(episode_paths) == self.n_episodes, (
            f"Expected {self.n_episodes} episodes, found {len(episode_paths)}"
        )

        latent_list = []
        action_list = []
        for ep_path in episode_paths:
            ep = torch.load(ep_path, weights_only=False)
            latent_list.append(ep["latent"])  # (T, C, H, W)
            action_list.append(ep["action"])  # (T, A)

        self._data = {
            "latent": torch.cat(latent_list, dim=0).numpy(),  # (N_total, C, H, W)
            "action": torch.cat(action_list, dim=0).numpy(),  # (N_total, A)
        }

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data


class LatentDataset(BaseImageDataset):
    """Dataset loading pre-encoded latent .pt files for Stage 2 dynamics training.

    Config requires:
        dataset_dir: root dir containing train/ and val/ with .pt files
        horizon, skip_frame, pad_before, pad_after: temporal window params
        bootstrap_seed: (optional) seed for trajectory-level bootstrap resampling
        action_mode: action mode string (for normalizer compatibility)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        dataset_dir = cfg.dataset_dir
        horizon = cfg.horizon * cfg.skip_frame
        self.val_horizon = (
            cfg.val_horizon * cfg.skip_frame if "val_horizon" in cfg else horizon
        )
        self.skip_idx = cfg.skip_idx if "skip_idx" in cfg else 1
        self.skip_frame = cfg.skip_frame
        self.goal_sample = cfg.goal_sample if "goal_sample" in cfg else "intermediate"
        self.dataset_dir = dataset_dir
        self.pad_before = cfg.pad_before
        self.pad_after = cfg.pad_after
        self.action_mode = cfg.action_mode

        # Bootstrap seed for ensemble diversity
        bootstrap_seed = cfg.bootstrap_seed if "bootstrap_seed" in cfg else None

        # Load pre-encoded replay buffer
        train_dir = os.path.join(dataset_dir, "train")
        self.replay_buffer = LatentReplayBuffer(train_dir)

        # Build episode mask (integer counts for bootstrap)
        n_eps = self.replay_buffer.n_episodes
        if bootstrap_seed is not None:
            rng = np.random.default_rng(seed=int(bootstrap_seed))
            bootstrap_indices = rng.choice(n_eps, size=n_eps, replace=True)
            train_mask = np.zeros(n_eps, dtype=np.int64)
            for idx in bootstrap_indices:
                train_mask[idx] += 1
        else:
            train_mask = np.ones(n_eps, dtype=np.int64)

        all_keys = list(self.replay_buffer.keys())

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=cfg.pad_before,
            pad_after=cfg.pad_after,
            episode_mask=train_mask,
            goal_sample=self.goal_sample,
            keys=all_keys,
            skip_frame=cfg.skip_frame,
            keys_to_keep_intermediate=["action"],
        )
        self.train_mask = train_mask

    def get_normalizer(self, mode: str = "none", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        # Action normalizer
        stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = get_range_normalizer_from_stat(stat)
        return normalizer

    def __len__(self) -> int:
        if self.is_val:
            return self.replay_buffer.n_episodes // self.skip_idx
        return len(self.sampler)

    def get_validation_dataset(self) -> "LatentDataset":
        val_set = copy.copy(self)
        val_set.is_val = True
        val_dir = os.path.join(self.dataset_dir, "val")
        val_set.replay_buffer = LatentReplayBuffer(val_dir)
        val_mask = np.ones(val_set.replay_buffer.n_episodes, dtype=np.int64)
        val_set.sampler = SequenceSampler(
            replay_buffer=val_set.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
            skip_idx=self.skip_idx,
            goal_sample=self.goal_sample,
            skip_frame=self.skip_frame,
            keys_to_keep_intermediate=["action"],
        )
        val_set.train_mask = val_mask
        return val_set

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_val:
            epi_idx = idx * self.skip_idx
            epi_start = (
                self.replay_buffer.episode_ends[epi_idx - 1] if epi_idx > 0 else 0
            )
            epi_end = self.replay_buffer.episode_ends[epi_idx]
            val_horizon = self.val_horizon
            seq_end = min(epi_end, epi_start + val_horizon)
            sample = dict()
            for key in self.sampler.keys:
                sample[key] = self.replay_buffer[key][epi_start:seq_end]
                if sample[key].shape[0] < val_horizon:
                    pad_len = val_horizon - sample[key].shape[0]
                    pad_shape = (pad_len, *np.ones_like(sample[key].shape[1:]).tolist())
                    sample_pad = np.tile(sample[key][-1:], pad_shape)
                    sample[key] = np.concatenate([sample[key], sample_pad], axis=0)
                if key in self.sampler.keys_to_keep_intermediate:
                    inter_frames = sample[key].shape[0] // self.skip_frame
                    sample_shape = list(sample[key].shape[1:])
                    sample_shape[0] = sample_shape[0] * self.skip_frame
                    sample[key] = sample[key].reshape(
                        inter_frames, self.skip_frame, *sample[key].shape[1:]
                    )
                    sample[key] = sample[key].reshape(-1, *sample_shape)
                else:
                    sample[key] = sample[key][:: self.skip_frame]
        else:
            sample = self.sampler.sample_sequence(idx)

        # Convert to tensors — output format matches Stage 2 expectations
        latent = torch.from_numpy(sample["latent"].astype(np.float32))  # (T, C, H, W)
        action = torch.from_numpy(sample["action"].astype(np.float32))  # (T, A)

        return {
            "latent": latent,
            "action": action,
        }
