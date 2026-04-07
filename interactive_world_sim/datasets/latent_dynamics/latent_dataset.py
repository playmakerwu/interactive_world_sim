"""Dataset that loads pre-encoded latent tensors (.pt) for Stage 2 training.

ARCHITECTURE: Strict Lazy Loading
==================================
__init__  → only loads metadata.pt (~1 KB) and stores file paths (strings)
__getitem__ → loads ONE episode .pt from disk on demand, slices the window

This eliminates the catastrophic memory explosion caused by:
1. Eagerly loading all .pt files into a giant numpy array in __init__
2. torch.cat() creating peak 3x memory during concatenation
3. Linux fork() duplicating the numpy arrays across num_workers processes

Memory footprint per process: ~O(1) regardless of dataset size.
"""

import copy
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_range_normalizer_from_stat,
)

from .base_dataset import BaseImageDataset


class LatentDataset(BaseImageDataset):
    """Lazy-loading dataset for pre-encoded latent .pt files.

    Memory-safe: only metadata and file paths are held in memory.
    Each __getitem__ call loads a single episode from disk and slices it.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        dataset_dir = cfg.dataset_dir
        horizon = cfg.horizon * cfg.skip_frame
        self.horizon = horizon
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

        bootstrap_seed = cfg.bootstrap_seed if "bootstrap_seed" in cfg else None

        # --- LAZY LOADING: only read metadata, never load episode data ---
        train_dir = os.path.join(dataset_dir, "train")
        self._split_dir = train_dir
        metadata = torch.load(
            os.path.join(train_dir, "metadata.pt"), weights_only=False
        )
        self._episode_ends = metadata["episode_ends"]  # np.int64, cumulative
        self._n_episodes = metadata["n_episodes"]
        self._obs_keys = metadata["obs_keys"]

        # Store sorted file paths (strings only — no data loaded)
        self._episode_paths: List[str] = sorted(
            glob.glob(os.path.join(train_dir, "episode_*.pt")),
            key=lambda p: int(Path(p).stem.split("_")[-1]),
        )
        assert len(self._episode_paths) == self._n_episodes

        # Compute per-episode lengths from cumulative ends
        self._episode_lengths = np.diff(
            np.concatenate([[0], self._episode_ends])
        ).astype(np.int64)

        # Build bootstrap mask
        if bootstrap_seed is not None:
            rng = np.random.default_rng(seed=int(bootstrap_seed))
            bootstrap_indices = rng.choice(
                self._n_episodes, size=self._n_episodes, replace=True
            )
            self.train_mask = np.zeros(self._n_episodes, dtype=np.int64)
            for idx in bootstrap_indices:
                self.train_mask[idx] += 1
        else:
            self.train_mask = np.ones(self._n_episodes, dtype=np.int64)

        # Build sample index: list of (episode_idx, frame_offset) tuples
        # This replaces SequenceSampler — lightweight, no data references
        self._sample_indices = self._build_sample_indices(
            self._episode_lengths, horizon, cfg.pad_before, cfg.pad_after,
            self.train_mask,
        )

        # Cache action stats for normalizer (load once, tiny memory)
        self._action_stats = self._compute_action_stats()

    @staticmethod
    def _build_sample_indices(
        episode_lengths: np.ndarray,
        sequence_length: int,
        pad_before: int,
        pad_after: int,
        episode_mask: np.ndarray,
    ) -> np.ndarray:
        """Build (episode_idx, start_offset) pairs for all valid windows."""
        pad_before = min(max(pad_before, 0), sequence_length - 1)
        pad_after = min(max(pad_after, 0), sequence_length - 1)

        indices = []
        for ep_idx, ep_len in enumerate(episode_lengths):
            repeat_count = int(episode_mask[ep_idx])
            if repeat_count <= 0:
                continue
            min_start = -pad_before
            max_start = ep_len - sequence_length + pad_after
            for _rep in range(repeat_count):
                for offset in range(min_start, max_start + 1):
                    indices.append((ep_idx, offset))

        return np.array(indices, dtype=np.int64) if indices else np.zeros((0, 2), dtype=np.int64)

    def _compute_action_stats(self) -> dict:
        """Load all actions once to compute normalizer stats, then discard."""
        action_chunks = []
        for ep_path in self._episode_paths:
            ep = torch.load(ep_path, weights_only=False)
            action_chunks.append(ep["action"].numpy())
        all_actions = np.concatenate(action_chunks, axis=0)
        stats = array_to_stats(all_actions)
        # Discard the data — only stats (a few floats) are kept
        return stats

    def _load_episode(self, ep_idx: int) -> Dict[str, np.ndarray]:
        """Load a single episode from disk. Called per __getitem__."""
        ep = torch.load(self._episode_paths[ep_idx], weights_only=False)
        return {
            "latent": ep["latent"].numpy(),  # (T, C, H, W)
            "action": ep["action"].numpy(),  # (T, A)
        }

    def get_normalizer(self, mode: str = "none", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = get_range_normalizer_from_stat(self._action_stats)
        return normalizer

    def __len__(self) -> int:
        if self.is_val:
            return self._n_episodes // self.skip_idx
        return len(self._sample_indices)

    def get_validation_dataset(self) -> "LatentDataset":
        val_set = copy.copy(self)
        val_set.is_val = True

        val_dir = os.path.join(self.dataset_dir, "val")
        val_meta = torch.load(
            os.path.join(val_dir, "metadata.pt"), weights_only=False
        )
        val_set._split_dir = val_dir
        val_set._episode_ends = val_meta["episode_ends"]
        val_set._n_episodes = val_meta["n_episodes"]
        val_set._episode_paths = sorted(
            glob.glob(os.path.join(val_dir, "episode_*.pt")),
            key=lambda p: int(Path(p).stem.split("_")[-1]),
        )
        val_set._episode_lengths = np.diff(
            np.concatenate([[0], val_set._episode_ends])
        ).astype(np.int64)
        val_set.train_mask = np.ones(val_set._n_episodes, dtype=np.int64)
        val_set._sample_indices = self._build_sample_indices(
            val_set._episode_lengths, self.val_horizon,
            self.pad_before, self.pad_after, val_set.train_mask,
        )
        return val_set

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_val:
            return self._getitem_val(idx)
        return self._getitem_train(idx)

    def _getitem_train(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start_offset = self._sample_indices[idx]
        ep_data = self._load_episode(ep_idx)
        ep_len = self._episode_lengths[ep_idx]

        # Compute buffer/sample boundaries (mirrors create_indices logic)
        buffer_start = max(start_offset, 0)
        buffer_end = min(start_offset + self.horizon, ep_len)
        sample_start = buffer_start - start_offset
        sample_end = self.horizon - ((start_offset + self.horizon) - buffer_end)

        result = {}
        for key in ["latent", "action"]:
            data = ep_data[key][buffer_start:buffer_end]

            # Pad if needed
            if sample_start > 0 or sample_end < self.horizon:
                padded = np.zeros(
                    (self.horizon,) + data.shape[1:], dtype=data.dtype
                )
                if sample_start > 0:
                    padded[:sample_start] = data[0]
                if sample_end < self.horizon:
                    padded[sample_end:] = data[-1]
                padded[sample_start:sample_end] = data
                data = padded

            # Apply skip_frame
            if key == "action":
                # Keep intermediate frames for action
                inter_frames = self.horizon // self.skip_frame
                data_shape = list(data.shape[1:])
                data_shape[0] = data_shape[0] * self.skip_frame
                data = data.reshape(
                    inter_frames, self.skip_frame, *data.shape[1:]
                )
                data = data.reshape(-1, *data_shape)
            else:
                data = data[:: self.skip_frame]

            result[key] = torch.from_numpy(data.astype(np.float32))

        return result

    def _getitem_val(self, idx: int) -> Dict[str, torch.Tensor]:
        epi_idx = idx * self.skip_idx
        ep_data = self._load_episode(epi_idx)
        val_horizon = self.val_horizon
        ep_len = self._episode_lengths[epi_idx]
        seq_len = min(ep_len, val_horizon)

        result = {}
        for key in ["latent", "action"]:
            data = ep_data[key][:seq_len]

            # Pad if needed
            if data.shape[0] < val_horizon:
                pad_len = val_horizon - data.shape[0]
                pad_shape = (pad_len, *np.ones_like(data.shape[1:]).tolist())
                data = np.concatenate(
                    [data, np.tile(data[-1:], pad_shape)], axis=0
                )

            # Apply skip_frame
            if key == "action":
                inter_frames = data.shape[0] // self.skip_frame
                data_shape = list(data.shape[1:])
                data_shape[0] = data_shape[0] * self.skip_frame
                data = data.reshape(
                    inter_frames, self.skip_frame, *data.shape[1:]
                )
                data = data.reshape(-1, *data_shape)
            else:
                data = data[:: self.skip_frame]

            result[key] = torch.from_numpy(data.astype(np.float32))

        return result
