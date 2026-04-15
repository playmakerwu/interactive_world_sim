"""Replay buffer that stores real trajectories as pre-encoded latent sequences."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

from rl.models.world_model import DifferentiableDynamics


class LatentReplayBuffer:
    """Stores pre-encoded latent trajectories from the training dataset.

    Supports multiple sampling modes:
      - "uniform": sample any frame from any episode
      - "hard": sample frames whose cosine sim to goal is below median
      - "curriculum": gradually shift from easy to hard over training
    """

    def __init__(
        self,
        dynamics: DifferentiableDynamics,
        dataset_dir: str,
        obs_key: str = "camera_1_color",
        resolution: int = 128,
        device: str = "cuda:0",
        z_goal: torch.Tensor | None = None,
        sampling_mode: str = "uniform",
        hard_threshold_percentile: float = 50.0,
    ):
        self.device = device
        self.sampling_mode = sampling_mode
        self.hard_threshold_percentile = hard_threshold_percentile
        self.latent_seqs: list[torch.Tensor] = []  # each (T, C, H, W)
        self.action_seqs: list[torch.Tensor] = []  # each (T, A)

        ep_paths = sorted(Path(dataset_dir).glob("episode_*.hdf5"))
        print(f"Replay buffer: encoding {len(ep_paths)} episodes from {dataset_dir}")

        for ep_path in ep_paths:
            epi_data, _ = load_dict_from_hdf5(str(ep_path))
            images = epi_data["obs"]["images"][obs_key][()]  # (T, H, W, 3) uint8
            actions = epi_data["action"][()]  # (T, A) float32

            actions_norm = dynamics.normalizer["action"].normalize(
                torch.from_numpy(actions)
            ).float()

            T = images.shape[0]
            latents = []
            batch_sz = 32
            for i in range(0, T, batch_sz):
                batch_imgs = []
                for j in range(i, min(i + batch_sz, T)):
                    img = center_crop(images[j], (resolution, resolution))
                    img = cv2.resize(
                        img, (resolution, resolution),
                        interpolation=cv2.INTER_AREA,
                    )
                    img = img.astype(np.float32) / 255.0
                    batch_imgs.append(torch.from_numpy(img).permute(2, 0, 1))
                batch_tensor = torch.stack(batch_imgs).to(device)
                z = dynamics.encode(batch_tensor)
                latents.append(z.cpu())

            self.latent_seqs.append(torch.cat(latents, dim=0))
            self.action_seqs.append(actions_norm)

        self.total_frames = sum(s.shape[0] for s in self.latent_seqs)
        print(f"Replay buffer: {len(self.latent_seqs)} episodes, "
              f"{self.total_frames} frames cached")

        # precompute per-frame cosine similarity to goal for hard/curriculum modes
        self._all_frames: list[torch.Tensor] = []  # flat list of all frames
        self._all_cos_sims: list[float] = []
        if z_goal is not None:
            z_goal_flat = z_goal.reshape(1, -1).cpu()
            for seq in self.latent_seqs:
                for t in range(seq.shape[0]):
                    self._all_frames.append(seq[t])
                    cs = F.cosine_similarity(
                        seq[t].reshape(1, -1), z_goal_flat
                    ).item()
                    self._all_cos_sims.append(cs)
            self._all_cos_sims_np = np.array(self._all_cos_sims)

            # compute threshold for hard sampling
            self._hard_threshold = float(np.percentile(
                self._all_cos_sims_np, hard_threshold_percentile
            ))
            self._hard_indices = np.where(
                self._all_cos_sims_np <= self._hard_threshold
            )[0]
            self._easy_indices = np.where(
                self._all_cos_sims_np > self._hard_threshold
            )[0]

            print(f"Replay buffer: cosine sim range "
                  f"[{self._all_cos_sims_np.min():.4f}, {self._all_cos_sims_np.max():.4f}], "
                  f"mean={self._all_cos_sims_np.mean():.4f}")
            print(f"  Hard threshold (p{hard_threshold_percentile:.0f}): {self._hard_threshold:.4f}")
            print(f"  Hard frames: {len(self._hard_indices)}/{len(self._all_frames)}")
        else:
            self._all_cos_sims_np = None

        self._train_step = 0

    def set_train_step(self, step: int):
        self._train_step = step

    def get_stats(self) -> dict:
        pool = self._get_active_indices()
        cos_sims = self._all_cos_sims_np[pool] if self._all_cos_sims_np is not None else []
        return {
            "n_frames": len(pool),
            "n_total": len(self._all_frames) if self._all_frames else self.total_frames,
            "min_cos": float(cos_sims.min()) if len(cos_sims) else 0,
            "max_cos": float(cos_sims.max()) if len(cos_sims) else 0,
            "mean_cos": float(cos_sims.mean()) if len(cos_sims) else 0,
            "mode": self.sampling_mode,
        }

    def _get_active_indices(self) -> np.ndarray:
        if self.sampling_mode == "uniform" or self._all_cos_sims_np is None:
            return np.arange(len(self._all_frames)) if self._all_frames else np.arange(self.total_frames)
        elif self.sampling_mode == "hard":
            return self._hard_indices if len(self._hard_indices) > 0 else np.arange(len(self._all_frames))
        elif self.sampling_mode == "curriculum":
            # linear schedule: step 0 → easy only, step 2500 → all, step 5000 → hard only
            if self._train_step < 2500:
                frac = self._train_step / 2500.0  # 0 → 1
                # start with easy, gradually mix in all
                n_use = int(len(self._easy_indices) + frac * len(self._hard_indices))
                # sort by cos_sim descending, take top n_use
                sorted_idx = np.argsort(-self._all_cos_sims_np)
                return sorted_idx[:max(1, n_use)]
            else:
                frac = (self._train_step - 2500) / 2500.0  # 0 → 1
                # gradually restrict to hard only
                n_all = len(self._all_frames)
                n_use = int(n_all - frac * len(self._easy_indices))
                sorted_idx = np.argsort(self._all_cos_sims_np)
                return sorted_idx[:max(1, n_use)]
        return np.arange(len(self._all_frames))

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample initial latents according to current mode."""
        if not self._all_frames:
            # fallback to old behavior if no goal was provided
            inits = []
            for _ in range(batch_size):
                ep_idx = np.random.randint(len(self.latent_seqs))
                t_idx = np.random.randint(self.latent_seqs[ep_idx].shape[0])
                inits.append(self.latent_seqs[ep_idx][t_idx])
            return torch.stack(inits).unsqueeze(1).to(self.device).float()

        pool = self._get_active_indices()
        chosen = np.random.choice(pool, size=batch_size, replace=True)
        inits = [self._all_frames[i] for i in chosen]
        return torch.stack(inits).unsqueeze(1).to(self.device).float()
