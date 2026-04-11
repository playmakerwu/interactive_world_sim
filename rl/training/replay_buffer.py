"""Replay buffer that stores real trajectories as pre-encoded latent sequences."""

from pathlib import Path

import cv2
import numpy as np
import torch
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

from rl.models.world_model import DifferentiableDynamics


class LatentReplayBuffer:
    """Stores pre-encoded latent trajectories from the training dataset.

    On init, loads all episodes, encodes every frame to latent space,
    and caches the results.  Sampling returns random initial latents
    for imagination rollouts.
    """

    def __init__(
        self,
        dynamics: DifferentiableDynamics,
        dataset_dir: str,
        obs_key: str = "camera_1_color",
        resolution: int = 128,
        device: str = "cuda:0",
    ):
        self.device = device
        self.latent_seqs: list[torch.Tensor] = []  # each (T, C, H, W)
        self.action_seqs: list[torch.Tensor] = []  # each (T, A)

        ep_paths = sorted(Path(dataset_dir).glob("episode_*.hdf5"))
        print(f"Replay buffer: encoding {len(ep_paths)} episodes from {dataset_dir}")

        for ep_path in ep_paths:
            epi_data, _ = load_dict_from_hdf5(str(ep_path))
            images = epi_data["obs"]["images"][obs_key][()]  # (T, H, W, 3) uint8
            actions = epi_data["action"][()]  # (T, A) float32

            # normalize actions to [-1, 1] via the model's normalizer
            actions_norm = dynamics.normalizer["action"].normalize(
                torch.from_numpy(actions)
            ).float()

            # encode all frames in batches
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
                    batch_imgs.append(
                        torch.from_numpy(img).permute(2, 0, 1)
                    )
                batch_tensor = torch.stack(batch_imgs).to(device)
                z = dynamics.encode(batch_tensor)  # (batch, C, H, W)
                latents.append(z.cpu())

            self.latent_seqs.append(torch.cat(latents, dim=0))  # (T, C, H, W)
            self.action_seqs.append(actions_norm)  # (T, A)

        self.total_frames = sum(s.shape[0] for s in self.latent_seqs)
        print(f"Replay buffer: {len(self.latent_seqs)} episodes, "
              f"{self.total_frames} frames cached")

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample random initial latents.  Returns (B, 1, C, H, W)."""
        inits = []
        for _ in range(batch_size):
            ep_idx = np.random.randint(len(self.latent_seqs))
            t_idx = np.random.randint(self.latent_seqs[ep_idx].shape[0])
            inits.append(self.latent_seqs[ep_idx][t_idx])
        z = torch.stack(inits).unsqueeze(1).to(self.device)  # (B, 1, C, H, W)
        return z.float()
