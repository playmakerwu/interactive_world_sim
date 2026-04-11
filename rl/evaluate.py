"""Evaluate trained actor: imagination rollout → decode → save video.

Usage (from repo root):
    conda run -n iws python rl/evaluate.py [--checkpoint rl/outputs/checkpoints/final.pt]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

from rl.models.actor import Actor
from rl.models.world_model import DifferentiableDynamics
from rl.utils.config import DreamerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str,
        default="rl/outputs/checkpoints/final.pt",
    )
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    cfg = DreamerConfig()
    device = cfg.device
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load world model
    print("Loading world model …")
    dynamics = DifferentiableDynamics(cfg.ckpt_path, device=device)

    # load goal
    z_goal = torch.load(cfg.goal_path, map_location=device, weights_only=True).float()
    z_goal_flat = z_goal.reshape(1, -1)

    # decode goal to image for side-by-side display
    goal_img = dynamics.decode(z_goal.squeeze(0).unsqueeze(0))  # (1, 3, 128, 128)
    goal_img_np = (
        goal_img[0].permute(1, 2, 0).cpu().numpy() * 255
    ).clip(0, 255).astype(np.uint8)

    # load actor
    print(f"Loading actor from {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    actor = Actor(cfg.latent_dim, cfg.action_dim, cfg.actor_hidden, cfg.actor_layers)
    actor.load_state_dict(ckpt["actor"])
    actor = actor.to(device).float()
    actor.eval()

    # load val episodes
    val_dir = Path("data/mini/pusht/val")
    ep_paths = sorted(val_dir.glob("episode_*.hdf5"))[: args.episodes]

    for ep_i, ep_path in enumerate(ep_paths):
        epi_data, _ = load_dict_from_hdf5(str(ep_path))
        raw_img = epi_data["obs"]["images"][cfg.obs_key][0]
        img = center_crop(raw_img, (cfg.resolution, cfg.resolution))
        img = cv2.resize(
            img, (cfg.resolution, cfg.resolution), interpolation=cv2.INTER_AREA
        )
        img_float = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
        z_init = dynamics.encode(img_tensor).unsqueeze(1)  # (1, 1, C, H, W)

        # imagination rollout
        z = z_init
        action_hist = [torch.zeros(1, 1, cfg.action_dim, device=device)]
        latents = [z[:, -1]]

        with torch.no_grad():
            for t in range(args.horizon):
                a = actor.act_eval(z[:, -1])
                action_hist.append(a.unsqueeze(1))
                T_hist = z.shape[1]
                all_act = torch.cat(action_hist, dim=1)
                act_input = all_act[:, -(T_hist + 1):]
                z_next = dynamics.step(z, act_input, use_checkpoint=False)
                z = torch.cat([z, z_next], dim=1)[:, -cfg.hist_context:]
                latents.append(z_next[:, 0])

        # decode all latents
        all_z = torch.stack(latents, dim=0).squeeze(1)  # (H+1, C, Hl, Wl)
        decoded = dynamics.decode(all_z)  # (H+1, 3, 128, 128)
        frames = (
            decoded.permute(0, 2, 3, 1).cpu().numpy() * 255
        ).clip(0, 255).astype(np.uint8)

        # cosine similarity to goal
        final_z = latents[-1].reshape(1, -1)
        cos_sim = F.cosine_similarity(final_z, z_goal_flat).item()
        print(f"Episode {ep_i}: final cos_sim = {cos_sim:.4f}")

        # write video: imagined (left) | goal (right)
        vid_path = out_dir / f"eval_episode_{ep_i}.mp4"
        h, w = cfg.resolution, cfg.resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid_path), fourcc, 2, (w * 2, h))
        for frame in frames:
            left = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            right = cv2.cvtColor(goal_img_np, cv2.COLOR_RGB2BGR)
            side = np.concatenate([left, right], axis=1)
            writer.write(side)
        writer.release()
        print(f"  → saved {vid_path}")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
