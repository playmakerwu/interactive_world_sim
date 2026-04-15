"""Evaluate trained actor: rollout, decode, save video + metrics.

Usage (from repo root):
    conda run -n iws python rl/evaluate.py [--checkpoint ...] [--horizon N] [--episodes N]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

from rl.models.actor import Actor
from rl.models.actor_discrete import DiscreteActor
from rl.models.world_model import DifferentiableDynamics
from rl.utils.config import DreamerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str,
        default="rl/outputs/checkpoints/final.pt",
    )
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--val_dir", type=str, default="data/mini/pusht/val")
    args = parser.parse_args()

    cfg = DreamerConfig()
    device = cfg.device
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading world model …")
    dynamics = DifferentiableDynamics(cfg.ckpt_path, device=device)

    z_goal = torch.load(cfg.goal_path, map_location=device, weights_only=True).float()
    z_goal_flat = z_goal.reshape(1, -1)

    goal_img = dynamics.decode(z_goal.squeeze(0).unsqueeze(0))
    goal_img_np = (
        goal_img[0].permute(1, 2, 0).cpu().numpy() * 255
    ).clip(0, 255).astype(np.uint8)

    print(f"Loading actor from {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    is_discrete = ckpt.get("discrete", False)
    if is_discrete:
        actor = DiscreteActor(
            latent_dim=cfg.latent_dim,
            hidden=cfg.actor_hidden,
            n_layers=cfg.actor_layers,
            action_table_path=cfg.action_table_path,
        )
    else:
        actor = Actor(
            cfg.latent_dim, cfg.action_dim, cfg.actor_hidden, cfg.actor_layers,
        )
    actor.load_state_dict(ckpt["actor"])
    actor = actor.to(device).float()
    actor.eval()
    print(f"  actor type: {'discrete' if is_discrete else 'continuous'}")

    val_paths = sorted(Path(args.val_dir).glob("episode_*.hdf5"))[: args.episodes]
    if not val_paths:
        raise FileNotFoundError(f"No val episodes in {args.val_dir}")

    per_ep = []  # dicts with init/final/best cos_sim
    for ep_i, ep_path in enumerate(val_paths):
        epi_data, _ = load_dict_from_hdf5(str(ep_path))
        raw_img = epi_data["obs"]["images"][cfg.obs_key][0]
        img = center_crop(raw_img, (cfg.resolution, cfg.resolution))
        img = cv2.resize(
            img, (cfg.resolution, cfg.resolution), interpolation=cv2.INTER_AREA,
        )
        img_float = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
        z_init = dynamics.encode(img_tensor).unsqueeze(1)  # (1, 1, C, H, W)

        z = z_init
        action_hist = [torch.zeros(1, 1, cfg.action_dim, device=device)]
        latents = [z[:, -1]]
        cos_sims = [F.cosine_similarity(z[:, -1].reshape(1, -1), z_goal_flat).item()]

        with torch.no_grad():
            for t in range(args.horizon):
                if is_discrete:
                    a, _ = actor.act_eval(z[:, -1])
                else:
                    a = actor.act_eval(z[:, -1])
                action_hist.append(a.unsqueeze(1))
                T_hist = z.shape[1]
                all_act = torch.cat(action_hist, dim=1)
                act_input = all_act[:, -(T_hist + 1):]
                z_next = dynamics.step(z, act_input, use_checkpoint=False)
                z = torch.cat([z, z_next], dim=1)[:, -cfg.hist_context:]
                latents.append(z_next[:, 0])
                cos_sims.append(
                    F.cosine_similarity(
                        z_next[:, 0].reshape(1, -1), z_goal_flat,
                    ).item()
                )

        all_z = torch.stack(latents, dim=0).squeeze(1)
        # decode in batches to avoid OOM at long horizons
        parts = []
        for i in range(0, all_z.shape[0], 10):
            parts.append(dynamics.decode(all_z[i : i + 10]))
        decoded = torch.cat(parts, dim=0)
        frames = (
            decoded.permute(0, 2, 3, 1).cpu().numpy() * 255
        ).clip(0, 255).astype(np.uint8)

        init_cs, final_cs = cos_sims[0], cos_sims[-1]
        best_cs = float(max(cos_sims))
        per_ep.append({
            "episode": ep_path.name,
            "init_cos_sim": init_cs,
            "final_cos_sim": final_cs,
            "best_cos_sim": best_cs,
            "delta": final_cs - init_cs,
        })
        print(
            f"Episode {ep_i} ({ep_path.name}): init={init_cs:.4f} "
            f"final={final_cs:.4f} best={best_cs:.4f} Δ={final_cs - init_cs:+.4f}"
        )

        vid_path = out_dir / f"eval_episode_{ep_i}.mp4"
        h, w = cfg.resolution, cfg.resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid_path), fourcc, 4, (w * 2, h))
        for frame in frames:
            left = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            right = cv2.cvtColor(goal_img_np, cv2.COLOR_RGB2BGR)
            writer.write(np.concatenate([left, right], axis=1))
        writer.release()
        print(f"  → saved {vid_path}")

    # aggregate metrics
    avg_init = float(np.mean([e["init_cos_sim"] for e in per_ep]))
    avg_final = float(np.mean([e["final_cos_sim"] for e in per_ep]))
    avg_best = float(np.mean([e["best_cos_sim"] for e in per_ep]))
    summary = {
        "checkpoint": args.checkpoint,
        "horizon": args.horizon,
        "n_episodes": len(per_ep),
        "avg_init_cos_sim": avg_init,
        "avg_final_cos_sim": avg_final,
        "avg_best_cos_sim": avg_best,
        "avg_delta": avg_final - avg_init,
        "per_episode": per_ep,
    }
    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(f"Horizon {args.horizon}, {len(per_ep)} episodes")
    print(f"  avg init : {avg_init:.4f}")
    print(f"  avg final: {avg_final:.4f}")
    print(f"  avg best : {avg_best:.4f}")
    print(f"  avg Δ    : {avg_final - avg_init:+.4f}")
    print(f"Metrics written to {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
