"""Evaluate trained actor on the hardest initial states.

Usage (from repo root):
    conda run -n iws python rl/evaluate_hard.py [--checkpoint rl/outputs/checkpoints/final.pt]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

from rl.models.actor import Actor
from rl.models.world_model import DifferentiableDynamics
from rl.utils.config import DreamerConfig


def encode_frame(dynamics, raw_img, resolution, device):
    img = center_crop(raw_img, (resolution, resolution))
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    img_float = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
    return dynamics.encode(img_tensor)


@torch.no_grad()
def rollout_metrics(dynamics, z_init, action_fn, H, z_goal_flat, cfg):
    """Run rollout, return per-step cos sims and action norms."""
    device = cfg.device
    z = z_init.clone()
    action_hist = [torch.zeros(1, 1, cfg.action_dim, device=device)]
    cos_sims = [F.cosine_similarity(z[:, -1].reshape(1, -1), z_goal_flat).item()]
    action_norms = []

    for t in range(H):
        a = action_fn(z[:, -1])
        action_norms.append(a.norm().item())
        action_hist.append(a.unsqueeze(1))
        T_hist = z.shape[1]
        all_act = torch.cat(action_hist, dim=1)
        act_input = all_act[:, -(T_hist + 1):]
        z_next = dynamics.step(z, act_input, use_checkpoint=False)
        z = torch.cat([z, z_next], dim=1)[:, -cfg.hist_context:]
        cos_sims.append(
            F.cosine_similarity(z_next[:, 0].reshape(1, -1), z_goal_flat).item()
        )

    return cos_sims, action_norms, z  # return final z for decoding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="rl/outputs/checkpoints/final.pt")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--n_hard", type=int, default=5)
    args = parser.parse_args()

    cfg = DreamerConfig()
    device = cfg.device
    H = args.horizon

    out_dir = Path("rl/outputs/eval_hard")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path("rl/outputs/eval_hard_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("Loading world model …")
    dynamics = DifferentiableDynamics(cfg.ckpt_path, device=device)

    z_goal = torch.load(cfg.goal_path, map_location=device, weights_only=True).float()
    z_goal_flat = z_goal.reshape(1, -1)

    # decode goal for video
    goal_img = dynamics.decode(z_goal.squeeze(0).unsqueeze(0))
    goal_img_np = (goal_img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # load actor
    print(f"Loading actor from {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    actor = Actor(cfg.latent_dim, cfg.action_dim, cfg.actor_hidden, cfg.actor_layers)
    actor.load_state_dict(ckpt["actor"])
    actor = actor.to(device).float()
    actor.eval()

    # find hardest episodes across train+val
    all_eps = []
    for split, ddir in [("train", "data/mini/pusht/train"), ("val", "data/mini/pusht/val")]:
        for ep_path in sorted(Path(ddir).glob("episode_*.hdf5")):
            ep_idx = int(ep_path.stem.split("_")[-1])
            epi_data, _ = load_dict_from_hdf5(str(ep_path))
            raw_img = epi_data["obs"]["images"][cfg.obs_key][0]
            z_init = encode_frame(dynamics, raw_img, cfg.resolution, device)
            cs = F.cosine_similarity(z_init.reshape(1, -1), z_goal_flat).item()
            all_eps.append({
                "split": split, "ep_idx": ep_idx, "path": str(ep_path),
                "init_cos": cs, "raw_img_first": raw_img,
            })

    all_eps.sort(key=lambda e: e["init_cos"])
    hard_eps = all_eps[:args.n_hard]

    print(f"\n{args.n_hard} hardest episodes:")
    for i, ep in enumerate(hard_eps):
        print(f"  {i}: {ep['split']}/ep{ep['ep_idx']}  init_cos={ep['init_cos']:.4f}")

    # run rollouts
    policies = {
        "Trained": lambda z: actor.act_eval(z),
        "Random": lambda z: torch.randn(1, cfg.action_dim, device=device) * 0.1,
    }

    results = {pn: {"cos_sims": [], "action_norms": []} for pn in policies}

    for ep_i, ep in enumerate(hard_eps):
        z_init = encode_frame(
            dynamics, ep["raw_img_first"], cfg.resolution, device
        ).unsqueeze(1)  # (1, 1, C, H, W)

        for pol_name, action_fn in policies.items():
            print(f"  {pol_name}, hard_ep {ep_i} ({ep['split']}/ep{ep['ep_idx']}) …")
            cos_sims, action_norms, z_final_state = rollout_metrics(
                dynamics, z_init, action_fn, H, z_goal_flat, cfg,
            )
            results[pol_name]["cos_sims"].append(cos_sims)
            results[pol_name]["action_norms"].append(action_norms)

            # save video for trained actor only
            if pol_name == "Trained":
                # re-run to collect latents for decoding
                z = z_init.clone()
                action_hist_vid = [torch.zeros(1, 1, cfg.action_dim, device=device)]
                latents_vid = [z[:, -1]]
                with torch.no_grad():
                    for t in range(H):
                        a = action_fn(z[:, -1])
                        action_hist_vid.append(a.unsqueeze(1))
                        T_hist = z.shape[1]
                        all_act = torch.cat(action_hist_vid, dim=1)
                        act_input = all_act[:, -(T_hist + 1):]
                        z_next = dynamics.step(z, act_input, use_checkpoint=False)
                        z = torch.cat([z, z_next], dim=1)[:, -cfg.hist_context:]
                        latents_vid.append(z_next[:, 0])

                all_z = torch.stack(latents_vid).squeeze(1)
                # decode in batches to avoid OOM
                decoded_parts = []
                dec_batch = 10
                for di in range(0, all_z.shape[0], dec_batch):
                    decoded_parts.append(dynamics.decode(all_z[di:di+dec_batch]))
                decoded = torch.cat(decoded_parts, dim=0)
                frames = (decoded.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                # save video
                vid_path = out_dir / f"hard_ep{ep_i}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(vid_path), fourcc, 2, (cfg.resolution * 2, cfg.resolution))
                for frame in frames:
                    left = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    right = cv2.cvtColor(goal_img_np, cv2.COLOR_RGB2BGR)
                    writer.write(np.concatenate([left, right], axis=1))
                writer.release()

                # save first, last, and every 10th frame as PNGs
                for fi, frame in enumerate(frames):
                    if fi == 0 or fi == len(frames) - 1 or fi % 10 == 0:
                        png_path = frames_dir / f"hard_ep{ep_i}_frame{fi:03d}.png"
                        cv2.imwrite(str(png_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # print comparison table
    print(f"\n{'='*80}")
    print("HARD-START EVALUATION (H=50)")
    print(f"{'='*80}")
    print(f"{'Policy':<10} | {'Ep':>2} | {'Init CosSim':>11} | {'Final CosSim':>12} | "
          f"{'Best CosSim':>11} | {'Avg ||a||':>10}")
    print("-" * 80)
    for pol_name in policies:
        for ep_i in range(len(hard_eps)):
            cs = results[pol_name]["cos_sims"][ep_i]
            an = results[pol_name]["action_norms"][ep_i]
            print(f"{pol_name:<10} | {ep_i:>2} | {cs[0]:>11.4f} | {cs[-1]:>12.4f} | "
                  f"{max(cs):>11.4f} | {np.mean(an):>10.4f}")
        print("-" * 80)

    # averages
    print(f"\n{'AVERAGES':<10} | {'':>2} | {'Init CosSim':>11} | {'Final CosSim':>12} | "
          f"{'Best CosSim':>11} | {'Avg ||a||':>10}")
    print("-" * 80)
    for pol_name in policies:
        all_cs = results[pol_name]["cos_sims"]
        all_an = results[pol_name]["action_norms"]
        avg_init = np.mean([cs[0] for cs in all_cs])
        avg_final = np.mean([cs[-1] for cs in all_cs])
        avg_best = np.mean([max(cs) for cs in all_cs])
        avg_anorm = np.mean([np.mean(an) for an in all_an])
        print(f"{pol_name:<10} | {'--':>2} | {avg_init:>11.4f} | {avg_final:>12.4f} | "
              f"{avg_best:>11.4f} | {avg_anorm:>10.4f}")
    print(f"{'='*80}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"Trained": "tab:blue", "Random": "tab:gray"}

    ax = axes[0]
    ts = np.arange(H + 1)
    for pol_name in policies:
        cs = np.array(results[pol_name]["cos_sims"])
        mean = cs.mean(axis=0)
        std = cs.std(axis=0)
        ax.plot(ts, mean, label=pol_name, color=colors[pol_name])
        ax.fill_between(ts, mean - std, mean + std, alpha=0.2, color=colors[pol_name])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cosine Similarity to Goal")
    ax.set_title("Hard-Start Episodes: Goal Reaching")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ts_a = np.arange(H)
    for pol_name in policies:
        an = np.array(results[pol_name]["action_norms"])
        mean = an.mean(axis=0)
        std = an.std(axis=0)
        ax.plot(ts_a, mean, label=pol_name, color=colors[pol_name])
        ax.fill_between(ts_a, mean - std, mean + std, alpha=0.2, color=colors[pol_name])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Action L2 Norm")
    ax.set_title("Hard-Start Episodes: Action Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = Path("rl/outputs/hard_eval_comparison.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")

    # conclusion
    trained_cs = results["Trained"]["cos_sims"]
    random_cs = results["Random"]["cos_sims"]
    trained_final = np.mean([cs[-1] for cs in trained_cs])
    random_final = np.mean([cs[-1] for cs in random_cs])
    trained_init = np.mean([cs[0] for cs in trained_cs])
    trained_anorm = np.mean([np.mean(an) for an in results["Trained"]["action_norms"]])

    # check if cos_sim improves from init to final
    improvement = trained_final - trained_init

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print(f"Init avg cos_sim:         {trained_init:.4f}")
    print(f"Trained final avg cos_sim: {trained_final:.4f}  (Δ = {improvement:+.4f})")
    print(f"Random  final avg cos_sim: {random_final:.4f}")
    print(f"Trained avg action norm:   {trained_anorm:.4f}")
    print()
    if improvement > 0.001:
        print("The actor MOVES TOWARD the goal from hard starts.")
    elif improvement > -0.001:
        print("The actor MAINTAINS position — neither improving nor degrading.")
    else:
        print("The actor DRIFTS AWAY from the goal.")
    if trained_final > random_final:
        print("The trained actor outperforms random actions.")
    else:
        print("The trained actor does NOT outperform random actions.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
