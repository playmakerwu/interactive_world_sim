"""Evaluate discrete actor on the hardest initial states."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from collections import Counter

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5

from rl.models.actor_discrete import DiscreteActor
from rl.models.world_model import DifferentiableDynamics
from rl.utils.config import DreamerConfig


def encode_frame(dynamics, raw_img, resolution, device):
    img = center_crop(raw_img, (resolution, resolution))
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    img_float = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
    return dynamics.encode(img_tensor)


@torch.no_grad()
def rollout_discrete(
    dynamics, z_init, action_fn, H, z_goal_flat, cfg,
    collect_latents: bool = False,
):
    device = cfg.device
    z = z_init.clone()
    action_hist = [torch.zeros(1, 1, cfg.action_dim, device=device)]
    cos_sims = [F.cosine_similarity(z[:, -1].reshape(1, -1), z_goal_flat).item()]
    action_idxs = []
    latents = [z[:, -1]] if collect_latents else None

    for t in range(H):
        a, idx = action_fn(z[:, -1])
        action_idxs.append(int(idx.item()))
        action_hist.append(a.unsqueeze(1))
        T_hist = z.shape[1]
        all_act = torch.cat(action_hist, dim=1)
        act_input = all_act[:, -(T_hist + 1):]
        z_next = dynamics.step(z, act_input, use_checkpoint=False)
        z = torch.cat([z, z_next], dim=1)[:, -cfg.hist_context:]
        cos_sims.append(
            F.cosine_similarity(z_next[:, 0].reshape(1, -1), z_goal_flat).item()
        )
        if collect_latents:
            latents.append(z_next[:, 0])

    return cos_sims, action_idxs, latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="rl/outputs/checkpoints/final.pt")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--n_hard", type=int, default=5)
    args = parser.parse_args()

    cfg = DreamerConfig()
    device = cfg.device
    H = args.horizon

    out_dir = Path("rl/outputs/eval_discrete")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path("rl/outputs/eval_discrete_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("Loading world model …")
    dynamics = DifferentiableDynamics(cfg.ckpt_path, device=device)

    z_goal = torch.load(cfg.goal_path, map_location=device, weights_only=True).float()
    z_goal_flat = z_goal.reshape(1, -1)
    goal_img = dynamics.decode(z_goal.squeeze(0).unsqueeze(0))
    goal_img_np = (goal_img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    print(f"Loading actor from {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    actor = DiscreteActor(
        latent_dim=cfg.latent_dim,
        hidden=cfg.actor_hidden,
        n_layers=cfg.actor_layers,
        action_table_path=cfg.action_table_path,
    )
    actor.load_state_dict(ckpt["actor"])
    actor = actor.to(device).float()
    actor.eval()

    # find hardest
    all_eps = []
    for split, ddir in [("train", "data/mini/pusht/train"), ("val", "data/mini/pusht/val")]:
        for ep_path in sorted(Path(ddir).glob("episode_*.hdf5")):
            ep_idx = int(ep_path.stem.split("_")[-1])
            epi_data, _ = load_dict_from_hdf5(str(ep_path))
            raw_img = epi_data["obs"]["images"][cfg.obs_key][0]
            z_init = encode_frame(dynamics, raw_img, cfg.resolution, device)
            cs = F.cosine_similarity(z_init.reshape(1, -1), z_goal_flat).item()
            all_eps.append({
                "split": split, "ep_idx": ep_idx,
                "init_cos": cs, "raw_img_first": raw_img,
            })
    all_eps.sort(key=lambda e: e["init_cos"])
    hard_eps = all_eps[:args.n_hard]

    print(f"\n{args.n_hard} hardest episodes:")
    for i, ep in enumerate(hard_eps):
        print(f"  {i}: {ep['split']}/ep{ep['ep_idx']}  init_cos={ep['init_cos']:.4f}")

    def trained_fn(z):
        return actor.act_eval(z)

    def random_fn(z):
        idx = torch.randint(0, actor.n_actions, (z.shape[0],), device=device)
        return actor.action_table[idx], idx

    policies = {"Trained": trained_fn, "Random": random_fn}
    results = {pn: {"cos_sims": [], "action_idxs": []} for pn in policies}

    for ep_i, ep in enumerate(hard_eps):
        z_init = encode_frame(
            dynamics, ep["raw_img_first"], cfg.resolution, device,
        ).unsqueeze(1)

        for pol_name, action_fn in policies.items():
            print(f"  {pol_name}, hard_ep {ep_i} ({ep['split']}/ep{ep['ep_idx']}) …")
            collect = (pol_name == "Trained")
            cos_sims, idxs, latents = rollout_discrete(
                dynamics, z_init, action_fn, H, z_goal_flat, cfg,
                collect_latents=collect,
            )
            results[pol_name]["cos_sims"].append(cos_sims)
            results[pol_name]["action_idxs"].append(idxs)

            if collect:
                all_z = torch.stack(latents).squeeze(1)
                decoded_parts = []
                for di in range(0, all_z.shape[0], 10):
                    decoded_parts.append(dynamics.decode(all_z[di:di+10]))
                decoded = torch.cat(decoded_parts, dim=0)
                frames = (decoded.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                # video
                vid_path = out_dir / f"hard_ep{ep_i}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(vid_path), fourcc, 2, (cfg.resolution * 2, cfg.resolution))
                for frame in frames:
                    left = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    right = cv2.cvtColor(goal_img_np, cv2.COLOR_RGB2BGR)
                    writer.write(np.concatenate([left, right], axis=1))
                writer.release()

                # PNGs: first, last, every 10th
                for fi, frame in enumerate(frames):
                    if fi == 0 or fi == len(frames) - 1 or fi % 10 == 0:
                        png_path = frames_dir / f"hard_ep{ep_i}_frame{fi:03d}.png"
                        cv2.imwrite(str(png_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # ── print table ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"DISCRETE EVAL (H={H})")
    print(f"{'='*80}")
    print(f"{'Policy':<10} | {'Ep':>2} | {'Init CS':>8} | {'Final CS':>9} | {'Best CS':>8}")
    print("-" * 80)
    for pol_name in policies:
        for ep_i in range(len(hard_eps)):
            cs = results[pol_name]["cos_sims"][ep_i]
            print(f"{pol_name:<10} | {ep_i:>2} | {cs[0]:>8.4f} | {cs[-1]:>9.4f} | {max(cs):>8.4f}")
        print("-" * 80)
    print(f"\n{'AVERAGES':<10} | {'':>2} | {'Init CS':>8} | {'Final CS':>9} | {'Best CS':>8}")
    print("-" * 80)
    for pol_name in policies:
        all_cs = results[pol_name]["cos_sims"]
        avg_init = np.mean([cs[0] for cs in all_cs])
        avg_final = np.mean([cs[-1] for cs in all_cs])
        avg_best = np.mean([max(cs) for cs in all_cs])
        print(f"{pol_name:<10} | {'--':>2} | {avg_init:>8.4f} | {avg_final:>9.4f} | {avg_best:>8.4f}")
    print(f"{'='*80}")

    # action distribution (trained)
    all_idxs = [i for ep_list in results["Trained"]["action_idxs"] for i in ep_list]
    counts = Counter(all_idxs)
    total = sum(counts.values())
    print("\nTrained actor action distribution:")
    for i, name in enumerate(actor.action_names):
        pct = 100 * counts.get(i, 0) / total
        print(f"  {name:<22}: {pct:5.1f}%")

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Trained": "tab:blue", "Random": "tab:gray"}
    ts = np.arange(H + 1)
    for pn in policies:
        cs = np.array(results[pn]["cos_sims"])
        mean = cs.mean(axis=0)
        std = cs.std(axis=0)
        ax.plot(ts, mean, label=pn, color=colors[pn])
        ax.fill_between(ts, mean - std, mean + std, alpha=0.2, color=colors[pn])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cosine Similarity to Goal")
    ax.set_title("Discrete Actor on Hard-Start Episodes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = Path("rl/outputs/discrete_eval_comparison.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")

    # conclusion
    trained_final = np.mean([cs[-1] for cs in results["Trained"]["cos_sims"]])
    random_final = np.mean([cs[-1] for cs in results["Random"]["cos_sims"]])
    trained_init = np.mean([cs[0] for cs in results["Trained"]["cos_sims"]])

    # action preference — how peaked is the distribution?
    top_pct = max(counts.values()) / total * 100
    uniform_pct = 100 / actor.n_actions
    preferential = top_pct > 2 * uniform_pct  # twice uniform ≥ meaningful

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print(f"Trained final avg cos_sim: {trained_final:.4f}  (init {trained_init:.4f}, Δ {trained_final-trained_init:+.4f})")
    print(f"Random  final avg cos_sim: {random_final:.4f}")
    print(f"Trained top action: {actor.action_names[max(counts, key=counts.get)]} ({top_pct:.1f}%)")
    print(f"Uniform would be:   {uniform_pct:.1f}%")
    print(f"Meaningful action preference: {'YES' if preferential else 'NO'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
