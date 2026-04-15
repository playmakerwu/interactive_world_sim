"""Compare trained policy vs random baseline on goal reaching.

Usage (from repo root):
    conda run -n iws python rl/compare_policies.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


@torch.no_grad()
def rollout(dynamics, z_init, action_fn, H, hist_context, action_dim, device):
    """Run imagination rollout, return per-step metrics.

    action_fn(z_curr) -> (action_dim,) tensor
    Returns dict with keys: cos_sims, l2_dists, action_norms  (each length H+1 or H)
    """
    z = z_init.clone()  # (1, 1, C, H, W)
    action_hist = [torch.zeros(1, 1, action_dim, device=device)]
    latents = [z[:, -1]]  # z_0
    actions = []

    for t in range(H):
        a = action_fn(z[:, -1])  # (1, A)
        actions.append(a)
        action_hist.append(a.unsqueeze(1))
        T_hist = z.shape[1]
        all_act = torch.cat(action_hist, dim=1)
        act_input = all_act[:, -(T_hist + 1) :]
        z_next = dynamics.step(z, act_input, use_checkpoint=False)
        z = torch.cat([z, z_next], dim=1)[:, -hist_context:]
        latents.append(z_next[:, 0])

    return latents, actions


def compute_metrics(latents, actions, z_goal_flat):
    """Compute per-step cosine similarity, L2 distance, action norms."""
    cos_sims = []
    l2_dists = []
    for z in latents:
        z_flat = z.reshape(1, -1)
        cos_sims.append(F.cosine_similarity(z_flat, z_goal_flat).item())
        l2_dists.append((z_flat - z_goal_flat).norm().item())

    action_norms = [a.norm().item() for a in actions]
    return cos_sims, l2_dists, action_norms


def main():
    cfg = DreamerConfig()
    device = cfg.device
    H = 50  # longer horizon for comparison

    # load world model
    print("Loading world model …")
    dynamics = DifferentiableDynamics(cfg.ckpt_path, device=device)

    # load goal
    z_goal = torch.load(cfg.goal_path, map_location=device, weights_only=True).float()
    z_goal_flat = z_goal.reshape(1, -1)

    # load penalized actor
    ckpt_path = "rl/outputs/checkpoints/final.pt"
    print(f"Loading penalized actor from {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    actor_pen = Actor(cfg.latent_dim, cfg.action_dim, cfg.actor_hidden, cfg.actor_layers)
    actor_pen.load_state_dict(ckpt["actor"])
    actor_pen = actor_pen.to(device).float()
    actor_pen.eval()

    # define action functions
    def penalized_action_fn(z_curr):
        return actor_pen.act_eval(z_curr)

    def random_action_fn(z_curr):
        return torch.randn(1, cfg.action_dim, device=device) * 0.1  # moderate random

    # load val episodes
    val_dir = Path("data/mini/pusht/val")
    ep_paths = sorted(val_dir.glob("episode_*.hdf5"))[:5]

    policies = {
        "Penalized": penalized_action_fn,
        "Random": random_action_fn,
    }

    all_results = {}
    for pol_name in policies:
        all_results[pol_name] = {
            "cos_sims": [],    # list of lists (per episode)
            "l2_dists": [],
            "action_norms": [],
            "init_cos": [],
            "final_cos": [],
            "best_cos": [],
            "avg_action_norm": [],
        }

    # run rollouts
    for ep_i, ep_path in enumerate(ep_paths):
        epi_data, _ = load_dict_from_hdf5(str(ep_path))
        raw_img = epi_data["obs"]["images"][cfg.obs_key][0]
        img = center_crop(raw_img, (cfg.resolution, cfg.resolution))
        img = cv2.resize(img, (cfg.resolution, cfg.resolution), interpolation=cv2.INTER_AREA)
        img_float = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
        z_init = dynamics.encode(img_tensor).unsqueeze(1)  # (1, 1, C, H, W)

        for pol_name, action_fn in policies.items():
            print(f"  {pol_name} policy, episode {ep_i} …")
            latents, actions = rollout(
                dynamics, z_init, action_fn, H,
                cfg.hist_context, cfg.action_dim, device,
            )
            cos_sims, l2_dists, action_norms = compute_metrics(
                latents, actions, z_goal_flat,
            )

            r = all_results[pol_name]
            r["cos_sims"].append(cos_sims)
            r["l2_dists"].append(l2_dists)
            r["action_norms"].append(action_norms)
            r["init_cos"].append(cos_sims[0])
            r["final_cos"].append(cos_sims[-1])
            r["best_cos"].append(max(cos_sims))
            r["avg_action_norm"].append(np.mean(action_norms))

    # print comparison table
    print("\n" + "=" * 90)
    print("POLICY COMPARISON: Goal Reaching (H=50 steps)")
    print("=" * 90)
    print(f"{'Policy':<12} | {'Ep':>2} | {'Init CosSim':>11} | {'Final CosSim':>12} | "
          f"{'Best CosSim':>11} | {'Avg ||a||':>10}")
    print("-" * 90)
    for pol_name in policies:
        r = all_results[pol_name]
        for ep_i in range(len(ep_paths)):
            print(f"{pol_name:<12} | {ep_i:>2} | {r['init_cos'][ep_i]:>11.4f} | "
                  f"{r['final_cos'][ep_i]:>12.4f} | {r['best_cos'][ep_i]:>11.4f} | "
                  f"{r['avg_action_norm'][ep_i]:>10.4f}")
        print("-" * 90)

    # print averages
    print(f"\n{'AVERAGES':<12} | {'':>2} | {'Init CosSim':>11} | {'Final CosSim':>12} | "
          f"{'Best CosSim':>11} | {'Avg ||a||':>10}")
    print("-" * 90)
    for pol_name in policies:
        r = all_results[pol_name]
        print(f"{pol_name:<12} | {'--':>2} | {np.mean(r['init_cos']):>11.4f} | "
              f"{np.mean(r['final_cos']):>12.4f} | {np.mean(r['best_cos']):>11.4f} | "
              f"{np.mean(r['avg_action_norm']):>10.4f}")
    print("=" * 90)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"Penalized": "tab:blue", "Random": "tab:gray"}

    # left: cosine similarity over time
    ax = axes[0]
    timesteps = np.arange(H + 1)
    for pol_name in policies:
        cs = np.array(all_results[pol_name]["cos_sims"])  # (N_ep, H+1)
        mean = cs.mean(axis=0)
        std = cs.std(axis=0)
        ax.plot(timesteps, mean, label=pol_name, color=colors[pol_name])
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=colors[pol_name])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cosine Similarity to Goal")
    ax.set_title("Goal Reaching: Cosine Similarity Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # right: action norm over time
    ax = axes[1]
    timesteps_a = np.arange(H)
    for pol_name in policies:
        an = np.array(all_results[pol_name]["action_norms"])  # (N_ep, H)
        mean = an.mean(axis=0)
        std = an.std(axis=0)
        ax.plot(timesteps_a, mean, label=pol_name, color=colors[pol_name])
        ax.fill_between(timesteps_a, mean - std, mean + std, alpha=0.2, color=colors[pol_name])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Action L2 Norm")
    ax.set_title("Action Magnitude Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = Path("rl/outputs/policy_comparison.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")

    # save raw data
    json_data = {}
    for pol_name in policies:
        r = all_results[pol_name]
        json_data[pol_name] = {
            "cos_sims": [list(map(float, cs)) for cs in r["cos_sims"]],
            "l2_dists": [list(map(float, ld)) for ld in r["l2_dists"]],
            "action_norms": [list(map(float, an)) for an in r["action_norms"]],
            "init_cos": list(map(float, r["init_cos"])),
            "final_cos": list(map(float, r["final_cos"])),
            "best_cos": list(map(float, r["best_cos"])),
            "avg_action_norm": list(map(float, r["avg_action_norm"])),
        }
    json_path = Path("rl/outputs/policy_comparison.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Raw data saved to {json_path}")

    # conclusion
    pen = all_results["Penalized"]
    rand = all_results["Random"]
    pen_final = np.mean(pen["final_cos"])
    rand_final = np.mean(rand["final_cos"])
    pen_best = np.mean(pen["best_cos"])
    rand_best = np.mean(rand["best_cos"])
    pen_anorm = np.mean(pen["avg_action_norm"])
    rand_anorm = np.mean(rand["avg_action_norm"])

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"Penalized policy vs Random baseline:")
    print(f"  Final cos_sim:  {pen_final:.4f} vs {rand_final:.4f}  "
          f"({'better' if pen_final > rand_final else 'worse'} by {abs(pen_final - rand_final):.4f})")
    print(f"  Best cos_sim:   {pen_best:.4f} vs {rand_best:.4f}  "
          f"({'better' if pen_best > rand_best else 'worse'} by {abs(pen_best - rand_best):.4f})")
    print(f"  Avg action norm: {pen_anorm:.4f} vs {rand_anorm:.4f}  "
          f"({pen_anorm/rand_anorm:.1f}x)")
    print()
    if pen_final >= rand_final:
        print("The penalized policy reaches the goal AS WELL OR BETTER than random,")
        print("while using controlled, moderate actions. The action penalty successfully")
        print("regularizes the policy without sacrificing goal-reaching performance.")
    else:
        print("The penalized policy reaches the goal WORSE than random.")
        print("The penalty coefficient may need tuning.")
    print("=" * 70)


if __name__ == "__main__":
    main()
