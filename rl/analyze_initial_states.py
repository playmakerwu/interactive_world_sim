"""Analyze initial state distances to goal across all episodes.

Usage (from repo root):
    conda run -n iws python rl/analyze_initial_states.py
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

from rl.models.world_model import DifferentiableDynamics
from rl.utils.config import DreamerConfig


def encode_frame(dynamics, raw_img, resolution, device):
    img = center_crop(raw_img, (resolution, resolution))
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    img_float = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
    return dynamics.encode(img_tensor)  # (1, C, H, W)


def main():
    cfg = DreamerConfig()
    device = cfg.device

    print("Loading world model …")
    dynamics = DifferentiableDynamics(cfg.ckpt_path, device=device)

    z_goal = torch.load(cfg.goal_path, map_location=device, weights_only=True).float()
    z_goal_flat = z_goal.reshape(1, -1)

    # scan all episodes
    data_dirs = [
        ("train", Path("data/mini/pusht/train")),
        ("val", Path("data/mini/pusht/val")),
    ]

    rows = []
    for split, ddir in data_dirs:
        ep_paths = sorted(ddir.glob("episode_*.hdf5"))
        for ep_path in ep_paths:
            ep_idx = int(ep_path.stem.split("_")[-1])
            epi_data, _ = load_dict_from_hdf5(str(ep_path))
            images = epi_data["obs"]["images"][cfg.obs_key][()]
            T = images.shape[0]

            with torch.no_grad():
                z_init = encode_frame(dynamics, images[0], cfg.resolution, device)
                z_final = encode_frame(dynamics, images[-1], cfg.resolution, device)

            init_cos = F.cosine_similarity(
                z_init.reshape(1, -1), z_goal_flat
            ).item()
            final_cos = F.cosine_similarity(
                z_final.reshape(1, -1), z_goal_flat
            ).item()
            init_l2 = (z_init.reshape(1, -1) - z_goal_flat).norm().item()
            final_l2 = (z_final.reshape(1, -1) - z_goal_flat).norm().item()

            rows.append({
                "split": split,
                "episode": ep_idx,
                "ep_path": str(ep_path),
                "length": T,
                "init_cos": init_cos,
                "final_cos": final_cos,
                "init_l2": init_l2,
                "final_l2": final_l2,
            })

    # sort ascending by init cosine sim (hardest first)
    rows.sort(key=lambda r: r["init_cos"])

    # print table
    print(f"\n{'Rank':>4} | {'Split':>5} | {'Ep':>3} | {'Len':>4} | "
          f"{'Init CosSim':>11} | {'Final CosSim':>12} | "
          f"{'Init L2':>8} | {'Final L2':>8}")
    print("-" * 80)
    for i, r in enumerate(rows):
        print(f"{i:4d} | {r['split']:>5} | {r['episode']:>3} | {r['length']:>4} | "
              f"{r['init_cos']:>11.4f} | {r['final_cos']:>12.4f} | "
              f"{r['init_l2']:>8.2f} | {r['final_l2']:>8.2f}")

    # thresholds
    init_cosines = [r["init_cos"] for r in rows]
    print(f"\n--- Distribution ---")
    print(f"Total episodes: {len(rows)}")
    print(f"Init cos_sim range: [{min(init_cosines):.4f}, {max(init_cosines):.4f}]")
    print(f"Init cos_sim mean:  {np.mean(init_cosines):.4f}")
    print(f"Init cos_sim std:   {np.std(init_cosines):.4f}")
    for thresh in [0.95, 0.90, 0.85, 0.80]:
        count = sum(1 for c in init_cosines if c < thresh)
        print(f"  Episodes with init cos_sim < {thresh}: {count}/{len(rows)}")

    # save hard init frames
    hard_dir = Path("rl/outputs/hard_init_frames")
    hard_dir.mkdir(parents=True, exist_ok=True)
    n_hard = min(10, len(rows))
    print(f"\nSaving first frame of {n_hard} hardest episodes …")
    for i in range(n_hard):
        r = rows[i]
        epi_data, _ = load_dict_from_hdf5(r["ep_path"])
        raw_img = epi_data["obs"]["images"][cfg.obs_key][0]
        img = center_crop(raw_img, (cfg.resolution, cfg.resolution))
        img = cv2.resize(img, (cfg.resolution, cfg.resolution), interpolation=cv2.INTER_AREA)
        out_path = hard_dir / f"rank{i}_{r['split']}_ep{r['episode']}_cos{r['init_cos']:.4f}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  {out_path.name}")

    # histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(init_cosines, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(np.median(init_cosines), color="red", linestyle="--",
               label=f"Median = {np.median(init_cosines):.4f}")
    ax.set_xlabel("Initial Cosine Similarity to Goal")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Init CosSim ({len(rows)} episodes)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = Path("rl/outputs/init_state_distribution.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nHistogram saved to {plot_path}")

    # save JSON
    json_data = {
        "episodes": [{k: v for k, v in r.items()} for r in rows],
        "summary": {
            "total": len(rows),
            "min_init_cos": min(init_cosines),
            "max_init_cos": max(init_cosines),
            "mean_init_cos": float(np.mean(init_cosines)),
            "median_init_cos": float(np.median(init_cosines)),
        },
    }
    json_path = Path("rl/outputs/init_state_analysis.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Analysis saved to {json_path}")


if __name__ == "__main__":
    main()
