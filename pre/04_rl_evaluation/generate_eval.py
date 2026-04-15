"""RL evaluation with the trained (discrete) actor.

Renders 3 rollouts, saves videos + first/last frames + a cosine-trajectory
plot + a reward-vs-visual figure that makes the key argument.

Usage (from repo root):
    conda run -n iws python pre/04_rl_evaluation/generate_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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

CKPT = "rl/outputs/checkpoints/step_1000.pt"
OUT_DIR = Path("pre/04_rl_evaluation")
HORIZON = 50
N_EPISODES = 3
plt.rcParams.update({"font.size": 14})


def encode_frame(dynamics, raw_img, res, device):
    img = center_crop(raw_img, (res, res))
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
    img_float = img.astype(np.float32) / 255.0
    t = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
    return dynamics.encode(t), img


@torch.no_grad()
def rollout(dynamics, actor, z_init, action_dim, hist_context, H, z_goal_flat):
    device = z_init.device
    z = z_init.clone()
    action_hist = [torch.zeros(1, 1, action_dim, device=device)]
    latents = [z[:, -1]]
    cos_sims = [F.cosine_similarity(z[:, -1].reshape(1, -1), z_goal_flat).item()]
    for _ in range(H):
        a, _ = actor.act_eval(z[:, -1])
        action_hist.append(a.unsqueeze(1))
        T_hist = z.shape[1]
        all_act = torch.cat(action_hist, dim=1)
        act_input = all_act[:, -(T_hist + 1) :]
        z_next = dynamics.step(z, act_input, use_checkpoint=False)
        z = torch.cat([z, z_next], dim=1)[:, -hist_context:]
        latents.append(z_next[:, 0])
        cos_sims.append(
            F.cosine_similarity(z_next[:, 0].reshape(1, -1), z_goal_flat).item()
        )
    return latents, cos_sims


def decode_batched(dynamics, all_z, batch: int = 10):
    parts = []
    for i in range(0, all_z.shape[0], batch):
        parts.append(dynamics.decode(all_z[i : i + batch]))
    return torch.cat(parts, dim=0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = DreamerConfig()
    device = cfg.device

    print("Loading world model …")
    dynamics = DifferentiableDynamics(cfg.ckpt_path, device=device)

    z_goal = torch.load(cfg.goal_path, map_location=device, weights_only=True).float()
    z_goal_flat = z_goal.reshape(1, -1)

    goal_img_t = dynamics.decode(z_goal.squeeze(0).unsqueeze(0))
    goal_img_np = (goal_img_t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(
        str(OUT_DIR / "goal_decoded.png"),
        cv2.cvtColor(goal_img_np, cv2.COLOR_RGB2BGR),
    )

    print(f"Loading actor from {CKPT} …")
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    actor = DiscreteActor(
        latent_dim=cfg.latent_dim, hidden=cfg.actor_hidden,
        n_layers=cfg.actor_layers, action_table_path=cfg.action_table_path,
    )
    actor.load_state_dict(ckpt["actor"])
    actor = actor.to(device).float()
    actor.eval()

    val_dir = Path("data/mini/pusht/val")
    ep_paths = sorted(val_dir.glob("episode_*.hdf5"))[:N_EPISODES]

    all_cos = []
    rollout_for_plot = None  # keep one rollout's frames + cos_sims for Part 6

    for i, ep_path in enumerate(ep_paths):
        print(f"Rolling out episode {i} ({ep_path.name}) …")
        epi, _ = load_dict_from_hdf5(str(ep_path))
        raw = epi["obs"]["images"][cfg.obs_key][0][()]
        z_init_2d, first_img = encode_frame(dynamics, raw, cfg.resolution, device)
        z_init = z_init_2d.unsqueeze(1)

        latents, cos_sims = rollout(
            dynamics, actor, z_init,
            cfg.action_dim, cfg.hist_context, HORIZON, z_goal_flat,
        )
        all_cos.append(cos_sims)

        # decode all latents to images
        all_z = torch.stack(latents).squeeze(1)
        decoded = decode_batched(dynamics, all_z, batch=10)
        frames = (decoded.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # save first / last frame PNGs
        first_path = OUT_DIR / f"episode_{i}_first.png"
        last_path = OUT_DIR / f"episode_{i}_last.png"
        cv2.imwrite(str(first_path), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(last_path), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))

        # video
        vid_path = OUT_DIR / f"episode_{i}_rollout.mp4"
        H = W = cfg.resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid_path), fourcc, 2, (W * 2, H))
        for frame in frames:
            left = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            right = cv2.cvtColor(goal_img_np, cv2.COLOR_RGB2BGR)
            writer.write(np.concatenate([left, right], axis=1))
        writer.release()

        if rollout_for_plot is None:
            rollout_for_plot = (frames, cos_sims, i)

    # ── cosine trajectory plot ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ts = np.arange(HORIZON + 1)
    for i, cs in enumerate(all_cos):
        ax.plot(ts, cs, label=f"episode {i}  ({cs[0]:.4f} → {cs[-1]:.4f})",
                linewidth=2)
        ax.scatter([0], [cs[0]], s=60, zorder=5)
        ax.scatter([ts[-1]], [cs[-1]], s=60, zorder=5)
    ax.set_xlabel("timestep")
    ax.set_ylabel("cosine similarity to goal")
    ax.set_title("Rollout trajectory — cosine similarity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    avg_init = np.mean([cs[0] for cs in all_cos])
    avg_final = np.mean([cs[-1] for cs in all_cos])
    ax.text(
        0.02, 0.04,
        f"mean init = {avg_init:.4f}\n"
        f"mean final = {avg_final:.4f}\n"
        f"Δ = {avg_final - avg_init:+.4f}  (change <1%)",
        transform=ax.transAxes, fontsize=11,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cosine_trajectory.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'cosine_trajectory.png'}")

    # ── reward-vs-visual figure: the key argument ───────────────────
    frames, cs, ep_i = rollout_for_plot
    snap_ts = [0, 15, 35, min(49, HORIZON)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7),
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, t in enumerate(snap_ts):
        axes[0, col].imshow(frames[t])
        axes[0, col].set_title(f"t = {t}", fontsize=13)
        axes[0, col].axis("off")
        axes[1, col].barh([0], [cs[t]], color="tab:green", height=0.5)
        axes[1, col].set_xlim(0.95, 1.00)
        axes[1, col].set_yticks([])
        axes[1, col].axvline(cs[0], color="k", linestyle="--", alpha=0.4,
                             label="init")
        axes[1, col].set_xlabel(f"cos_sim = {cs[t]:.4f}", fontsize=11)
        axes[1, col].set_title("")

    fig.suptitle(
        "Cosine similarity changes <1% despite visible robot movement —\n"
        "reward is dominated by robot-arm position, not T-block",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reward_vs_visual.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'reward_vs_visual.png'}")

    # ── print summary ────────────────────────────────────────────────
    print("\n=== Evaluation summary ===")
    for i, cs in enumerate(all_cos):
        print(f"  ep {i}: init={cs[0]:.4f}  final={cs[-1]:.4f}  "
              f"best={max(cs):.4f}")
    print(f"  mean init:  {avg_init:.4f}")
    print(f"  mean final: {avg_final:.4f}")
    print(f"  mean Δ:     {avg_final - avg_init:+.4f}")


if __name__ == "__main__":
    main()
