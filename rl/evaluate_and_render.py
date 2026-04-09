#!/usr/bin/env python3
"""Phase 4: Dream Decoder — Evaluate trained PPO + render latent rollouts to video.

Pipeline:
    1. Load trained PPO checkpoint (rl-games PpoPlayerContinuous)
    2. Run deterministic autoregressive rollout in LatentPushTForRLGames
    3. Capture (z_t, action_t, reward_t, distance_t) trajectory
    4. Render as MP4:
       - mock mode  : PCA-based 2D scatter animation + distance subplot
       - real mode  : uses LatentWorldModel.decoder via render_img_cm

Usage:
    # Render with mock visualization (no real AE needed)
    PYTHONPATH=. python rl/evaluate_and_render.py \
        --checkpoint runs/latent_pusht_ppo_phase4_input/nn/last_latent_pusht_ppo_ep_30_rew__99.99447_.pth \
        --output runs/latent_pusht_ppo_phase4_input/videos/episode.mp4

    # Render multiple episodes
    PYTHONPATH=. python rl/evaluate_and_render.py \
        --checkpoint <path> \
        --n_episodes 4 \
        --output_dir runs/latent_pusht_ppo_phase4_input/videos/

    # Render with REAL autoencoder (requires latent_dim matches the AE)
    PYTHONPATH=. python rl/evaluate_and_render.py \
        --checkpoint <path> \
        --real_decoder outputs/pusht_cam1/checkpoints/best.ckpt \
        --output runs/.../episode_real.mp4
"""

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

# Register OmegaConf resolvers (needed for any rl-games config that uses ${eval:...})
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
if not OmegaConf.has_resolver("torch"):
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))

from rl.latent_env import LatentEnvConfig, LatentPushTForRLGames
from rl.vec_env_wrapper import register_latent_pusht_env


# ============================================================
# 1. Trained policy loader
# ============================================================

def load_ppo_player(config_path: str, checkpoint_path: str, num_envs: int):
    """Load a trained PPO policy from rl-games checkpoint.

    Returns the rl-games PpoPlayerContinuous instance which handles
    LSTM hidden state internally via init_rnn() / reset() / get_action().
    """
    from rl_games.algos_torch.players import PpoPlayerContinuous

    # Register the env so PpoPlayerContinuous can build it
    register_latent_pusht_env()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Override num_actors so we don't spin up 64 envs for evaluation
    cfg["params"]["config"]["num_actors"] = num_envs
    h = cfg["params"]["config"]["horizon_length"]
    cfg["params"]["config"]["minibatch_size"] = min(
        cfg["params"]["config"]["minibatch_size"], num_envs * h
    )
    # Make dataset path absolute (player creates the env internally)
    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    cfg["params"]["config"]["env_config"]["env_kwargs"]["dataset_path"] = \
        os.path.join(repo_root, cfg["params"]["config"]["env_config"]["env_kwargs"]["dataset_path"])
    # Disable telemetry during eval
    cfg["params"]["config"]["env_config"]["env_kwargs"]["tb_log_dir"] = None

    player = PpoPlayerContinuous(cfg["params"])
    player.restore(checkpoint_path)
    player.has_batch_dimension = True
    print(f"  Loaded checkpoint: {checkpoint_path}")
    print(f"  Network: {type(player.model).__name__}")
    return player, cfg


# ============================================================
# 2. Autoregressive rollout
# ============================================================

@torch.no_grad()
def rollout_episode(
    player,
    env: LatentPushTForRLGames,
    max_steps: int = 60,
    min_initial_distance: float = 0.5,
) -> Dict:
    """Run one deterministic episode and capture the full trajectory.

    To get an interesting visualization, we resample the initial state
    until z_start is at least `min_initial_distance` from z_goal.
    Otherwise the policy may instantly succeed in 1 step.

    The env may auto-reset on success/death/timeout, so we capture
    z_current BEFORE calling step() and break on done before the
    auto-reset overwrites the state.
    """
    device = env.device

    # Resample until we get a non-trivial starting state
    for resample in range(20):
        obs = env.reset()
        initial_dist = (env.z_current[0] - env.z_goal[0]).norm().item()
        if initial_dist >= min_initial_distance:
            break

    if hasattr(player, "init_rnn"):
        player.init_rnn()

    z_goal = env.z_goal[0].clone()
    z_history = [env.z_current[0].clone()]
    distances = [initial_dist]
    actions, rewards = [], []
    done_at = -1
    final_z = None
    final_reward = None

    for step in range(max_steps):
        # Capture current state for visualization (BEFORE step / auto-reset)
        z_t_pre = env.z_current[0].clone()

        action = player.get_action(obs["obs"], is_deterministic=True)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Store action before step
        actions.append(action[0].clone())

        # Step (may trigger auto-reset internally)
        obs, reward, done, info = env.step(action)

        rewards.append(reward[0].item())
        done_flag = done[0].item() > 0

        if done_flag:
            # The env has already auto-reset; we cannot trust env.z_current.
            # But we know the agent took `action` from z_t_pre.
            # Compute the next state directly via the base model so we can
            # show the final position relative to the (still-known) goal.
            with torch.no_grad():
                z_next = env.ensemble.base_model(z_t_pre.unsqueeze(0), action)[0]
            z_history.append(z_next.clone())
            distances.append((z_next - z_goal).norm().item())
            done_at = step + 1
            break
        else:
            # Normal case — env.z_current is the post-step state
            z_history.append(env.z_current[0].clone())
            distances.append((env.z_current[0] - z_goal).norm().item())

    return {
        "z_history": torch.stack(z_history),
        "z_goal": z_goal,
        "actions": torch.stack(actions) if actions else torch.zeros(0, env.cfg.action_dim),
        "rewards": torch.tensor(rewards),
        "distances": torch.tensor(distances),
        "done_at": done_at,
        "n_steps": len(actions),
        "initial_distance": initial_dist,
    }


# ============================================================
# 3a. Mock mode renderer — PCA scatter animation
# ============================================================

def render_mock_video(traj: Dict, output_path: str, fps: int = 10) -> None:
    """Render a trajectory as a 2-panel matplotlib animation:
        Left:  PCA-projected latent space scatter (z_history + z_goal + path)
        Right: distance to goal over time
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    z_hist = traj["z_history"].cpu().numpy()       # (T+1, D)
    z_goal = traj["z_goal"].cpu().numpy()          # (D,)
    distances = traj["distances"].cpu().numpy()
    n_frames = z_hist.shape[0]

    # Fit PCA on the union of trajectory + goal for stable projection
    all_points = np.vstack([z_hist, z_goal[None]])
    centered = all_points - all_points.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pcs = Vt[:2]                                    # (2, D)

    z_hist_2d = (z_hist - all_points.mean(0)) @ pcs.T   # (T+1, 2)
    z_goal_2d = (z_goal - all_points.mean(0)) @ pcs.T   # (2,)

    # Setup figure
    fig, (ax_pca, ax_dist) = plt.subplots(1, 2, figsize=(14, 6))

    # ----- Left panel: PCA scatter -----
    ax_pca.set_title("Latent Trajectory (PCA 2D projection)")
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.grid(alpha=0.3)
    pad = 0.15
    xlim = [z_hist_2d[:, 0].min() - pad, z_hist_2d[:, 0].max() + pad]
    ylim = [z_hist_2d[:, 1].min() - pad, z_hist_2d[:, 1].max() + pad]
    ax_pca.set_xlim(xlim)
    ax_pca.set_ylim(ylim)
    # Static elements
    ax_pca.scatter(*z_hist_2d[0], s=120, c="green", marker="o", label="z_start", zorder=5, edgecolors="black")
    ax_pca.scatter(*z_goal_2d, s=200, c="red", marker="*", label="z_goal", zorder=5, edgecolors="black")
    # Dynamic elements (we'll update these)
    path_line, = ax_pca.plot([], [], "b-", lw=1.5, alpha=0.5, label="z_t path")
    current_dot, = ax_pca.plot([], [], "bo", markersize=10, markeredgecolor="black", label="z_current")
    ax_pca.legend(loc="upper right", fontsize=9)

    # ----- Right panel: distance over time -----
    ax_dist.set_title("Distance to Goal")
    ax_dist.set_xlabel("Step")
    ax_dist.set_ylabel("L2 distance")
    ax_dist.grid(alpha=0.3)
    ax_dist.set_xlim(0, n_frames - 1)
    ax_dist.set_ylim(0, distances.max() * 1.1 + 0.1)
    ax_dist.axhline(y=0, color="green", linestyle="--", alpha=0.5, label="goal reached")
    dist_line, = ax_dist.plot([], [], "b-", lw=2, label="distance")
    dist_dot, = ax_dist.plot([], [], "bo", markersize=8)
    ax_dist.legend(loc="upper right", fontsize=9)

    # Title with episode info
    fig.suptitle(
        f"PPO Latent Rollout — {traj['n_steps']} steps, "
        f"final distance={distances[-1]:.3f}",
        fontsize=12,
    )

    def update(frame):
        # Path up to current frame
        path_line.set_data(z_hist_2d[: frame + 1, 0], z_hist_2d[: frame + 1, 1])
        current_dot.set_data([z_hist_2d[frame, 0]], [z_hist_2d[frame, 1]])
        # Distance plot
        dist_line.set_data(np.arange(frame + 1), distances[: frame + 1])
        dist_dot.set_data([frame], [distances[frame]])
        return path_line, current_dot, dist_line, dist_dot

    anim = FuncAnimation(fig, update, frames=n_frames, blit=True, interval=1000 // fps)

    # Save as MP4
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        anim.save(output_path, writer="ffmpeg", fps=fps, dpi=120)
        print(f"  Saved MP4: {output_path}")
    except Exception as e:
        # Fallback to GIF
        gif_path = output_path.replace(".mp4", ".gif")
        anim.save(gif_path, writer="pillow", fps=fps)
        print(f"  ffmpeg failed ({e}); fell back to GIF: {gif_path}")
    plt.close(fig)


# ============================================================
# 3b. Real mode renderer — uses LatentWorldModel decoder
# ============================================================

def render_real_video(traj: Dict, ae_ckpt: str, output_path: str, fps: int = 10) -> None:
    """Render trajectory using the real LatentWorldModel autoencoder decoder.

    Note: requires the env's latent shape to match the AE's expected input.
    For our mock env (latent_dim=256, flat), this won't work directly with
    the real AE which expects (4, 32, 32). This function is the architectural
    placeholder for when the env is upgraded to wrap the real LatentWorldModel.
    """
    import imageio
    from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel
    from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

    z_hist = traj["z_history"]  # (T+1, D_flat)
    n_frames, D_flat = z_hist.shape

    # Try to reshape to (T+1, C, H, W) for the AE — this only works if D matches
    # Standard AE: C=4, H=W=32 → D=4096
    # Our mock:   D=256 → does not match
    expected_dims = [(4, 32, 32), (8, 16, 16), (16, 8, 8)]
    z_4d = None
    for c, h, w in expected_dims:
        if c * h * w == D_flat:
            z_4d = z_hist.view(n_frames, c, h, w)
            print(f"  Reshaped latent: {z_hist.shape} → {z_4d.shape}")
            break

    if z_4d is None:
        raise RuntimeError(
            f"Latent dim {D_flat} doesn't match any standard AE shape "
            f"({expected_dims}). Real-decoder mode only works when the env "
            f"uses the real LatentWorldModel encoder. For mock env, use mock mode."
        )

    # Load AE
    print(f"  Loading AE from {ae_ckpt}")
    ckpt_dir = os.path.dirname(os.path.dirname(ae_ckpt))
    cfg = OmegaConf.load(os.path.join(ckpt_dir, ".hydra", "config.yaml"))
    cfg.algorithm.training_stage = 3
    cfg.algorithm.load_ae = None
    cfg.algorithm.use_prebaked_latent = False

    model = LatentWorldModel.load_from_checkpoint(
        ae_ckpt, cfg=cfg.algorithm, map_location="cuda",
        weights_only=False, strict=False,
    )
    model.eval().to("cuda")

    # Render frames
    frames = []
    with torch.no_grad():
        for t in range(n_frames):
            z_t = z_4d[t : t + 1].cuda()  # (1, C, H, W)
            img = render_img_cm(
                model, z_t, resolution=128,
                normalizer=model.normalizer, num_views=1,
            )  # expected (1, 3, H, W) in [0, 1]
            img_np = (img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            frames.append(img_np)

    # Save MP4
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"  Saved real-decoder MP4: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Trained PPO .pth")
    parser.add_argument("--config", default="rl/configs/latent_pusht_ppo.yaml")
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--min_initial_distance", type=float, default=0.5,
                        help="Resample start state until z_start is at least this far from goal")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--disable_success_term", action="store_true",
                        help="Don't terminate on success — render full max_steps trajectory")
    parser.add_argument("--success_threshold", type=float, default=None,
                        help="Override success threshold (smaller = harder to terminate)")
    parser.add_argument("--output", default=None,
                        help="Output MP4 path (single episode only)")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir for multi-episode rendering")
    parser.add_argument("--real_decoder", default=None,
                        help="Path to LatentWorldModel checkpoint for real RGB decoding")
    args = parser.parse_args()

    # --- 1. Load player ---
    print("[1/4] Loading PPO checkpoint...")
    player, cfg = load_ppo_player(args.config, args.checkpoint, num_envs=1)

    # --- 2. Build env (single env for clean rollout) ---
    print("[2/4] Building env (num_envs=1)...")
    env_kwargs = cfg["params"]["config"]["env_config"]["env_kwargs"]

    # Apply eval-time overrides
    if args.disable_success_term:
        env_kwargs["success_threshold"] = -1.0  # never reach
    elif args.success_threshold is not None:
        env_kwargs["success_threshold"] = args.success_threshold
    # Allow longer episodes than the training chunk_size for visualization
    env_kwargs["max_steps"] = args.max_steps
    env_kwargs["chunk_size"] = args.max_steps

    env_cfg = LatentEnvConfig(**env_kwargs)
    env = LatentPushTForRLGames(env_cfg, num_envs=1)

    # --- 3. Roll out episodes ---
    print(f"[3/4] Running {args.n_episodes} episode(s)...")
    trajectories = []
    for ep in range(args.n_episodes):
        traj = rollout_episode(
            player, env,
            max_steps=args.max_steps,
            min_initial_distance=args.min_initial_distance,
        )
        trajectories.append(traj)
        print(f"  Episode {ep + 1}: {traj['n_steps']} steps, "
              f"final distance={traj['distances'][-1]:.3f}, "
              f"total reward={traj['rewards'].sum():.2f}, "
              f"done_at={traj['done_at']}")

    # --- 4. Render ---
    print("[4/4] Rendering video(s)...")
    if args.n_episodes == 1 and args.output:
        out_path = args.output
        if args.real_decoder:
            render_real_video(trajectories[0], args.real_decoder, out_path, args.fps)
        else:
            render_mock_video(trajectories[0], out_path, args.fps)
    else:
        out_dir = args.output_dir or "runs/eval_videos"
        for i, traj in enumerate(trajectories):
            out_path = os.path.join(out_dir, f"episode_{i:03d}.mp4")
            if args.real_decoder:
                render_real_video(traj, args.real_decoder, out_path, args.fps)
            else:
                render_mock_video(traj, out_path, args.fps)

    print("\n=== Phase 4 Complete ===")


if __name__ == "__main__":
    main()
