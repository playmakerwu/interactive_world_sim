"""Dreamer-style RL training entry point.

Usage (from repo root):
    conda run -n iws python rl/train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import torch

from rl.models.actor import Actor
from rl.models.critic import Critic, make_target_critic
from rl.models.world_model import DifferentiableDynamics
from rl.training.dreamer import train_step
from rl.training.replay_buffer import LatentReplayBuffer
from rl.utils.config import DreamerConfig
from rl.utils.logger import Logger


def main():
    cfg = DreamerConfig()

    # directories
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    log_dir = Path(cfg.output_dir) / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # world model
    print("Loading world model …")
    dynamics = DifferentiableDynamics(cfg.ckpt_path, device=cfg.device)

    # goal
    z_goal = torch.load(cfg.goal_path, map_location=cfg.device, weights_only=True)
    z_goal = z_goal.float()
    print(f"Goal latent: {tuple(z_goal.shape)}, norm={z_goal.flatten().norm():.2f}")

    # replay buffer
    replay_buffer = LatentReplayBuffer(
        dynamics, cfg.dataset_dir,
        obs_key=cfg.obs_key, resolution=cfg.resolution, device=cfg.device,
    )

    # actor & critic
    actor = Actor(cfg.latent_dim, cfg.action_dim, cfg.actor_hidden, cfg.actor_layers)
    actor = actor.to(cfg.device).float()
    actor.train()

    critic = Critic(cfg.latent_dim, cfg.critic_hidden, cfg.critic_layers)
    critic = critic.to(cfg.device).float()
    critic.train()

    target_critic = make_target_critic(critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    logger = Logger(str(log_dir))

    print(f"\nStarting training for {cfg.total_steps} steps "
          f"(B={cfg.batch_size}, H={cfg.imagination_horizon})\n")

    for step in range(1, cfg.total_steps + 1):
        t0 = time.time()
        metrics = train_step(
            actor, critic, target_critic, dynamics,
            replay_buffer, actor_opt, critic_opt,
            z_goal, cfg,
        )
        dt = time.time() - t0

        logger.log(step, metrics)

        if step % cfg.log_every == 0:
            print(
                f"step {step:6d} | "
                f"actor_loss {metrics['actor_loss']:+.4f} | "
                f"critic_loss {metrics['critic_loss']:.4f} | "
                f"avg_r {metrics['mean_reward']:.4f} | "
                f"final_r {metrics['mean_final_reward']:.4f} | "
                f"λ_ret {metrics['mean_lambda_return']:.4f} | "
                f"a_grad {metrics['actor_grad_norm']:.2e} | "
                f"{dt:.1f}s"
            )

        if step % cfg.save_every == 0:
            torch.save({
                "step": step,
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": target_critic.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
            }, ckpt_dir / f"step_{step}.pt")
            print(f"  → checkpoint saved: {ckpt_dir / f'step_{step}.pt'}")

    logger.close()
    # save final
    torch.save({
        "step": cfg.total_steps,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
    }, ckpt_dir / "final.pt")
    print(f"\nTraining complete. Final checkpoint: {ckpt_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
