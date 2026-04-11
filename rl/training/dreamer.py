"""Dreamer-style training logic: imagination → lambda returns → actor/critic update."""

import torch
import torch.nn.functional as F

from rl.models.actor import Actor
from rl.models.critic import Critic, soft_update
from rl.models.world_model import DifferentiableDynamics
from rl.training.imagination import ImaginedTrajectory, imagine
from rl.training.replay_buffer import LatentReplayBuffer
from rl.utils.config import DreamerConfig


def compute_lambda_returns(
    rewards: torch.Tensor,   # (B, H)
    values: torch.Tensor,    # (B, H+1)
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """Compute GAE-style lambda returns.  Returns (B, H)."""
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)

    for t in reversed(range(H)):
        next_val = values[:, t + 1]
        td_target = rewards[:, t] + gamma * next_val
        if t == H - 1:
            returns[:, t] = td_target
        else:
            returns[:, t] = (
                (1 - lambda_) * td_target
                + lambda_ * (rewards[:, t] + gamma * returns[:, t + 1])
            )
    return returns


def train_step(
    actor: Actor,
    critic: Critic,
    target_critic: Critic,
    dynamics: DifferentiableDynamics,
    replay_buffer: LatentReplayBuffer,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    z_goal: torch.Tensor,
    cfg: DreamerConfig,
) -> dict:
    """One Dreamer training step.  Returns metrics dict."""
    # 1. Sample initial states
    z_init = replay_buffer.sample(cfg.batch_size)  # (B, 1, C, H, W)

    # 2. Imagination rollout — one shared rollout for both actor and critic.
    #    rewards have grad (connected to actor), latents have grad.
    traj: ImaginedTrajectory = imagine(
        actor, dynamics, z_init, z_goal,
        H=cfg.imagination_horizon,
        hist_context=cfg.hist_context,
    )

    # 3. Compute value targets with target critic (no grad)
    with torch.no_grad():
        flat_z = traj.latents.detach().reshape(-1, *traj.latents.shape[2:])
        flat_v = target_critic(flat_z)
        values = flat_v.reshape(cfg.batch_size, cfg.imagination_horizon + 1)

    # 4. Update critic FIRST (detach everything — critic doesn't backprop into actor)
    critic_opt.zero_grad()
    critic_latents = traj.latents[:, :-1].detach().reshape(-1, *traj.latents.shape[2:])
    with torch.no_grad():
        lambda_targets = compute_lambda_returns(
            traj.rewards.detach(), values, cfg.gamma, cfg.lambda_
        )
    v1, v2 = critic.both(critic_latents)
    v1 = v1.reshape(cfg.batch_size, cfg.imagination_horizon)
    v2 = v2.reshape(cfg.batch_size, cfg.imagination_horizon)
    critic_loss = 0.5 * (
        F.mse_loss(v1, lambda_targets) + F.mse_loss(v2, lambda_targets)
    )
    critic_loss.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        critic.parameters(), cfg.grad_clip
    ).item()
    critic_opt.step()

    # 5. Update actor — rewards keep grad, values are detached
    actor_opt.zero_grad()
    actor_lambda = compute_lambda_returns(
        traj.rewards, values.detach(), cfg.gamma, cfg.lambda_
    )
    actor_loss = -actor_lambda.mean()
    actor_loss.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), cfg.grad_clip
    ).item()
    actor_opt.step()

    # 6. Soft update target critic
    soft_update(target_critic, critic, cfg.target_tau)

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "mean_reward": traj.rewards.detach().mean().item(),
        "mean_final_reward": traj.rewards[:, -1].detach().mean().item(),
        "mean_lambda_return": lambda_targets.mean().item(),
        "actor_grad_norm": actor_grad_norm,
        "critic_grad_norm": critic_grad_norm,
    }
