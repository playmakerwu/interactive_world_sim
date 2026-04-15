"""Dreamer-style training logic with support for discrete actors."""

import numpy as np
import torch
import torch.nn.functional as F

from rl.models.critic import Critic, soft_update
from rl.models.world_model import DifferentiableDynamics
from rl.training.imagination import ImaginedTrajectory, imagine
from rl.training.replay_buffer import LatentReplayBuffer
from rl.utils.config import DreamerConfig


def compute_lambda_returns(
    rewards: torch.Tensor, values: torch.Tensor,
    gamma: float, lambda_: float,
) -> torch.Tensor:
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
    actor,
    critic: Critic,
    target_critic: Critic,
    dynamics: DifferentiableDynamics,
    replay_buffer: LatentReplayBuffer,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    z_goal: torch.Tensor,
    cfg: DreamerConfig,
    temperature: float = 1.0,
) -> dict:
    z_init = replay_buffer.sample(cfg.batch_size)

    traj: ImaginedTrajectory = imagine(
        actor, dynamics, z_init, z_goal,
        H=cfg.imagination_horizon,
        hist_context=cfg.hist_context,
        discrete=cfg.discrete,
        temperature=temperature,
        use_checkpoint=cfg.use_gradient_checkpointing,
    )

    with torch.no_grad():
        flat_z = traj.latents.detach().reshape(-1, *traj.latents.shape[2:])
        flat_v = target_critic(flat_z)
        values = flat_v.reshape(cfg.batch_size, cfg.imagination_horizon + 1)

    # ── critic update ────────────────────────────────────────────────
    critic_opt.zero_grad()
    critic_latents = traj.latents[:, :-1].detach().reshape(-1, *traj.latents.shape[2:])
    with torch.no_grad():
        lambda_targets = compute_lambda_returns(
            traj.rewards.detach(), values, cfg.gamma, cfg.lambda_,
        )
    critic_unwrapped = getattr(critic, "module", critic)
    v1, v2 = critic_unwrapped.both(critic_latents)
    v1 = v1.reshape(cfg.batch_size, cfg.imagination_horizon)
    v2 = v2.reshape(cfg.batch_size, cfg.imagination_horizon)
    critic_loss = 0.5 * (
        F.mse_loss(v1, lambda_targets) + F.mse_loss(v2, lambda_targets)
    )
    critic_loss.backward()
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        critic.parameters(), cfg.grad_clip,
    ).item()
    critic_opt.step()

    # ── actor update ─────────────────────────────────────────────────
    actor_opt.zero_grad()
    actor_lambda = compute_lambda_returns(
        traj.rewards, values.detach(), cfg.gamma, cfg.lambda_,
    )
    actor_loss = -actor_lambda.mean()

    entropy_val = 0.0
    if cfg.discrete and traj.logits is not None:
        # entropy of categorical from logits
        log_probs = F.log_softmax(traj.logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, H)
        mean_entropy = entropy.mean()
        entropy_val = mean_entropy.item()
        actor_loss = actor_loss - cfg.entropy_coef * mean_entropy

    actor_loss.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), cfg.grad_clip,
    ).item()
    actor_opt.step()

    soft_update(target_critic, critic, cfg.target_tau)

    # action distribution metrics
    action_hist = None
    if cfg.discrete and traj.action_idx is not None:
        actor_unwrapped = getattr(actor, "module", actor)
        idx_np = traj.action_idx.detach().cpu().numpy().ravel()
        action_hist = np.bincount(
            idx_np, minlength=actor_unwrapped.n_actions,
        ).astype(np.float32)
        action_hist = action_hist / action_hist.sum()

    metrics = {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "mean_reward": traj.rewards.detach().mean().item(),
        "mean_final_reward": traj.rewards[:, -1].detach().mean().item(),
        "mean_lambda_return": lambda_targets.mean().item(),
        "actor_grad_norm": actor_grad_norm,
        "critic_grad_norm": critic_grad_norm,
        "mean_entropy": entropy_val,
        "temperature": temperature,
    }
    if action_hist is not None:
        metrics["action_hist"] = action_hist
    return metrics
