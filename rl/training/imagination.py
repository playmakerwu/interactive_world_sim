"""Imagination rollout: actor + differentiable dynamics."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rl.models.actor import Actor
from rl.models.world_model import DifferentiableDynamics


@dataclass
class ImaginedTrajectory:
    latents: torch.Tensor   # (B, H+1, C, Hl, Wl)
    actions: torch.Tensor   # (B, H, A)
    rewards: torch.Tensor   # (B, H)


def imagine(
    actor: Actor,
    dynamics: DifferentiableDynamics,
    z_init: torch.Tensor,   # (B, 1, C, H, W)
    z_goal: torch.Tensor,   # (1, C, H, W)
    H: int,
    hist_context: int = 10,
) -> ImaginedTrajectory:
    """Imagination rollout.  All tensors preserve grad for actor training."""
    B = z_init.shape[0]
    device = z_init.device
    A = actor.net[-2].out_features  # action dim from last Linear before Tanh
    use_ckpt = True  # always checkpoint to save memory

    z = z_init  # (B, T_hist, C, H, W)  starts as (B, 1, ...)
    action_hist = [torch.zeros(B, 1, A, device=device)]  # dummy initial

    latents = [z[:, -1]]  # z_0
    actions_list = []
    rewards_list = []

    z_goal_flat = z_goal.reshape(1, -1).expand(B, -1)  # (B, D)

    for t in range(H):
        # actor produces action
        a = actor.act(z[:, -1])  # (B, A)
        actions_list.append(a)

        # build action input
        action_hist.append(a.unsqueeze(1))
        T_hist = z.shape[1]
        all_act = torch.cat(action_hist, dim=1)
        act_input = all_act[:, -(T_hist + 1) :]

        # dynamics step
        z_next = dynamics.step(z, act_input, use_checkpoint=use_ckpt)

        # update latent history
        z = torch.cat([z, z_next], dim=1)[:, -hist_context:]

        # reward = cosine similarity to goal
        z_flat = z_next[:, 0].reshape(B, -1)
        reward = F.cosine_similarity(z_flat, z_goal_flat, dim=1)  # (B,)
        rewards_list.append(reward)
        latents.append(z_next[:, 0])

    return ImaginedTrajectory(
        latents=torch.stack(latents, dim=1),      # (B, H+1, C, H, W)
        actions=torch.stack(actions_list, dim=1),  # (B, H, A)
        rewards=torch.stack(rewards_list, dim=1),  # (B, H)
    )
