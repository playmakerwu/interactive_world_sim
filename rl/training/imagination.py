"""Imagination rollout: actor + differentiable dynamics.

Supports both continuous (Actor) and discrete (DiscreteActor) actors.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rl.models.world_model import DifferentiableDynamics


@dataclass
class ImaginedTrajectory:
    latents: torch.Tensor      # (B, H+1, C, Hl, Wl)
    actions: torch.Tensor      # (B, H, A)
    rewards: torch.Tensor      # (B, H)  — cosine sim to goal per step
    logits: torch.Tensor | None = None   # (B, H, N) for discrete
    action_idx: torch.Tensor | None = None  # (B, H) argmax indices for discrete


def imagine(
    actor,
    dynamics: DifferentiableDynamics,
    z_init: torch.Tensor,
    z_goal: torch.Tensor,
    H: int,
    hist_context: int = 10,
    discrete: bool = False,
    temperature: float = 1.0,
    use_checkpoint: bool = True,
) -> ImaginedTrajectory:
    """Imagination rollout.  Gradients flow through actor and dynamics."""
    B = z_init.shape[0]
    device = z_init.device

    # DataParallel wraps expose action_dim/act/act_eval via .module
    actor_unwrapped = getattr(actor, "module", actor)

    if discrete:
        A = actor_unwrapped.action_dim
    else:
        A = actor_unwrapped.net[-2].out_features

    use_ckpt = use_checkpoint

    z = z_init
    action_hist = [torch.zeros(B, 1, A, device=device)]
    latents = [z[:, -1]]
    actions_list = []
    rewards_list = []
    logits_list = []
    idx_list = []

    z_goal_flat = z_goal.reshape(1, -1).expand(B, -1)

    for t in range(H):
        if discrete:
            a, logits, one_hot = actor_unwrapped.act(z[:, -1], temperature=temperature)
            logits_list.append(logits)
            idx_list.append(one_hot.argmax(dim=-1))
        else:
            a = actor_unwrapped.act(z[:, -1])
        actions_list.append(a)

        action_hist.append(a.unsqueeze(1))
        T_hist = z.shape[1]
        all_act = torch.cat(action_hist, dim=1)
        act_input = all_act[:, -(T_hist + 1) :]

        z_next = dynamics.step(z, act_input, use_checkpoint=use_ckpt)
        z = torch.cat([z, z_next], dim=1)[:, -hist_context:]

        # reward = cosine similarity to goal
        z_flat = z_next[:, 0].reshape(B, -1)
        reward = F.cosine_similarity(z_flat, z_goal_flat, dim=1)
        rewards_list.append(reward)
        latents.append(z_next[:, 0])

    return ImaginedTrajectory(
        latents=torch.stack(latents, dim=1),
        actions=torch.stack(actions_list, dim=1),
        rewards=torch.stack(rewards_list, dim=1),
        logits=torch.stack(logits_list, dim=1) if logits_list else None,
        action_idx=torch.stack(idx_list, dim=1) if idx_list else None,
    )
