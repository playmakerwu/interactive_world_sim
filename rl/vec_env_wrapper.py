"""rl-games VecEnv wrapper around LatentPushTForRLGames.

Bridges our pure-GPU env to rl-games' IVecEnv interface.
This is a thin adapter — all logic stays in latent_env.py.
"""

from typing import Any, Dict, Tuple

import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.vecenv import IVecEnv

from rl.latent_env import LatentEnvConfig, LatentPushTForRLGames


class RLGamesLatentVecEnv(IVecEnv):
    """rl-games-compliant wrapper.

    rl-games will call:
        reset() → obs (dict or tensor)
        step(actions) → (obs, rewards, dones, infos)
        get_env_info() → dict with observation_space, action_space, agents, value_size
        get_number_of_agents() → int
    """

    def __init__(self, config_name: str, num_actors: int, **kwargs: Any):
        env_kwargs = kwargs.get("env_kwargs", {})
        cfg = LatentEnvConfig(**env_kwargs)
        self.env = LatentPushTForRLGames(cfg, num_envs=num_actors)
        self.num_envs = num_actors

    def step(self, actions: torch.Tensor) -> Tuple:
        obs_dict, rewards, dones, infos = self.env.step(actions)
        return obs_dict, rewards, dones, infos

    def reset(self) -> Dict[str, torch.Tensor]:
        return self.env.reset()

    def reset_done(self) -> Dict[str, torch.Tensor]:
        # Auto-reset already happens in step(); this is a no-op
        return self.env.reset()

    def get_number_of_agents(self) -> int:
        return self.env.get_number_of_agents()

    def get_env_info(self) -> dict:
        return self.env.get_env_info()

    def has_action_masks(self) -> bool:
        return False


def _create_single_env(**kwargs: Any) -> LatentPushTForRLGames:
    """Single-env factory used by rl-games BasePlayer (eval / play mode).

    BasePlayer calls env_creator(**env_config) and then queries
    env.observation_space and env.action_space directly.
    """
    env_kwargs = kwargs.get("env_kwargs", {})
    cfg = LatentEnvConfig(**env_kwargs)
    return LatentPushTForRLGames(cfg, num_envs=1)


def register_latent_pusht_env() -> None:
    """Register LatentPushT with rl-games env_configurations and vecenv registries."""
    env_configurations.register(
        "LatentPushT",
        {
            "vecenv_type": "LATENT_PUSHT",
            "env_creator": _create_single_env,  # used by BasePlayer for eval
        },
    )
    vecenv.register(
        "LATENT_PUSHT",
        lambda config_name, num_actors, **kwargs: RLGamesLatentVecEnv(
            config_name, num_actors, **kwargs
        ),
    )
