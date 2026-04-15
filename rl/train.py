"""Dreamer-style RL training entry point.

Usage (from repo root):
    conda run -n iws python rl/train.py [--flag value ...]

Any dataclass field of DreamerConfig can be overridden via CLI, e.g.:
    python rl/train.py --total_steps 50 --batch_size 4 --imagination_horizon 15
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import dataclasses
import time

import numpy as np
import torch

from rl.models.actor import Actor
from rl.models.actor_discrete import DiscreteActor
from rl.models.critic import Critic, make_target_critic
from rl.models.world_model import DifferentiableDynamics
from rl.training.dreamer import train_step
from rl.training.replay_buffer import LatentReplayBuffer
from rl.utils.config import DreamerConfig
from rl.utils.logger import Logger


def _parse_overrides() -> dict:
    """Parse CLI flags as DreamerConfig overrides (type-coerced from the dataclass)."""
    parser = argparse.ArgumentParser()
    for f in dataclasses.fields(DreamerConfig):
        if f.type is bool or f.type == "bool":
            parser.add_argument(
                f"--{f.name}",
                type=lambda s: str(s).lower() in ("1", "true", "yes", "y"),
                default=None,
            )
        else:
            parser.add_argument(f"--{f.name}", type=f.type, default=None)
    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}


def _maybe_data_parallel(module: torch.nn.Module, cfg: DreamerConfig):
    """Wrap module in DataParallel if multi-GPU is enabled.

    Note: actor/critic are small MLPs; DataParallel is mostly symbolic for them.
    The world model stays on cfg.device and is never replicated.
    """
    if cfg.use_multi_gpu and cfg.num_gpus > 1 and torch.cuda.device_count() > 1:
        device_ids = list(range(min(cfg.num_gpus, torch.cuda.device_count())))
        return torch.nn.DataParallel(module, device_ids=device_ids)
    return module


def main():
    overrides = _parse_overrides()
    cfg = dataclasses.replace(DreamerConfig(), **overrides)

    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    log_dir = Path(cfg.output_dir) / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # report
    print("=" * 60)
    print("DreamerConfig (overrides applied):")
    for f in dataclasses.fields(DreamerConfig):
        v = getattr(cfg, f.name)
        tag = " (override)" if f.name in overrides else ""
        print(f"  {f.name:30s} = {v!r}{tag}")
    print("=" * 60)

    # GPU report
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"CUDA: {n} GPU(s) visible")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name} ({p.total_memory / 1e9:.1f} GB)")

    print("\nLoading world model …")
    dynamics = DifferentiableDynamics(cfg.ckpt_path, device=cfg.device)

    z_goal = torch.load(cfg.goal_path, map_location=cfg.device, weights_only=True)
    z_goal = z_goal.float()
    print(f"Goal latent: {tuple(z_goal.shape)}, norm={z_goal.flatten().norm():.2f}")

    replay_buffer = LatentReplayBuffer(
        dynamics, cfg.dataset_dir,
        obs_key=cfg.obs_key, resolution=cfg.resolution, device=cfg.device,
        z_goal=z_goal,
        sampling_mode=cfg.sampling_mode,
        hard_threshold_percentile=cfg.hard_threshold_percentile,
    )
    stats = replay_buffer.get_stats()
    print(f"\nReplay buffer stats ({stats['mode']} mode):")
    print(f"  Active frames: {stats['n_frames']}/{stats['n_total']}")
    print(f"  Cos sim range: [{stats['min_cos']:.4f}, {stats['max_cos']:.4f}]")
    print(f"  Cos sim mean:  {stats['mean_cos']:.4f}")

    # actor
    if cfg.discrete:
        actor = DiscreteActor(
            latent_dim=cfg.latent_dim,
            hidden=cfg.actor_hidden,
            n_layers=cfg.actor_layers,
            action_table_path=cfg.action_table_path,
        )
        print(f"Discrete actor: {actor.n_actions} actions")
        print(f"  Action names: {actor.action_names}")
    else:
        actor = Actor(cfg.latent_dim, cfg.action_dim, cfg.actor_hidden, cfg.actor_layers)
    actor = actor.to(cfg.device).float()
    actor.train()
    actor = _maybe_data_parallel(actor, cfg)

    critic = Critic(cfg.latent_dim, cfg.critic_hidden, cfg.critic_layers)
    critic = critic.to(cfg.device).float()
    critic.train()
    target_critic = make_target_critic(critic)
    critic = _maybe_data_parallel(critic, cfg)

    actor_params = actor.parameters()
    critic_params = critic.parameters()
    actor_opt = torch.optim.Adam(actor_params, lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic_params, lr=cfg.critic_lr)

    logger = Logger(str(log_dir))

    actor_for_meta = getattr(actor, "module", actor)
    print(f"\nStarting training for {cfg.total_steps} steps "
          f"(B={cfg.batch_size}, H={cfg.imagination_horizon}, "
          f"ckpt={cfg.use_gradient_checkpointing}, "
          f"discrete={cfg.discrete}, sampling={cfg.sampling_mode})\n")

    cumulative_action_hist = np.zeros(
        actor_for_meta.n_actions if cfg.discrete else 0, dtype=np.float64
    )
    cum_count = 0

    for step in range(1, cfg.total_steps + 1):
        replay_buffer.set_train_step(step)

        # anneal temperature
        if cfg.discrete and cfg.temperature_anneal:
            frac = (step - 1) / max(1, cfg.total_steps - 1)
            temperature = (
                cfg.gumbel_temperature * (1 - frac)
                + cfg.gumbel_temperature_end * frac
            )
        else:
            temperature = cfg.gumbel_temperature

        t0 = time.time()
        metrics = train_step(
            actor, critic, target_critic, dynamics,
            replay_buffer, actor_opt, critic_opt,
            z_goal, cfg, temperature=temperature,
        )
        dt = time.time() - t0

        if cfg.discrete and "action_hist" in metrics:
            cumulative_action_hist += metrics["action_hist"]
            cum_count += 1

        log_scalars = {k: v for k, v in metrics.items() if k != "action_hist"}
        logger.log(step, log_scalars)

        if step % cfg.log_every == 0:
            if cfg.discrete:
                hist = cumulative_action_hist / max(1, cum_count)
                top_idx = int(np.argmax(hist))
                top_name = actor_for_meta.action_names[top_idx]
                top_pct = 100 * hist[top_idx]
                print(
                    f"step {step:6d} | "
                    f"a_loss {metrics['actor_loss']:+.3f} | "
                    f"c_loss {metrics['critic_loss']:.3f} | "
                    f"r {metrics['mean_reward']:.4f} | "
                    f"ent {metrics['mean_entropy']:.3f} | "
                    f"τ {temperature:.3f} | "
                    f"top: {top_name} ({top_pct:.0f}%) | "
                    f"λ {metrics['mean_lambda_return']:.3f} | {dt:.1f}s"
                )
                cumulative_action_hist[:] = 0
                cum_count = 0
            else:
                print(
                    f"step {step:6d} | "
                    f"a_loss {metrics['actor_loss']:+.4f} | "
                    f"c_loss {metrics['critic_loss']:.4f} | "
                    f"r {metrics['mean_reward']:.4f} | "
                    f"λ {metrics['mean_lambda_return']:.4f} | {dt:.1f}s"
                )

        if step % cfg.save_every == 0:
            torch.save({
                "step": step,
                "actor": actor_for_meta.state_dict(),
                "critic": getattr(critic, "module", critic).state_dict(),
                "target_critic": target_critic.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "discrete": cfg.discrete,
            }, ckpt_dir / f"step_{step}.pt")
            print(f"  → checkpoint saved: {ckpt_dir / f'step_{step}.pt'}")

    logger.close()
    torch.save({
        "step": cfg.total_steps,
        "actor": actor_for_meta.state_dict(),
        "critic": getattr(critic, "module", critic).state_dict(),
        "discrete": cfg.discrete,
    }, ckpt_dir / "final.pt")
    print(f"\nTraining complete. Final checkpoint: {ckpt_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
