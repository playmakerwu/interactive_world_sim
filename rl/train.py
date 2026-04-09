#!/usr/bin/env python3
"""Train PPO agent in latent space using rl-games.

Usage:
    # Train (default config)
    PYTHONPATH=. python rl/train.py

    # Train with custom config + num_envs override
    PYTHONPATH=. python rl/train.py \
        --config rl/configs/latent_pusht_ppo.yaml \
        --num_envs 128 \
        --wandb

    # Play (eval) from checkpoint
    PYTHONPATH=. python rl/train.py --play --checkpoint runs/.../best.pth

    # Quick smoke test (1 epoch, 4 envs)
    PYTHONPATH=. python rl/train.py --num_envs 4 --max_epochs 1
"""

import argparse
import os
import sys

import yaml

# rl-games imports
from rl_games.torch_runner import Runner

# Our env wrapper + registration
from rl.vec_env_wrapper import register_latent_pusht_env


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def maybe_init_wandb(cfg: dict, run_name: str) -> None:
    try:
        import wandb
        wandb.init(
            project="latent-pusht-mbrl",
            name=run_name,
            config=cfg,
            sync_tensorboard=True,  # auto-mirror tensorboard scalars
        )
        print(f"  wandb initialized: project=latent-pusht-mbrl, run={run_name}")
    except ImportError:
        print("  wandb not installed; skipping")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rl/configs/latent_pusht_ppo.yaml")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name_suffix", default="")
    args = parser.parse_args()

    # --- 1. Register env ---
    register_latent_pusht_env()
    print("Registered LatentPushT env with rl-games")

    # --- 2. Load and override config ---
    cfg = load_config(args.config)
    if args.num_envs is not None:
        cfg["params"]["config"]["num_actors"] = args.num_envs
        # Adjust minibatch_size to remain valid: must divide num_envs * horizon_length
        h = cfg["params"]["config"]["horizon_length"]
        total_steps = args.num_envs * h
        if cfg["params"]["config"]["minibatch_size"] > total_steps:
            cfg["params"]["config"]["minibatch_size"] = total_steps
    if args.max_epochs is not None:
        cfg["params"]["config"]["max_epochs"] = args.max_epochs
    if args.seed is not None:
        cfg["params"]["seed"] = args.seed

    run_name = cfg["params"]["config"]["name"] + args.name_suffix
    cfg["params"]["config"]["full_experiment_name"] = run_name

    # --- Auto-configure env telemetry to write into the same run dir as rl-games ---
    # rl-games writes to runs/<full_experiment_name>/summaries
    # We write env telemetry to runs/<full_experiment_name>/env_telemetry
    env_kwargs = cfg["params"]["config"]["env_config"]["env_kwargs"]
    if env_kwargs.get("tb_log_dir") is None:
        env_kwargs["tb_log_dir"] = os.path.join("runs", run_name, "env_telemetry")
    print(f"  Env telemetry → {env_kwargs['tb_log_dir']}")

    # --- 3. wandb (optional) ---
    if args.wandb:
        maybe_init_wandb(cfg, run_name)

    # --- 4. Runner ---
    runner = Runner()
    runner.load(cfg)
    runner.reset()

    # --- 5. Train or Play ---
    runner.run({
        "train": not args.play,
        "play": args.play,
        "checkpoint": args.checkpoint,
        "sigma": None,
    })


if __name__ == "__main__":
    main()
