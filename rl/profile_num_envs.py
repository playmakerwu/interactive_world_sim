#!/usr/bin/env python3
"""Profile VRAM usage at different num_envs levels."""
import os, sys, yaml, torch, gc, tempfile

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

from rl.vec_env_wrapper import register_latent_pusht_env
from rl_games.torch_runner import Runner

register_latent_pusht_env()

def measure(num_envs):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    with open("rl/configs/latent_pusht_ppo.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["params"]["config"]["num_actors"] = num_envs
    cfg["params"]["config"]["horizon_length"] = 32
    # minibatch must divide num_envs * horizon
    total = num_envs * 32
    cfg["params"]["config"]["minibatch_size"] = min(1024, total)
    cfg["params"]["config"]["seq_length"] = 16
    cfg["params"]["config"]["max_epochs"] = 1
    cfg["params"]["config"]["save_frequency"] = 99999
    cfg["params"]["config"]["save_best_after"] = 99999
    cfg["params"]["config"]["mini_epochs"] = 2
    cfg["params"]["config"]["name"] = f"profile_n{num_envs}"
    cfg["params"]["config"]["full_experiment_name"] = f"profile_n{num_envs}"
    cfg["params"]["config"]["env_config"]["env_kwargs"]["dataset_path"] = \
        os.path.join(REPO_ROOT, "data/mock_latent_dataset.pt")

    runner = Runner()
    runner.load(cfg)
    runner.reset()

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            runner.run({"train": True, "play": False, "checkpoint": None, "sigma": None})
        finally:
            os.chdir(cwd)

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return peak_mb

print(f"{'num_envs':>10s} {'Peak VRAM':>14s}")
print("-" * 30)
for n in [4096, 8192, 16384, 32768]:
    try:
        peak = measure(n)
        print(f"{n:>10d} {peak:>11.0f} MB")
    except torch.cuda.OutOfMemoryError as e:
        print(f"{n:>10d} {'OOM':>14s}")
        break
    except Exception as e:
        print(f"{n:>10d} {'ERROR':>14s}: {e}")
        break
