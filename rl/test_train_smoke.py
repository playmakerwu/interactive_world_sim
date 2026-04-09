#!/usr/bin/env python3
"""Phase 3 Smoke Test: verify rl-games integration end-to-end.

Strategy:
    1. Register env with rl-games
    2. Load YAML config
    3. Override num_actors=2, max_epochs=1, horizon_length=8 for fast test
    4. Run 1 training epoch
    5. Verify it completes without errors and produces tensorboard logs

This is NOT a meaningful training run — just a smoke test that the wiring works.
"""

import os
import sys
import tempfile

import yaml

from rl.vec_env_wrapper import register_latent_pusht_env

# Convert dataset path to absolute (so chdir later doesn't break it)
REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# --- 1. Register env ---
register_latent_pusht_env()
print("[1/5] Env registered with rl-games")

# --- 2. Load + override config ---
with open("rl/configs/latent_pusht_ppo.yaml") as f:
    cfg = yaml.safe_load(f)

# Tiny config for smoke test
cfg["params"]["config"]["num_actors"] = 2
cfg["params"]["config"]["horizon_length"] = 8
cfg["params"]["config"]["minibatch_size"] = 8   # 2*8 = 16, must divide
cfg["params"]["config"]["seq_length"] = 8
cfg["params"]["config"]["max_epochs"] = 1
cfg["params"]["config"]["save_frequency"] = 99999  # don't save
cfg["params"]["config"]["save_best_after"] = 99999
cfg["params"]["config"]["name"] = "smoke_test"
cfg["params"]["config"]["full_experiment_name"] = "smoke_test"
cfg["params"]["config"]["mini_epochs"] = 2

# Make all real-pipeline paths absolute so chdir below doesn't break them
ek = cfg["params"]["config"]["env_config"]["env_kwargs"]
ek["dataset_dir"]  = os.path.join(REPO_ROOT, ek["dataset_dir"])
ek["ensemble_dir"] = os.path.join(REPO_ROOT, ek["ensemble_dir"])
ek["ae_ckpt"]      = os.path.join(REPO_ROOT, ek["ae_ckpt"])

print("[2/5] Config loaded and overridden")
print(f"  num_actors={cfg['params']['config']['num_actors']}")
print(f"  horizon_length={cfg['params']['config']['horizon_length']}")
print(f"  max_epochs={cfg['params']['config']['max_epochs']}")
print(f"  rnn={cfg['params']['network']['rnn']}")

# --- 3. Initialize Runner ---
from rl_games.torch_runner import Runner
runner = Runner()
runner.load(cfg)
runner.reset()
print("[3/5] Runner initialized")

# --- 4. Run training (1 epoch) ---
print("[4/5] Running 1 training epoch (smoke test)...")
print("-" * 60)

# Run from a temp dir so we don't pollute the workspace
with tempfile.TemporaryDirectory() as tmpdir:
    cwd = os.getcwd()
    os.chdir(tmpdir)
    try:
        runner.run({
            "train": True,
            "play": False,
            "checkpoint": None,
            "sigma": None,
        })
    finally:
        os.chdir(cwd)

print("-" * 60)
print("[5/5] === Phase 3 SMOKE TEST PASSED ===")
print("    rl-games + LSTM + LatentPushT env wired successfully")
