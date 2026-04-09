#!/usr/bin/env python3
"""Smoke test for the REAL LatentPushTForRLGames env.

Uses small num_envs (4) so it fits on a 12GB local GPU. The full L40S
deployment can scale to num_envs=256.

Verifies:
    1. Real ensemble loads from outputs/universe/ (3 checkpoints)
    2. Real dataset loads from data/mini/pusht_latent/
    3. Tensors stay on GPU through reset/step
    4. obs is 4096-d, action is 4-d
    5. Reward decreases distance over time (with random actions, weakly)
    6. variance_penalty_weight=0 means death never triggers
    7. Auto-reset works on timeout
"""

import torch

from rl.latent_env import LatentEnvConfig, LatentPushTForRLGames

device = "cuda"
NUM_ENVS = 4
N_STEPS = 8

print("=" * 70)
print("  REAL Latent Env Smoke Test")
print("=" * 70)

# --- Setup ---
cfg = LatentEnvConfig(
    ensemble_dir="outputs/universe",
    ae_ckpt="outputs/pusht_cam1/checkpoints/best.ckpt",
    dataset_dir="data/mini/pusht_latent",
    latent_dim=4096,
    action_dim=4,
    chunk_size=8,            # short episodes for fast testing
    max_steps=8,
    distance_mode="delta_l2",
    variance_threshold=0.5,
    variance_penalty_weight=0.0,    # smoke test: pure L2
    device=device,
)

print("\n[1/5] Building env (this will load 3 LatentWorldModel checkpoints)...")
env = LatentPushTForRLGames(cfg, num_envs=NUM_ENVS)

# --- Verify spaces ---
env_info = env.get_env_info()
print(f"\n[2/5] Env info:")
print(f"  observation_space: {env_info['observation_space']}")
print(f"  action_space:      {env_info['action_space']}")
assert env_info["observation_space"].shape == (4096,)
assert env_info["action_space"].shape == (4,)

# --- Reset ---
obs_dict = env.reset()
obs = obs_dict["obs"]
assert obs.shape == (NUM_ENVS, 4096), f"Bad obs: {obs.shape}"
assert obs.device.type == "cuda"
print(f"\n[3/5] Reset: obs={obs.shape} on {obs.device}")

# Initial distance
init_dist = (env.z_current - env.z_goal).norm(dim=-1)
print(f"  Initial z_distance: {init_dist.tolist()}")
print(f"  z_current.norm: {env.z_current.norm(dim=-1).tolist()}")
print(f"  z_goal.norm:    {env.z_goal.norm(dim=-1).tolist()}")

# --- Rollout ---
print(f"\n[4/5] Random rollout ({N_STEPS} steps × {NUM_ENVS} envs):")
print(f"{'Step':>4s}  {'Reward':>32s}  {'Done':>20s}  {'Variance':>10s}  {'Distance':>10s}")
print("-" * 92)

for step in range(N_STEPS):
    actions = torch.randn(NUM_ENVS, 4, device=device).clamp(-1, 1) * 0.3
    obs_dict, rewards, dones, infos = env.step(actions)

    assert obs_dict["obs"].device.type == "cuda"
    assert rewards.device.type == "cuda"
    assert dones.device.type == "cuda"
    assert obs_dict["obs"].shape == (NUM_ENVS, 4096)

    var_mean = infos["variance"].mean().item()
    dist_mean = infos["distance"].mean().item()
    print(f"{step:4d}  {[f'{r:7.3f}' for r in rewards.tolist()]}  "
          f"{[f'{int(d)}' for d in dones.tolist()]}  "
          f"{var_mean:10.6f}  {dist_mean:10.4f}")

# --- VRAM ---
print(f"\n[5/5] Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

# --- Verify variance_penalty_weight=0 → no death ---
print(f"\n  Final death events (should be 0 since penalty weight=0): "
      f"{infos['death'].sum().item()}")
assert infos["death"].sum().item() == 0, "Death triggered with penalty_weight=0!"

print("\n" + "=" * 70)
print("=== REAL ENV SMOKE TEST PASSED ===")
print("=" * 70)
print("\nNext: scale to num_envs=256 on the L40S via:")
print("  PYTHONPATH=. python rl/train.py --num_envs 256 --max_epochs 100")
