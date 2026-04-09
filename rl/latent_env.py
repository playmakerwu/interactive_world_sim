"""Pure-GPU vectorized latent env for rl-games — REAL LatentWorldModel ensemble.

Hard-switch from the mock pipeline. No MockDynamicsModel anywhere.

Architecture:
    Real ensemble = N x LatentWorldModel from outputs/universe/*.ckpt
    Real dataset  = pre-encoded latents from data/.../pusht_latent/

    obs   : (B, 4096) — z_goal - z_current (goal-relative observation)
    action: (B, 4)    — normalized [-1, 1] (matches model.normalizer["action"] range)
    z     : (B, 4096) flat ↔ (B, 4, 32, 32) for the model

Reward: delta_l2 by default. Variance penalty is GATED by variance_penalty_weight
        (default 0.0 — pure L2 smoke test before turning on death penalty).
"""

import glob
import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from omegaconf import OmegaConf

# Register OmegaConf resolvers needed by LatentWorldModel's config (idempotent)
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
if not OmegaConf.has_resolver("torch"):
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))

from interactive_world_sim.algorithms.latent_dynamics import LatentWorldModel


# ============================================================
# Config
# ============================================================

@dataclass
class LatentEnvConfig:
    """All env hyperparameters."""

    # --- Real model paths ---
    ensemble_dir: str = "outputs/universe"
    ae_ckpt: str = "outputs/pusht_cam1/checkpoints/best.ckpt"

    # --- Real pre-encoded dataset ---
    dataset_dir: str = "data/mini/pusht_latent"

    # --- Hardcoded dimensions for real PushT (camera_1_color) ---
    latent_dim: int = 4096        # = 4 * 32 * 32
    action_dim: int = 4

    # --- Episode ---
    chunk_size: int = 30
    max_steps: int = 30

    # --- Reward shaping ---
    distance_mode: str = "delta_l2"   # "delta_l2" | "l2" | "cosine" | "normalized_l2"
    reward_scale: float = 1.0
    success_threshold: float = 0.5
    success_reward: float = 100.0
    step_penalty: float = 0.0

    # --- Variance / death penalty (GATED by variance_penalty_weight) ---
    variance_threshold: float = 0.5
    variance_penalty_weight: float = 0.0   # 0 = disabled (smoke test); 1 = full
    death_penalty: float = -100.0

    # --- Device ---
    device: str = "cuda"

    # --- Telemetry ---
    tb_log_dir: Optional[str] = None
    tb_log_interval: int = 100


# ============================================================
# Real Latent World Model Wrapper
# ============================================================

class RealLatentWorldModelWrapper(nn.Module):
    """Loads N trained LatentWorldModels from outputs/universe/ and exposes
    a single step(z_flat, action) → (z_next_mean_flat, variance) interface.

    - Input/output latents are FLAT (B, 4096).
    - Internally reshapes to (B, 1, 4, 32, 32) for dynamics_forward.
    - All non-dynamics submodules (decoder, FVD, FID, LPIPS) are stripped
      after loading to free VRAM.
    - prev_action buffer maintains the 1-step history that
      dynamics_forward needs for its action sequence convention.
    """

    LATENT_C: int = 4
    LATENT_H: int = 32
    LATENT_W: int = 32

    @property
    def LATENT_DIM_FLAT(self) -> int:
        return self.LATENT_C * self.LATENT_H * self.LATENT_W   # 4096

    def __init__(self, ensemble_dir: str, ae_ckpt: str, device: str = "cuda"):
        super().__init__()

        # Load base config from AE checkpoint (used to instantiate each LatentWorldModel)
        ae_dir = os.path.dirname(os.path.dirname(ae_ckpt))
        cfg_path = os.path.join(ae_dir, ".hydra", "config.yaml")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"AE config not found: {cfg_path}")
        base_cfg = OmegaConf.load(cfg_path)
        base_cfg.algorithm.training_stage = 2
        base_cfg.algorithm.load_ae = None
        base_cfg.algorithm.use_prebaked_latent = True

        # Find all ensemble .ckpt files
        ckpt_paths = sorted(glob.glob(os.path.join(ensemble_dir, "*.ckpt")))
        if not ckpt_paths:
            raise FileNotFoundError(f"No .ckpt files in {ensemble_dir}")

        models = []
        for cp in ckpt_paths:
            try:
                m = LatentWorldModel.load_from_checkpoint(
                    cp, cfg=base_cfg.algorithm, map_location=device,
                    weights_only=False, strict=True,
                )
            except RuntimeError:
                m = LatentWorldModel.load_from_checkpoint(
                    cp, cfg=base_cfg.algorithm, map_location=device,
                    weights_only=False, strict=False,
                )

            m.eval()
            for p in m.parameters():
                p.requires_grad = False

            # Strip submodules unused at env-step time → free VRAM
            if hasattr(m, "decoder"):
                m.decoder = nn.Identity()
            if hasattr(m, "validation_fvd_model"):
                m.validation_fvd_model = None
            if hasattr(m, "validation_fid_model"):
                m.validation_fid_model = None
            if hasattr(m, "validation_lpips_model"):
                m.validation_lpips_model = None

            models.append(m)
            print(f"  [RealLatentWorldModelWrapper] Loaded {os.path.basename(cp)}")

        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        torch.cuda.empty_cache()

        n_dyn_params = sum(p.numel() for m in self.models for p in m.dynamics.parameters())
        print(f"  [RealLatentWorldModelWrapper] Ensemble size: {self.n_models}")
        print(f"  [RealLatentWorldModelWrapper] Total dynamics params: {n_dyn_params:,}")

        # Action history (initialized lazily on first step)
        self.prev_action: Optional[torch.Tensor] = None

    def reset_history(self, env_mask: torch.Tensor) -> None:
        """Zero out action history for envs that just reset."""
        if self.prev_action is not None:
            self.prev_action = torch.where(
                env_mask.unsqueeze(-1),
                torch.zeros_like(self.prev_action),
                self.prev_action,
            )

    @torch.no_grad()
    def step(
        self, z_flat: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One env step through ALL ensemble models.

        Args:
            z_flat: (B, 4096) flat current latent
            action: (B, 4)    normalized [-1, 1] action

        Returns:
            z_next_flat: (B, 4096) ensemble MEAN next latent
            variance:    (B,)     per-env ensemble variance (mean over latent dims)
        """
        B = z_flat.shape[0]
        device = z_flat.device

        # Reshape (B, 4096) → (B, 1, 4, 32, 32)
        z = z_flat.view(B, 1, self.LATENT_C, self.LATENT_H, self.LATENT_W)

        # dynamics_forward needs action shape (B, T_hist + T_act, A) = (B, 2, 4)
        if self.prev_action is None or self.prev_action.shape[0] != B:
            self.prev_action = torch.zeros(B, action.shape[1], device=device, dtype=action.dtype)
        action_seq = torch.stack([self.prev_action, action], dim=1)  # (B, 2, A)

        # Forward through each model — sequential (3D conv ops are heavy)
        preds = []
        for m in self.models:
            z_pred = m.dynamics_forward(z, action_seq)  # (B, 1, 4, 32, 32)
            preds.append(z_pred[:, 0])                  # (B, 4, 32, 32)

        # Ensemble aggregation
        preds_stack = torch.stack(preds, dim=0)         # (N, B, 4, 32, 32)
        z_next = preds_stack.mean(dim=0)                # (B, 4, 32, 32)

        if self.n_models >= 2:
            variance = preds_stack.var(dim=0).flatten(1).mean(dim=-1)  # (B,)
        else:
            variance = torch.zeros(B, device=device)

        # Update history for next call
        self.prev_action = action.clone()

        return z_next.flatten(1), variance


# ============================================================
# Real Latent Dataset — loads pre-encoded LatentWorldModel latents
# ============================================================

class RealLatentDataset:
    """Loads pre-encoded latents from disk into a single contiguous GPU tensor.

    Expected directory structure:
        dataset_dir/
            train/
                metadata.pt    {"episode_ends": np.int64, "n_episodes": int, ...}
                episode_0.pt   {"latent": (T, 4, 32, 32), "action": (T, 4)}
                episode_1.pt
                ...
    """

    def __init__(self, dataset_dir: str, device: str = "cuda"):
        train_dir = os.path.join(dataset_dir, "train")
        meta_path = os.path.join(train_dir, "metadata.pt")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Missing {meta_path} — pre-encode pusht latents first via "
                f"scripts/pre_encode_latents.py"
            )

        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        n_eps = int(meta["n_episodes"])

        latents_list, actions_list, lengths = [], [], []
        for i in range(n_eps):
            ep = torch.load(
                os.path.join(train_dir, f"episode_{i}.pt"),
                map_location="cpu", weights_only=False,
            )
            lat = ep["latent"].flatten(1)   # (T, 4, 32, 32) → (T, 4096)
            act = ep["action"]              # (T, 4)
            latents_list.append(lat)
            actions_list.append(act)
            lengths.append(lat.shape[0])

        # Concatenate to single GPU tensors
        self.latents = torch.cat(latents_list, dim=0).to(device)   # (N_total, 4096)
        self.actions = torch.cat(actions_list, dim=0).to(device)   # (N_total, A)

        # Episode boundary info (for chunk sampling)
        self.episode_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.episode_starts = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(self.episode_lengths, dim=0)[:-1],
        ])

        self.n_episodes = n_eps
        self.latent_dim = int(self.latents.shape[-1])
        self.action_dim = int(self.actions.shape[-1])

        print(f"  [RealLatentDataset] Loaded {n_eps} episodes from {dataset_dir}")
        print(f"    total frames: {self.latents.shape[0]:,}")
        print(f"    latent_dim:   {self.latent_dim}  (expect 4096)")
        print(f"    action_dim:   {self.action_dim}  (expect 4)")

    def sample_chunks(
        self, num_envs: int, chunk_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (z_start, z_goal, expert_actions) — respects episode boundaries.

        Returns:
            z_start:        (num_envs, 4096)
            z_goal:         (num_envs, 4096)
            expert_actions: (num_envs, chunk_size, action_dim)
        """
        device = self.latents.device

        # Pick random episodes
        traj_idx = torch.randint(0, self.n_episodes, (num_envs,), device=device)
        ep_start = self.episode_starts[traj_idx]
        ep_len = self.episode_lengths[traj_idx]

        # Sample valid offset within each episode (chunk must fit)
        max_offset = (ep_len - chunk_size - 1).clamp(min=0)
        rand = torch.rand(num_envs, device=device)
        offset = (rand * (max_offset + 1).float()).long().clamp(max=max_offset)

        start_global = ep_start + offset
        goal_global = ep_start + offset + chunk_size

        z_start = self.latents[start_global]
        z_goal = self.latents[goal_global]

        # Vectorized expert action gather
        time_offsets = torch.arange(chunk_size, device=device).unsqueeze(0)
        time_idx_global = start_global.unsqueeze(1) + time_offsets   # (num_envs, chunk_size)
        expert_actions = self.actions[time_idx_global]               # (num_envs, chunk_size, A)

        return z_start, z_goal, expert_actions


# ============================================================
# Reward Computer
# ============================================================

class RewardComputer:
    """Goal-conditioned reward + GATED variance penalty. Pure GPU."""

    def __init__(self, cfg: LatentEnvConfig):
        self.mode = cfg.distance_mode
        self.reward_scale = cfg.reward_scale
        self.success_threshold = cfg.success_threshold
        self.success_reward = cfg.success_reward
        self.step_penalty = cfg.step_penalty
        self.variance_threshold = cfg.variance_threshold
        self.variance_penalty_weight = cfg.variance_penalty_weight
        self.death_penalty = cfg.death_penalty
        self.prev_distance: Optional[torch.Tensor] = None

    def _compute_distance(self, z_current: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        if self.mode in ("l2", "delta_l2"):
            return torch.norm(z_current - z_goal, dim=-1)
        if self.mode == "cosine":
            return 1.0 - F.cosine_similarity(z_current, z_goal, dim=-1)
        if self.mode == "normalized_l2":
            zc = F.normalize(z_current, dim=-1)
            zg = F.normalize(z_goal, dim=-1)
            return torch.norm(zc - zg, dim=-1)
        raise ValueError(f"Unknown distance_mode: {self.mode}")

    def compute(
        self, z_current: torch.Tensor, z_goal: torch.Tensor, variance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        distance = self._compute_distance(z_current, z_goal)

        if self.mode == "delta_l2":
            if self.prev_distance is None or self.prev_distance.shape != distance.shape:
                self.prev_distance = distance.clone()
            reward_distance = (self.prev_distance - distance) * self.reward_scale
            self.prev_distance = distance.clone()
        else:
            reward_distance = -distance * self.reward_scale

        reward = reward_distance + self.step_penalty

        # Success bonus
        success = distance < self.success_threshold
        reward = torch.where(success, torch.full_like(reward, self.success_reward), reward)

        # Variance death penalty — GATED
        if self.variance_penalty_weight > 0:
            death = variance > self.variance_threshold
            penalty = self.death_penalty * self.variance_penalty_weight
            reward = torch.where(death, torch.full_like(reward, penalty), reward)
        else:
            death = torch.zeros_like(success)

        info = {
            "reward_distance": reward_distance,
            "distance": distance,
            "variance": variance,
            "success": success,
            "death": death,
        }
        return reward, success, death, info


# ============================================================
# Main Vectorized Env (rl-games IVecEnv interface)
# ============================================================

class LatentPushTForRLGames:
    """Pure-GPU vectorized env wrapping the real LatentWorldModel ensemble.

    Observation: 4096-d goal-relative latent error (z_goal - z_current).
    Action:      4-d normalized [-1, 1].
    """

    def __init__(self, cfg: LatentEnvConfig, num_envs: int):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(cfg.device)

        # --- Real dataset ---
        self.dataset = RealLatentDataset(cfg.dataset_dir, cfg.device)
        assert self.dataset.latent_dim == cfg.latent_dim, (
            f"Dataset latent_dim={self.dataset.latent_dim} != cfg.latent_dim={cfg.latent_dim}"
        )
        assert self.dataset.action_dim == cfg.action_dim, (
            f"Dataset action_dim={self.dataset.action_dim} != cfg.action_dim={cfg.action_dim}"
        )

        # --- Real ensemble ---
        self.ensemble = RealLatentWorldModelWrapper(
            ensemble_dir=cfg.ensemble_dir,
            ae_ckpt=cfg.ae_ckpt,
            device=cfg.device,
        ).to(self.device)

        # --- Reward ---
        self.reward_computer = RewardComputer(cfg)

        # --- Env state buffers (all on GPU) ---
        self.z_current = torch.zeros(num_envs, cfg.latent_dim, device=self.device)
        self.z_goal = torch.zeros(num_envs, cfg.latent_dim, device=self.device)
        self.current_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(num_envs, device=self.device)
        self.episode_length = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # --- Spaces (HARDCODED to 4096-d obs) ---
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.latent_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(cfg.action_dim,), dtype=np.float32
        )

        # --- Telemetry ---
        self.global_step = 0
        self.tb_writer = None
        if cfg.tb_log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(cfg.tb_log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=cfg.tb_log_dir)
            print(f"  [LatentEnv] Telemetry → {cfg.tb_log_dir}")

    # --- Helpers ---

    def _build_obs(self) -> Dict[str, torch.Tensor]:
        """obs = (z_goal - z_current) — 4096-d goal-relative error vector."""
        return {"obs": self.z_goal - self.z_current}

    def _reset_envs_async(self, env_mask: torch.Tensor) -> None:
        """Reset masked envs without GPU→CPU sync."""
        z_start, z_goal, _ = self.dataset.sample_chunks(self.num_envs, self.cfg.chunk_size)
        mask_2d = env_mask.unsqueeze(-1)
        self.z_current = torch.where(mask_2d, z_start, self.z_current)
        self.z_goal = torch.where(mask_2d, z_goal, self.z_goal)
        self.current_step = torch.where(env_mask, torch.zeros_like(self.current_step), self.current_step)
        self.episode_return = torch.where(env_mask, torch.zeros_like(self.episode_return), self.episode_return)
        self.episode_length = torch.where(env_mask, torch.zeros_like(self.episode_length), self.episode_length)

        if self.reward_computer.prev_distance is not None:
            new_dist = self.reward_computer._compute_distance(z_start, z_goal)
            self.reward_computer.prev_distance = torch.where(
                env_mask, new_dist, self.reward_computer.prev_distance
            )

        self.ensemble.reset_history(env_mask)

    # --- IVecEnv interface ---

    def reset(self) -> Dict[str, torch.Tensor]:
        all_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_envs_async(all_mask)
        return self._build_obs()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        if actions.device != self.device:
            actions = actions.to(self.device)
        actions_clamped = actions.clamp(-1.0, 1.0)

        # Real dynamics step
        z_next, variance = self.ensemble.step(self.z_current, actions_clamped)
        self.z_current = z_next
        self.current_step += 1

        # Reward
        reward, success, death, info = self.reward_computer.compute(
            self.z_current, self.z_goal, variance
        )

        self.episode_return = self.episode_return + reward
        self.episode_length = self.episode_length + 1

        timeout = self.current_step >= self.cfg.max_steps
        dones = (success | death | timeout).float()
        info["time_outs"] = timeout & ~success & ~death

        # Telemetry (only fires every N steps)
        self.global_step += 1
        if self.tb_writer is not None and self.global_step % self.cfg.tb_log_interval == 0:
            self._log_telemetry_sync(actions, actions_clamped, reward, variance, info)

        # Auto-reset done envs (no .any() sync — _reset_envs_async handles empty mask via where)
        self._reset_envs_async(dones.bool())

        return self._build_obs(), reward, dones, info

    def get_number_of_agents(self) -> int:
        return 1

    def get_env_info(self) -> dict:
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": 1,
            "value_size": 1,
        }

    # --- Telemetry ---

    def _log_telemetry_sync(
        self,
        actions_raw: torch.Tensor,
        actions_clamped: torch.Tensor,
        reward: torch.Tensor,
        variance: torch.Tensor,
        info: dict,
    ) -> None:
        """Single batched GPU→CPU sync for all telemetry metrics."""
        gs = self.global_step
        w = self.tb_writer
        with torch.no_grad():
            metrics = {
                "action/norm_mean":     actions_clamped.norm(dim=-1).mean(),
                "action/mean_abs":      actions_clamped.abs().mean(),
                "action/std_per_dim":   actions_clamped.std(dim=0).mean(),
                "action/max_abs":       actions_clamped.abs().max(),
                "action/raw_norm":      actions_raw.norm(dim=-1).mean(),
                "action/clip_fraction": (actions_raw.abs() > 1.0).float().mean(),
                "latent/z_current_norm":  self.z_current.norm(dim=-1).mean(),
                "latent/z_current_std":   self.z_current.std(dim=-1).mean(),
                "latent/z_goal_norm":     self.z_goal.norm(dim=-1).mean(),
                "latent/z_distance_l2":   info["distance"].mean(),
                "latent/z_distance_max":  info["distance"].max(),
                "latent/z_distance_min":  info["distance"].min(),
                "reward/total_mean":          reward.mean(),
                "reward/total_std":           reward.std(),
                "reward/distance_component":  info["reward_distance"].mean(),
                "reward/raw_distance_mean":   info["distance"].mean(),
                "variance/mean":  variance.mean(),
                "variance/max":   variance.max(),
                "variance/p99":   variance.quantile(0.99),
                "term/success_rate": info["success"].float().mean(),
                "term/death_rate":   info["death"].float().mean(),
            }
            metrics_cpu = {k: v.item() for k, v in metrics.items()}
        for k, v in metrics_cpu.items():
            w.add_scalar(k, v, gs)
