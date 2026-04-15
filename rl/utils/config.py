from dataclasses import dataclass


@dataclass
class DreamerConfig:
    # ── World model ──────────────────────────────────────────────────
    ckpt_path: str = "outputs/pusht_cam1/checkpoints/best.ckpt"
    goal_path: str = "tests/goal_selection/z_goal.pt"
    dataset_dir: str = "data/mini/pusht/train"
    obs_key: str = "camera_1_color"
    resolution: int = 128

    # ── Latent / action ──────────────────────────────────────────────
    latent_dim: int = 4 * 32 * 32  # (C=4, H=32, W=32) → 4096
    action_dim: int = 4
    hist_context: int = 10

    # ── Imagination rollout ──────────────────────────────────────────
    imagination_horizon: int = 15  # compounding WM error ceiling; keep modest
    batch_size: int = 64           # cloud default; single 80GB A100 fits easily

    # ── Actor ────────────────────────────────────────────────────────
    actor_lr: float = 1e-4
    actor_hidden: int = 1024
    actor_layers: int = 4

    # ── Critic ───────────────────────────────────────────────────────
    critic_lr: float = 1e-4
    critic_hidden: int = 1024
    critic_layers: int = 4
    target_tau: float = 0.002

    # ── RL objective ─────────────────────────────────────────────────
    gamma: float = 0.995
    lambda_: float = 0.97
    sampling_mode: str = "hard"           # {"uniform","hard","curriculum"}
    hard_threshold_percentile: float = 50.0

    # ── Discrete actor (Gumbel-Softmax ST) ───────────────────────────
    discrete: bool = True
    gumbel_temperature: float = 1.0
    temperature_anneal: bool = True
    gumbel_temperature_end: float = 0.05
    entropy_coef: float = 0.003
    action_table_path: str = "rl/discrete_action_space.json"

    # ── Training loop ────────────────────────────────────────────────
    total_steps: int = 200000
    log_every: int = 200
    save_every: int = 10000
    grad_clip: float = 100.0
    use_gradient_checkpointing: bool = False  # ON for small-VRAM; OFF on A100

    # ── Multi-GPU ────────────────────────────────────────────────────
    device: str = "cuda:0"        # primary device; WM always lives here
    num_gpus: int = 1
    use_multi_gpu: bool = False   # if True, wrap actor/critic in DataParallel

    # ── Output ───────────────────────────────────────────────────────
    output_dir: str = "rl/outputs"
