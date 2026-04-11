from dataclasses import dataclass


@dataclass
class DreamerConfig:
    # World model
    ckpt_path: str = "outputs/pusht_cam1/checkpoints/best.ckpt"
    goal_path: str = "tests/goal_selection/z_goal.pt"
    dataset_dir: str = "data/mini/pusht/train"
    obs_key: str = "camera_1_color"
    resolution: int = 128

    # Latent
    latent_channels: int = 4
    latent_h: int = 32
    latent_w: int = 32
    latent_dim: int = 4 * 32 * 32  # 4096
    action_dim: int = 4
    hist_context: int = 10

    # Imagination
    imagination_horizon: int = 5
    batch_size: int = 4

    # Actor
    actor_lr: float = 3e-4
    actor_hidden: int = 512
    actor_layers: int = 3

    # Critic
    critic_lr: float = 3e-4
    critic_hidden: int = 512
    critic_layers: int = 3
    target_tau: float = 0.005

    # RL
    gamma: float = 0.99
    lambda_: float = 0.95

    # Training
    total_steps: int = 2000
    log_every: int = 100
    eval_every: int = 500
    save_every: int = 1000
    grad_clip: float = 100.0
    device: str = "cuda:0"

    # Output
    output_dir: str = "rl/outputs"
