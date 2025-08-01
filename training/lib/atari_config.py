import argparse
from dataclasses import dataclass, field

@dataclass
class Config:
    # Output dimensions
    output_width: int = 112 # downsample from original 
    output_height: int = 112

    # Environment
    n_channels: int = 1
    frame_stack: int = 5  # N frames to stack
    frame_skip: int = 4  # Repeat each action for this many frames
    frame_width: int = 448 # output_width * 8
    frame_height: int = 448 # output_height * 8
    
    # Model architecture (adapted from ViNT)
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    output_layers: list[int] = field(default_factory=lambda: [256, 128, 64, 32])
    start_log_std: float = 3.0
    
    # Action space
    action_dim: int = 2  # 2 for sine (mean, std), 2 for cosine (mean, std)
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1
    clip_vloss: bool = True
    value_coef: float = 0.5
    entropy_coef: float = 0.04
    entropy_coef_anneal: bool = True
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    
    # Training
    mini_batch_size: int = 512
    ppo_epochs: int = 15
    n_envs: int = 16
    n_steps: int = 1_024
    total_timesteps: int = 2_000_000
    batch_size: int = int(n_steps * n_envs)
    
    # Logging
    save_interval: int = 250 # updates
    save_dir: str = "models"
    log_dir: str = "logs"

    # Seeding
    seed: int = 42
    torch_deterministic: bool = True

    # GPU
    cuda: bool = True

    # Video recording
    record_video: bool = True

    # Target KL
    target_kl: float = None

    # Random action reset
    max_random_steps: int = 100
    random_action_reset: bool = True

    # Gray scale
    gray_scale: bool = True

    # Snake env
    world_size: int = 3000
    num_bots: int = 8
    num_foods: int = 150
