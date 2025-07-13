import argparse
from dataclasses import dataclass, field

@dataclass
class Config:
    # Output dimensions
    output_width: int = 85 # downsample from original frame width 85 * 8
    output_height: int = 64

    # Environment
    n_channels: int = 3
    frame_stack: int = 5  # N frames to stack
    frame_skip: int = 4  # Repeat each action for this many frames
    frame_width: int = 680 # output_width * 8
    frame_height: int = 512 # output_height * 8
    
    # Model architecture (adapted from ViNT)
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    output_layers: list[int] = field(default_factory=lambda: [256, 128, 64, 32])
    
    # Action space
    action_dim: int = 2  # 2 for sine (mean, std), 2 for cosine (mean, std)
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    mini_batch_size: int = 64
    ppo_epochs: int = 4
    n_envs: int = 1
    n_steps: int = 8_192
    total_timesteps: int = 1_000_000
    
    # Logging
    log_interval: int = 2
    save_interval: int = 20
    eval_interval: int = 20
    log_dir: str = "logs/tensorboard"