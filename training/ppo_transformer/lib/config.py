import argparse
from dataclasses import dataclass, field

@dataclass
class Config:
    # Output dimensions
    output_width: int = 224 # downsample from original frame width 85 * 8
    output_height: int = 224

    # Environment
    n_channels: int = 3
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
    
    # Action space
    action_dim: int = 2  # 2 for sine (mean, std), 2 for cosine (mean, std)
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    clip_vloss: bool = True
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    
    # Training
    mini_batch_size: int = 2_048
    ppo_epochs: int = 10
    n_envs: int = 2
    n_steps: int = 128
    total_timesteps: int = 1_000_000
    batch_size: int = int(n_steps * n_envs)
    
    # Logging
    log_interval: int = 2
    save_interval: int = 20
    eval_interval: int = 20
    log_dir: str = "logs/tensorboard"

    # Seeding
    seed: int = 42
    torch_deterministic: bool = True

    # GPU
    cuda: bool = True

    # Video recording
    record_video: bool = True

    # Target KL
    target_kl: float = None
