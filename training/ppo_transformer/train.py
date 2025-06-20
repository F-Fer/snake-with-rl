import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gym
from collections import namedtuple, deque
import math
import random
from typing import Tuple, Optional
from efficientnet_pytorch import EfficientNet

import argparse
import os
import wandb
from dataclasses import dataclass

@dataclass
class Config:
    # Environment
    n_channels: int = 3
    frame_stack: int = 5  # N frames to stack
    frame_width: int = 720
    frame_height: int = 480
    
    # Model architecture (adapted from ViNT)
    d_model: int = 512
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 2048
    dropout: float = 0.1
    output_layers: list[int] = [256, 128, 64, 32]
    
    # Action space
    action_dim: int = 4  # 2 for sine (mean, std), 2 for cosine (mean, std)
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    batch_size: int = 64
    mini_batch_size: int = 16
    ppo_epochs: int = 4
    n_envs: int = 8
    n_steps: int = 128
    total_timesteps: int = 1000000
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer, copied from ViNT: 
    https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/self_attention.py"""

    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x
    

class MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        # currently, x is [batch_size, seq_len, embed_dim]
        x = x.reshape(x.shape[0], -1) # Now x is [batch_size, seq_len * embed_dim]
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x
    

class ViNTActorCritic(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Initialize the EfficientNet model
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', in_channels=config.n_channels)

        # Initialize the transformer backnbone
        self.transformer_decoder = MultiLayerDecoder(embed_dim=config.d_model, seq_len=config.frame_stack, output_layers=config.output_layers)

        # Initialize the actor and critic networks
        self.actor = nn.Linear(config.output_layers[-1], config.action_dim * 2) # 2 for sine and cosine
        self.critic = nn.Linear(config.output_layers[-1], 1)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.transformer_decoder(x)
        return self.actor(x), self.critic(x)
    

class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, n_steps: int, n_envs: int, obs_shape: Tuple, action_dim: int, gamma: float, gae_lambda: float):
        self.n_steps = n_steps
        self.n_envs = n_envs
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = torch.zeros((n_steps, n_envs, *obs_shape))
        self.actions = torch.zeros((n_steps, n_envs, action_dim))
        self.logprobs = torch.zeros((n_steps, n_envs))
        self.rewards = torch.zeros((n_steps, n_envs))
        self.dones = torch.zeros((n_steps, n_envs))
        self.values = torch.zeros((n_steps, n_envs))
        
        self.ptr = 0
        self.full = False
    
    def add(self, obs, action, logprob, reward, done, value):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.n_steps
        if self.ptr == 0:
            self.full = True
    
    def get(self, device):
        """Get all data and compute advantages"""
        # Calculate advantages using GAE
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = 0  # Assuming episode ends
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_gae
            last_gae = advantages[t]
        
        returns = advantages + self.values
        
        # Flatten batch dimensions
        b_obs = self.observations.flatten(0, 1).to(device)
        b_actions = self.actions.flatten(0, 1).to(device)
        b_logprobs = self.logprobs.flatten(0, 1).to(device)
        b_advantages = advantages.flatten(0, 1).to(device)
        b_returns = returns.flatten(0, 1).to(device)
        b_values = self.values.flatten(0, 1).to(device)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        return b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values