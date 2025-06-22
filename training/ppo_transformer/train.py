import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
from collections import namedtuple, deque
import math
import random
from typing import Tuple, Optional
from efficientnet_pytorch import EfficientNet
import argparse
import os
import wandb
from dataclasses import dataclass

from src.snake_env.envs.snake_env import SnakeEnv

@dataclass
class Config:
    # Output dimensions
    output_width: int = 85 # downsample from original frame width 85 * 8
    output_height: int = 64

    # Environment
    n_channels: int = 3
    frame_stack: int = 5  # N frames to stack
    frame_width: int = output_width * 8
    frame_height: int = output_height * 8
    
    # Model architecture (adapted from ViNT)
    d_model: int = 512
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 2048
    dropout: float = 0.1
    output_layers: list[int] = [256, 128, 64, 32]
    
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
        """
        x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        """
        x = x.permute(0, 1, 4, 2, 3) # [batch_size, seq_len, n_channels, frame_height, frame_width]
        batch_size, seq_len, n_channels, frame_height, frame_width = x.shape
        x = x.view(batch_size, seq_len * n_channels, frame_height, frame_width) # [batch_size, seq_len * n_channels, frame_height, frame_width]
        x = self.efficientnet(x)
        x = self.transformer_decoder(x)
        return self.actor(x), self.critic(x)
    
    def get_action_and_value(self, x, use_mean=False):
        """
        x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        use_mean: if True, use the mean of the action distribution instead of sampling from it
        """
        action_logits, value = self.forward(x) # [batch_size, 4]
        
        # Split the action logits into mean and log_std
        # We interpret the spread on log scale, so we can exponentiate it and therefore guarantee positive values
        mean, log_std = action_logits.split(self.config.action_dim, dim=-1) 
        std = torch.exp(log_std)
        
        if not use_mean:
            # Sample from the normal distribution
            action = torch.normal(mean, std)
        else:
            # Use the mean directly
            action = mean
            
        return action, value
    

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
        last_advantage = 0
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = 0  # Assuming episode ends
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + self.values
        
        # Flatten batch dimensions
        b_obs = self.observations.flatten(0, 1).to(device)
        b_actions = self.actions.flatten(0, 1).to(device)
        b_advantages = advantages.flatten(0, 1).to(device)
        b_returns = returns.flatten(0, 1).to(device)
        b_values = self.values.flatten(0, 1).to(device)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        return b_obs, b_actions, b_advantages, b_returns, b_values


def make_env(config: Config):
    """Create environment factory"""
    def _init():
        env = gym.make('Snake-v0', screen_width=config.frame_width, screen_height=config.frame_height, zoom_level=1.0)
        env = gym.wrappers.ResizeObservation(env, (config.frame_height, config.frame_width))
        return env
    return _init


class PPOTrainer:
    """PPO Trainer for Snake Environment"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create environments
        self.envs = [make_env(config)() for _ in range(config.n_envs)]
        
        # Initialize model
        self.model = ViNTActorCritic(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Initialize buffer
        obs_shape = (config.frame_stack, 3, config.frame_height, config.frame_width)
        self.buffer = RolloutBuffer(config.n_steps, config.n_envs, obs_shape, config.action_dim)
        
        # Initialize tracking
        self.global_step = 0
        self.episode_rewards = []
        
    def collect_rollouts(self):
        """Collect rollouts from environments"""
        # Get initial observations
        observations = []
        for env in self.envs:
            obs = env.reset()
            observations.append(obs)
        
        observations = torch.FloatTensor(np.array(observations)).to(self.device) # [n_env, height, width, n_channels]
        
        for step in range(self.config.n_steps):
            with torch.no_grad():
                actions, values = self.model.get_action_and_value(observations)
            
            # Take actions in environments
            next_observations = []
            rewards = []
            dones = []
            
            for i, env in enumerate(self.envs):
                obs, reward, done, info = env.step(actions[i].cpu().numpy())
                
                if done:
                    obs = env.reset()
                
                next_observations.append(obs)
                rewards.append(reward)
                dones.append(done)
            
            # Store in buffer
            self.buffer.add(
                observations.cpu(),
                actions.cpu(),
                torch.FloatTensor(rewards),
                torch.FloatTensor(dones),
                values.cpu().squeeze()
            )
            
            observations = torch.FloatTensor(np.array(next_observations)).to(self.device)
            self.global_step += self.config.n_envs
    
    def update_policy(self):
        """Update policy using PPO"""
        b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = self.buffer.get(self.device)
        
        # Training loop
        for epoch in range(self.config.ppo_epochs):
            # Create mini-batches
            batch_size = len(b_obs)
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                mb_indices = indices[start:end]
                
                # Get mini-batch data
                mb_obs = b_obs[mb_indices]
                mb_actions = b_actions[mb_indices] 
                mb_logprobs = b_logprobs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_values = b_values[mb_indices]
                
                # Forward pass
                _, newlogprob, entropy, newvalue = self.model.get_action_and_value(mb_obs, mb_actions)
                
                # Calculate losses
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                                       1 + self.config.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = ((newvalue.squeeze() - mb_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss + self.config.value_coef * v_loss - self.config.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
    
    def train(self):
        """Main training loop"""
        iteration = 0
        
        while self.global_step < self.config.total_timesteps:
            # Collect rollouts
            self.collect_rollouts()
            
            # Update policy
            self.update_policy()
            
            iteration += 1
            
            # Logging
            if iteration % self.config.log_interval == 0:
                print(f"Iteration {iteration}, Global Step {self.global_step}")
            
            # Save model
            if iteration % self.config.save_interval == 0:
                torch.save(self.model.state_dict(), f"snake_ppo_model_{iteration}.pth")

def main():
    parser = argparse.ArgumentParser(description="PPO Training for Snake Environment")
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--frame_stack", type=int, default=5)
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.total_timesteps = args.total_timesteps
    config.learning_rate = args.learning_rate
    config.n_envs = args.n_envs
    config.frame_stack = args.frame_stack
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    # Start training
    print("Starting PPO training with ViNT-inspired Transformer architecture...")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    trainer.train()