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
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from snake_env.envs.snake_env import SnakeEnv

@dataclass
class Config:
    # Output dimensions
    output_width: int = 85 # downsample from original frame width 85 * 8
    output_height: int = 64

    # Environment
    n_channels: int = 3
    frame_stack: int = 5  # N frames to stack
    frame_width: int = 680 # output_width * 8
    frame_height: int = 512 # output_height * 8
    
    # Model architecture (adapted from ViNT)
    d_model: int = 512
    n_heads: int = 4
    n_layers: int = 4
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
    batch_size: int = 64
    mini_batch_size: int = 16
    ppo_epochs: int = 4
    n_envs: int = 1
    n_steps: int = 720
    total_timesteps: int = 1_000_000
    
    # Logging
    log_interval: int = 10
    save_interval: int = 20
    eval_interval: int = 20
    log_dir: str = "logs/tensorboard"


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
        for i, layer in enumerate(self.output_layers):
            x = layer(x)
            if i != len(self.output_layers) - 1: # Don't apply relu to the last layer
                x = F.relu(x)
        return x
    

class ImagePreprocessor(nn.Module):
    def __init__(self):
        super(ImagePreprocessor, self).__init__()
        # EfficientNet ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """
        Preprocess images for EfficientNet
        Args:
            x: tensor of shape [..., H, W, C] with values in range [0, 1] (float32)
               or [..., H, W, C] with values in range [0, 255] (uint8)
        Returns:
            tensor of shape [..., C, H, W] normalized for EfficientNet
        """
        # Handle uint8 input by converting to float32 and normalizing to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # Convert from [..., H, W, C] to [..., C, H, W]
        x = x.permute(*range(len(x.shape) - 3), -1, -3, -2)

        # Apply ImageNet normalization
        x = (x - self.mean) / self.std
        
        return x


class ViNTActorCritic(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Initialize the image preprocessor
        self.image_preprocessor = ImagePreprocessor()

        # Initialize the EfficientNet model
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', in_channels=config.n_channels)

        # Initialize the transformer backnbone
        self.transformer_decoder = MultiLayerDecoder(embed_dim=config.d_model, seq_len=config.frame_stack, output_layers=config.output_layers)

        # Adapter to go from the output of the convnet to the input d_model of the transformer
        self.adapter = nn.Linear(1280, config.d_model) # TODO: change this to the output of the convnet dynamically or smt

        # Initialize the actor and critic networks
        self.actor = nn.Linear(config.output_layers[-1], config.action_dim * 2) # 2 for sine and cosine
        self.critic = nn.Linear(config.output_layers[-1], 1)

    def forward(self, x):
        """
        x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        """
        batch_size, seq_len, frame_height, frame_width, n_channels = x.shape
        
        # Reshape to process all frames at once
        x = x.view(batch_size * seq_len, frame_height, frame_width, n_channels)
        
        # Preprocess images (converts to [batch_size * seq_len, n_channels, frame_height, frame_width] and normalizes)
        x = self.image_preprocessor(x)
        
        # Extract features using EfficientNet
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x) # [batch_size * seq_len, 1280, 1, 1]
        x = torch.flatten(x, start_dim=1) # [batch_size * seq_len, 1280]
        
        # Apply adapter
        x = self.adapter(x)
        
        # Reshape back to sequence format
        x = x.view(batch_size, seq_len, -1) # [batch_size, seq_len, d_model]
        
        # Apply transformer
        x = self.transformer_decoder(x)
        
        return self.actor(x), self.critic(x)
    
    def get_action_and_value(self, x):
        """
        x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        """
        action_logits, value = self.forward(x) # [batch_size, 4]
        
        # Split the action logits into mean and log_std
        # We interpret the spread on log scale, so we can exponentiate it and therefore guarantee positive values
        cos_mean, sin_mean, cos_log_std, sin_log_std = torch.chunk(action_logits, 4, dim=-1)
        cos_std = torch.exp(cos_log_std.clamp(-5, 2))
        sin_std = torch.exp(sin_log_std.clamp(-5, 2))
        
        # Create action distributions
        cos_action_distribution = torch.distributions.Normal(cos_mean, cos_std)
        sin_action_distribution = torch.distributions.Normal(sin_mean, sin_std)

        # Sample from the action distributions (raw actions before squashing)
        cos_action_raw = cos_action_distribution.sample()
        sin_action_raw = sin_action_distribution.sample()

        # Apply tanh squashing to ensure actions are in [-1, 1]
        cos_action = torch.tanh(cos_action_raw)
        sin_action = torch.tanh(sin_action_raw)

        # Combine the squashed actions
        action = torch.cat([cos_action, sin_action], dim=-1)

        # Calculate log probabilities with tanh correction
        # log_prob = log_prob_raw - log(1 - tanh^2(raw_action))
        sin_log_prob_raw = sin_action_distribution.log_prob(sin_action_raw).sum(dim=-1)
        cos_log_prob_raw = cos_action_distribution.log_prob(cos_action_raw).sum(dim=-1)
        
        # Tanh correction term
        sin_tanh_correction = torch.log(1 - torch.tanh(sin_action_raw)**2 + 1e-6).sum(dim=-1)
        cos_tanh_correction = torch.log(1 - torch.tanh(cos_action_raw)**2 + 1e-6).sum(dim=-1)
        
        sin_log_prob = sin_log_prob_raw - sin_tanh_correction
        cos_log_prob = cos_log_prob_raw - cos_tanh_correction
        total_log_prob = sin_log_prob + cos_log_prob

        # Calculate entropy (note: entropy changes due to tanh transformation)
        sin_entropy = sin_action_distribution.entropy().sum(dim=-1)
        cos_entropy = cos_action_distribution.entropy().sum(dim=-1)
        entropy = sin_entropy + cos_entropy

        # Calculate value
        value = value.squeeze()
            
        return action, total_log_prob, entropy, value
    
    def evaluate_actions(self, obs, actions):
       """
       obs: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
       actions: tensor of shape [batch_size, action_dim] - these are the squashed actions in [-1, 1]
       """
       logits, value = self.forward(obs)
       cos_mean, sin_mean, cos_log_std, sin_log_std = torch.chunk(logits, 4, dim=-1)
       cos_std = torch.exp(cos_log_std.clamp(-5, 2))
       sin_std = torch.exp(sin_log_std.clamp(-5, 2))
       
       # Create distributions for raw (unsquashed) actions
       cos_dist = Normal(cos_mean, cos_std)
       sin_dist = Normal(sin_mean, sin_std)
       
       # Convert squashed actions back to raw actions using inverse tanh
       cos_action_squashed = actions[:, 0:1]
       sin_action_squashed = actions[:, 1:2]
       
       # Clamp to avoid numerical issues with atanh
       cos_action_squashed = torch.clamp(cos_action_squashed, -0.999999, 0.999999)
       sin_action_squashed = torch.clamp(sin_action_squashed, -0.999999, 0.999999)
       
       cos_action_raw = torch.atanh(cos_action_squashed)
       sin_action_raw = torch.atanh(sin_action_squashed)
       
       # Calculate log probabilities for raw actions
       cos_log_prob_raw = cos_dist.log_prob(cos_action_raw).sum(-1)
       sin_log_prob_raw = sin_dist.log_prob(sin_action_raw).sum(-1)
       
       # Apply tanh correction
       cos_tanh_correction = torch.log(1 - cos_action_squashed**2 + 1e-6).sum(-1)
       sin_tanh_correction = torch.log(1 - sin_action_squashed**2 + 1e-6).sum(-1)
       
       cos_log_prob = cos_log_prob_raw - cos_tanh_correction
       sin_log_prob = sin_log_prob_raw - sin_tanh_correction
       log_prob = cos_log_prob + sin_log_prob
       
       # Calculate entropy
       cos_entropy = cos_dist.entropy().sum(-1)
       sin_entropy = sin_dist.entropy().sum(-1)
       entropy = cos_entropy + sin_entropy
       
       return log_prob, entropy, value.squeeze(-1)
    

class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, n_steps: int, n_envs: int, obs_shape: Tuple, action_dim: int, gamma: float, gae_lambda: float):
        self.n_steps = n_steps
        self.n_envs = n_envs
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Store observations as uint8 to save memory
        self.observations = torch.zeros((n_steps, n_envs, *obs_shape), dtype=torch.uint8)
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
        
        # Flatten batch dimensions and convert observations back to float32
        b_obs = self.observations.flatten(0, 1).to(device).float() / 255.0
        b_actions = self.actions.flatten(0, 1).to(device)
        b_logprobs = self.logprobs.flatten(0, 1).to(device)
        b_advantages = advantages.flatten(0, 1).to(device)
        b_returns = returns.flatten(0, 1).to(device)
        b_values = self.values.flatten(0, 1).to(device)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        return b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values


def make_env(config: Config):
    """Create environment factory"""
    def _init():
        env = gym.make('Snake-v0', screen_width=config.frame_width, screen_height=config.frame_height, zoom_level=1.0)
        env = gym.wrappers.ResizeObservation(env, (config.output_height, config.output_width))
        env = gym.wrappers.FrameStackObservation(env, config.frame_stack)
        return env
    return _init


class PPOTrainer:
    """PPO Trainer for Snake Environment"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup TensorBoard logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(config.log_dir, f"snake_ppo_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Create environments
        self.envs = [make_env(config)() for _ in range(config.n_envs)]
        
        # Initialize model
        self.model = ViNTActorCritic(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Initialize buffer
        obs_shape = (config.frame_stack, config.output_height, config.output_width, config.n_channels)
        self.buffer = RolloutBuffer(config.n_steps, config.n_envs, obs_shape, config.action_dim, config.gamma, config.gae_lambda)
        
        # Initialize tracking
        self.global_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'clip_fraction': [],
            'explained_variance': []
        }
        
    def collect_rollouts(self):
        """Collect rollouts from environments"""
        # Get initial observations
        observations = []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
        observations = torch.from_numpy(np.array(observations)) # [n_env, seq_len, height, width, n_channels]
        
        # Track episode metrics
        episode_rewards = [0] * self.config.n_envs
        episode_lengths = [0] * self.config.n_envs
        
        for step in range(self.config.n_steps):
            with torch.no_grad():
                actions, logprobs, entropy, values = self.model.get_action_and_value(observations.to(self.device))
            
            # Take actions in environments
            next_observations = []
            rewards = []
            dones = []
            
            for i, env in enumerate(self.envs):
                obs, reward, done, truncated, info = env.step(actions[i].cpu().numpy())
                
                # Track episode metrics
                episode_rewards[i] += reward
                episode_lengths[i] += 1
                
                if done or truncated:
                    # Log completed episode
                    self.episode_rewards.append(episode_rewards[i])
                    self.episode_lengths.append(episode_lengths[i])
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    obs, info = env.reset()
                
                next_observations.append(obs)
                rewards.append(reward)
                dones.append(done or truncated)
            
            # Store in buffer
            self.buffer.add(
                observations.cpu(),
                actions.cpu(),
                logprobs.cpu(),
                torch.FloatTensor(rewards),
                torch.FloatTensor(dones),
                values.cpu().squeeze()
            )
            
            observations = torch.from_numpy(np.array(next_observations))
            self.global_step += self.config.n_envs
    
    def update_policy(self):
        """Update policy using PPO"""
        b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = self.buffer.get(self.device)
        
        # Training metrics for this update
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropy_losses = []
        epoch_total_losses = []
        epoch_clip_fractions = []
        epoch_explained_variances = []
        
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
                newlogprob, entropy, newvalue = self.model.evaluate_actions(mb_obs, mb_actions)
                
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
                
                # Calculate additional metrics
                clip_fraction = (abs(ratio - 1) > self.config.clip_epsilon).float().mean()
                explained_variance = 1 - torch.var(mb_returns - newvalue.squeeze()) / torch.var(mb_returns)
                
                # Store metrics
                epoch_policy_losses.append(pg_loss.item())
                epoch_value_losses.append(v_loss.item())
                epoch_entropy_losses.append(entropy_loss.item())
                epoch_total_losses.append(loss.item())
                epoch_clip_fractions.append(clip_fraction.item())
                epoch_explained_variances.append(explained_variance.item())
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
        
        # Store average metrics for this update
        self.training_metrics['policy_loss'].append(np.mean(epoch_policy_losses))
        self.training_metrics['value_loss'].append(np.mean(epoch_value_losses))
        self.training_metrics['entropy_loss'].append(np.mean(epoch_entropy_losses))
        self.training_metrics['total_loss'].append(np.mean(epoch_total_losses))
        self.training_metrics['clip_fraction'].append(np.mean(epoch_clip_fractions))
        self.training_metrics['explained_variance'].append(np.mean(epoch_explained_variances))
    
    def log_metrics(self, iteration):
        """Log metrics to TensorBoard"""
        # Log training metrics
        if len(self.training_metrics['policy_loss']) > 0:
            self.writer.add_scalar('Loss/Policy', self.training_metrics['policy_loss'][-1], iteration)
            self.writer.add_scalar('Loss/Value', self.training_metrics['value_loss'][-1], iteration)
            self.writer.add_scalar('Loss/Entropy', self.training_metrics['entropy_loss'][-1], iteration)
            self.writer.add_scalar('Loss/Total', self.training_metrics['total_loss'][-1], iteration)
            self.writer.add_scalar('Metrics/ClipFraction', self.training_metrics['clip_fraction'][-1], iteration)
            self.writer.add_scalar('Metrics/ExplainedVariance', self.training_metrics['explained_variance'][-1], iteration)
        
        # Log episode metrics (last 100 episodes for rolling average)
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]
            recent_lengths = self.episode_lengths[-100:]
            
            self.writer.add_scalar('Episode/Reward_Mean', np.mean(recent_rewards), iteration)
            self.writer.add_scalar('Episode/Reward_Std', np.std(recent_rewards), iteration)
            self.writer.add_scalar('Episode/Reward_Max', np.max(recent_rewards), iteration)
            self.writer.add_scalar('Episode/Reward_Min', np.min(recent_rewards), iteration)
            
            self.writer.add_scalar('Episode/Length_Mean', np.mean(recent_lengths), iteration)
            self.writer.add_scalar('Episode/Length_Std', np.std(recent_lengths), iteration)
        
        # Log learning rate
        self.writer.add_scalar('Training/LearningRate', self.config.learning_rate, iteration)
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.writer.add_scalar('Model/TotalParameters', total_params, iteration)
        self.writer.add_scalar('Model/TrainableParameters', trainable_params, iteration)
        
        # Log gradients norm
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.writer.add_scalar('Training/GradientNorm', total_norm, iteration)
        
        # Flush writer
        self.writer.flush()
    
    def train(self):
        """Main training loop"""
        iteration = 0
        
        try:
            while self.global_step < self.config.total_timesteps:
                # Collect rollouts
                print(f"Collecting rollouts...")
                self.collect_rollouts()
                
                # Update policy
                print(f"Updating policy...")
                self.update_policy()
                
                iteration += 1
                
                # Logging
                if iteration % self.config.log_interval == 0:
                    print(f"Iteration {iteration}, Global Step {self.global_step}")
                    self.log_metrics(iteration)
                
                # Save model
                if iteration % self.config.save_interval == 0:
                    torch.save(self.model.state_dict(), f"snake_ppo_model_{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        
        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            # Close TensorBoard writer
            self.writer.close()
            print("TensorBoard logging closed")

def main():
    parser = argparse.ArgumentParser(description="PPO Training for Snake Environment")
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--frame_stack", type=int, default=5)
    parser.add_argument("--log_dir", type=str, default="logs/tensorboard")
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.total_timesteps = args.total_timesteps
    config.learning_rate = args.learning_rate
    config.n_envs = args.n_envs
    config.frame_stack = args.frame_stack
    config.log_dir = args.log_dir
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    # Start training
    print("Starting PPO training...")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"TensorBoard logs will be saved to: {os.path.join(config.log_dir, 'snake_ppo_*')}")
    trainer.train()


if __name__ == "__main__":
    main()