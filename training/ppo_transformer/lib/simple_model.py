import numpy as np
import torch
import math
import torch.nn as nn
import gymnasium as gym
from training.ppo_transformer.lib.config import Config

HID_SIZE = 256


def calculate_conv_output_size(input_size, kernel_size, stride, padding=0):
    """Calculate output size after convolution"""
    return (input_size + 2 * padding - kernel_size) // stride + 1


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        x = self.relu(x_in)
        x = self.relu(self.conv1(x))
        x_out = self.conv2(x)
        return x_in + x_out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, block_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=block_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResBlock(block_channels)
        self.res2 = ResBlock(block_channels)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.max_pool(x)
        x = self.res1(x)
        return self.res2(x)
    

class ConvNet(nn.Module):
    def __init__(self, in_channels, base_channels=16):
        super(ConvNet, self).__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, block_channels=base_channels)
        self.conv2 = ConvBlock(in_channels=base_channels, block_channels=base_channels * 2)
        self.conv3 = ConvBlock(in_channels=base_channels * 2, block_channels=base_channels * 2)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return x
    
    
class ModelResidual(nn.Module):
    def __init__(self, channels_in: int, num_outputs: int, hidden_size=HID_SIZE, base_channels=16):
        super(ModelResidual, self).__init__()
        self.conv_net = ConvNet(in_channels=channels_in, base_channels=base_channels)
        self.linear = nn.LazyLinear(hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert uint8 to float32 and normalize to [0, 1] range
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        x = self.relu(self.conv_net(x))
        x = self.relu(self.linear(x))
        x = self.linear2(x)
        return x
    

class SimpleImagePreprocessor(nn.Module):
    def __init__(self):
        super(SimpleImagePreprocessor, self).__init__()

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

        batch_size, seq_len, frame_height, frame_width, n_channels = x.shape

        # Rearrange so that sequence length and channel dims are adjacent, then stack
        # Resulting shape: [batch_size, seq_len * n_channels, frame_height, frame_width]
        x = x.permute(0, 1, 4, 2, 3).reshape(batch_size, seq_len * n_channels, frame_height, frame_width)
        
        return x
        

class SimpleModel(nn.Module):
    def __init__(self, config: Config):
        super(SimpleModel, self).__init__()
        self.config = config

        self.image_preprocessor = SimpleImagePreprocessor()

        # Calculate the size after convolutions
        # Input: (C, H, W) = (1, 84, 84) for grayscale
        c, h, w = config.n_channels * config.frame_stack, config.output_height, config.output_width
        
        # First conv: kernel=8, stride=4
        h1 = calculate_conv_output_size(h, 8, 4)
        w1 = calculate_conv_output_size(w, 8, 4)
        
        # Second conv: kernel=4, stride=2  
        h2 = calculate_conv_output_size(h1, 4, 2)
        w2 = calculate_conv_output_size(w1, 4, 2)
        
        # Third conv: kernel=3, stride=1
        h3 = calculate_conv_output_size(h2, 3, 1)
        w3 = calculate_conv_output_size(w2, 3, 1)
        
        # Final feature size: 64 channels * h3 * w3
        conv_output_size = 64 * h3 * w3

        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_output_size, HID_SIZE),
            nn.Dropout(self.config.dropout),
        )

        self.actor = nn.Sequential(
            nn.Tanh(),
            nn.Linear(HID_SIZE, 2 * self.config.action_dim) # 2 for sine and cosine
        )

        self.critic = nn.Sequential(
            nn.Tanh(),
            nn.Linear(HID_SIZE, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        """

        # Preprocess images (converts to [batch_size , n_channels , frame_height, frame_width] and normalizes)
        x = self.image_preprocessor(x)
            
        # Feed through network
        x = self.net(x)
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
    
    def evaluate_actions(self, x, actions):
       """
       x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
       actions: tensor of shape [batch_size, action_dim] - these are the squashed actions in [-1, 1]
       """
       logits, value = self.forward(x)
       cos_mean, sin_mean, cos_log_std, sin_log_std = torch.chunk(logits, 4, dim=-1)
       cos_std = torch.exp(cos_log_std.clamp(-5, 2))
       sin_std = torch.exp(sin_log_std.clamp(-5, 2))
       
       # Create distributions for raw (unsquashed) actions
       cos_dist = torch.distributions.Normal(cos_mean, cos_std)
       sin_dist = torch.distributions.Normal(sin_mean, sin_std)
       
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

