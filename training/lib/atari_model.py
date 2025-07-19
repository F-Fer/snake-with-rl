import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from training.lib.atari_config import Config
from torch.distributions import TransformedDistribution, TanhTransform, Independent, Normal

HID_SIZE = 512


def calculate_conv_output_size(input_size, kernel_size, stride, padding=0):
    """Calculate output size after convolution"""
    return (input_size + 2 * padding - kernel_size) // stride + 1


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    

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
            layer_init(nn.Linear(conv_output_size, HID_SIZE)),
            nn.ReLU(),
        )

        self.actor_mean = layer_init(nn.Linear(HID_SIZE, self.config.action_dim)) # Outputs for means
        self.actor_logstd = nn.Parameter(torch.ones(1, self.config.action_dim) * config.start_log_std)
        self.critic = layer_init(nn.Linear(HID_SIZE, 1))  # Unbounded value output
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        """
        x = self.image_preprocessor(x)
        x = self.net(x)
        return self.critic(x)
    
    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        """

        # Preprocess images (converts to [batch_size , n_channels , frame_height, frame_width] and normalizes)
        x = self.image_preprocessor(x)
            
        # Feed through network
        x = self.net(x)

        # Get action mean
        action_mean = self.actor_mean(x)

        # Get action logstd (state independent)
        # Clamp log std to a reasonable range to avoid numerical issues
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # Create normal distribution
        base_dist = Independent(Normal(action_mean, action_std), 1)
        probs = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])

        # Sample action if not provided, otherwise ensure provided action is within valid bounds
        if action is None:
            action = probs.sample()
        else:
            eps = 1e-6
            action = action.clamp(-1 + eps, 1 - eps)

        return action, probs.log_prob(action), base_dist.entropy(), self.critic(x)

