import numpy as np
import torch
import math
import torch.nn as nn
import gymnasium as gym
from training.ppo_transformer.lib.config import Config
from torch.distributions import TransformedDistribution, TanhTransform, Independent, Normal

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

        self.actor = nn.Linear(HID_SIZE, 2 * self.config.action_dim)  # Outputs for means and log_stds (unbounded)

        self.critic = nn.Linear(HID_SIZE, 1)  # Unbounded value output

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
        action_logits, value = self.forward(x)
        
        means = action_logits[:, :self.config.action_dim]
        log_stds = action_logits[:, self.config.action_dim:]
        stds = torch.exp(log_stds.clamp(-5, 2))
        
        base_dist = Independent(Normal(means, stds), 1)
        squashed_dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])
        
        action = squashed_dist.sample()
        log_prob = squashed_dist.log_prob(action)
        entropy = base_dist.entropy()
        
        value = value.squeeze()
        
        return action, log_prob, entropy, value
    
    def evaluate_actions(self, x, actions):
        """
        x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        actions: tensor of shape [batch_size, action_dim] - squashed actions in [-1, 1]
        """
        logits, value = self.forward(x)

        means = logits[:, :self.config.action_dim]
        log_stds = logits[:, self.config.action_dim:]
        stds = torch.exp(log_stds.clamp(-5, 2))

        base_dist = Independent(Normal(means, stds), 1)
        squashed_dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])

        # Clamp actions slightly inside the open interval (-1, 1) to avoid numerical
        # instabilities when computing the inverse tanh in log_prob. Actions that hit
        # exactly the boundaries yield +/-inf after atanh which in turn produces NaNs
        # in the subsequent computations and can corrupt the network weights.
        eps = 1e-6
        actions_clamped = actions.clamp(-1 + eps, 1 - eps)

        log_prob = squashed_dist.log_prob(actions_clamped)
        entropy = base_dist.entropy()

        return log_prob, entropy, value.squeeze(-1)

