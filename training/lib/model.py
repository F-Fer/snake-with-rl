import numpy as np
import torch
import math
import torch.nn as nn
import gymnasium as gym

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
        x = self.relu(self.conv_net(x))
        x = self.relu(self.linear(x))
        x = self.linear2(x)
        return x
        

class ModelActor(nn.Module):
    def __init__(self, obs_shape: tuple, act_size: int):
        super(ModelActor, self).__init__()
        
        # Calculate the size after convolutions
        # Input: (C, H, W) = (1, 84, 84) for grayscale
        c, h, w = obs_shape
        
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
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_output_size, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, 2 * act_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelCritic(nn.Module):
    def __init__(self, obs_shape: tuple):
        super(ModelCritic, self).__init__()
        
        # Calculate the size after convolutions
        # Input: (C, H, W) = (1, 84, 84) for grayscale
        c, h, w = obs_shape
        
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
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_output_size, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
