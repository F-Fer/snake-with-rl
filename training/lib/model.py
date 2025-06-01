import numpy as np
import torch
import math
import torch.nn as nn
import gymnasium as gym

HID_SIZE = 64


def calculate_conv_output_size(input_size, kernel_size, stride, padding=0):
    """Calculate output size after convolution"""
    return (input_size + 2 * padding - kernel_size) // stride + 1


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
