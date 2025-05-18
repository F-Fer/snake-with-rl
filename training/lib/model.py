import ptan
import numpy as np
import torch
import math
import torch.nn as nn
import gymnasium as gym

HID_SIZE = 64


class ModelActor(nn.Module):
    def __init__(self, obs_shape: tuple, act_size: int):
        super(ModelActor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the output size of the conv layer
        conv_out_size = self._get_conv_out(obs_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, act_size),
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        return self.fc(conv_out)
    
    def _get_conv_out(self, shape: tuple) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


class ModelCritic(nn.Module):
    def __init__(self, obs_shape: tuple):
        super(ModelCritic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = self._get_conv_out(obs_shape)

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def _get_conv_out(self, shape: tuple) -> int:
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        return self.value(conv_out)