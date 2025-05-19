import torch
import typing as tt
from training.lib.model import ModelActor, ModelCritic

GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4

PPO_EPS = 0.2
PPO_EPOCHS = 10
PPO_BATCH_SIZE = 64

device 