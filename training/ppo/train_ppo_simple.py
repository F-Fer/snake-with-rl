import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from training.lib.model import ImpalaCNN
import multiprocessing
from snake_env.envs.snake_env import SnakeEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import (
    ToTensorImage, TransformedEnv, Resize, FrameSkipTransform, 
    CatFrames, InitTracker
)
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
import torch.optim as optim
from torchrl.collectors import SyncDataCollector
from torchrl.envs.transforms import Compose
from torchrl.modules import ProbabilisticActor, TanhNormal, LSTMModule
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs import ParallelEnv
from datetime import datetime
import numpy as np

# Simplified hyperparameters
GAMMA = 0.99
LEARNING_RATE = 3e-4
NUM_ENVS = 4
SEQUENCE_LENGTH = 32
TOTAL_FRAMES = 10_000  # Very reduced for testing
LSTM_HIDDEN_SIZE = 128
CNN_FEATURE_DIM = 128

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

print(f"Using device: {device}")


class PolicyValueHead(nn.Module):
    """Combined policy and value head"""
    def __init__(self, input_size, action_size):
        super().__init__()
        self.policy_head = nn.Linear(input_size, action_size * 2)
        self.value_head = nn.Linear(input_size, 1)
        self.action_size = action_size

    def forward(self, x):
        policy_out = self.policy_head(x)
        value_out = self.value_head(x).squeeze(-1)
        
        loc = policy_out[..., :self.action_size]
        scale = F.softplus(policy_out[..., self.action_size:]) + 1e-4
        
        return {"loc": loc, "scale": scale, "state_value": value_out}


def create_networks(obs_shape, action_spec, device):
    """Create simplified networks"""
    
    # CNN backbone
    cnn_backbone = ImpalaCNN(
        in_channels=obs_shape[0], 
        feature_dim=CNN_FEATURE_DIM, 
        height=obs_shape[1], 
        width=obs_shape[2]
    ).to(device)
    
    # CNN module
    cnn_module = TensorDictModule(
        cnn_backbone,
        in_keys=["pixels"],
        out_keys=["cnn_features"]
    )
    
    # LSTM module
    lstm_module = LSTMModule(
        input_size=CNN_FEATURE_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
        in_keys=["cnn_features", "recurrent_state_h", "recurrent_state_c"],
        out_keys=["lstm_features", ("next", "recurrent_state_h"), ("next", "recurrent_state_c")]
    ).to(device)
    
    # Policy and value head
    policy_value_head = PolicyValueHead(LSTM_HIDDEN_SIZE, action_spec.shape[-1]).to(device)
    pv_module = TensorDictModule(
        policy_value_head,
        in_keys=["lstm_features"],
        out_keys=["loc", "scale", "state_value"]
    )
    
    # Combined network
    actor_critic = TensorDictSequential(
        cnn_module,
        lstm_module,
        pv_module
    ).to(device)
    
    # Wrap with ProbabilisticActor
    policy = ProbabilisticActor(
        module=actor_critic,
        spec=action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        return_log_prob=True,
    ).to(device)
    
    return policy, lstm_module


def compute_returns_and_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    """Simple GAE computation"""
    # Squeeze reward dimension to match values
    if rewards.dim() == 3:
        rewards = rewards.squeeze(-1)
    if dones.dim() == 3:
        dones = dones.squeeze(-1)
    
    T = rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    gae = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0  # Assume terminal
        else:
            next_value = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t].float()) - values[:, t]
        gae = delta + gamma * lam * (1 - dones[:, t].float()) * gae
        advantages[:, t] = gae
        returns[:, t] = advantages[:, t] + values[:, t]
    
    return returns, advantages


def main():
    # Create environment
    env = ParallelEnv(NUM_ENVS, lambda: GymWrapper(SnakeEnv(num_bots=5, num_foods=20, zoom_level=1.5)))

    # Add transforms
    env = TransformedEnv(env, Compose(
        FrameSkipTransform(4),
        ToTensorImage(in_keys=["pixels"], dtype=torch.uint8, from_int=False),
        Resize(84),
        CatFrames(dim=-3, N=2, in_keys=["pixels"]),
        InitTracker()
    ))

    # Get observation shape
    initial_data = env.reset()
    obs_shape = initial_data["pixels"].shape[-3:]
    print(f"Observation shape: {obs_shape}")

    # Create networks
    policy, lstm_module = create_networks(obs_shape, env.action_spec, device)
    
    # Add LSTM primer
    env = env.append_transform(lstm_module.make_tensordict_primer())
    
    # Create collector
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=NUM_ENVS * SEQUENCE_LENGTH,
        total_frames=TOTAL_FRAMES,
        device=device,
        compile_policy=False,
    )

    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    frames_done = 0
    pbar = tqdm(total=TOTAL_FRAMES)

    for iteration, rollout in enumerate(collector):
        frames_done += rollout.batch_size[0]
        
        # Extract data
        pixels = rollout["pixels"]
        rewards = rollout["next", "reward"]
        dones = rollout["next", "done"]
        actions = rollout["action"]
        
        print(f"\nIteration {iteration}")
        print(f"Rollout shapes - Rewards: {rewards.shape}, Actions: {actions.shape}")
        print(f"Reward mean: {rewards.mean().item():.3f}, Reward sum: {rewards.sum().item():.3f}")
        
        # Re-run the policy to get current log probs and values
        with torch.no_grad():
            # Reshape for recurrent mode
            B, T = rollout.batch_size
            rollout_for_eval = rollout.view(B * T)
            
        # Set recurrent mode and recompute values
        with lstm_module.set_recurrent_mode(True):
            policy_output = policy(rollout)
            current_values = policy_output["state_value"]
            current_log_probs = policy_output["sample_log_prob"]
        
        # Compute returns and advantages (use detached values)
        returns, advantages = compute_returns_and_advantages(
            rewards.cpu(), current_values.detach().cpu(), dones.cpu(), GAMMA
        )
        
        # Move back to device
        returns = returns.to(device)
        advantages = advantages.to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Flatten all tensors for loss computation
        log_probs_flat = current_log_probs.view(-1)
        advantages_flat = advantages.view(-1)
        values_flat = current_values.view(-1)
        returns_flat = returns.view(-1)
        
        print(f"Tensor shapes - log_probs: {log_probs_flat.shape}, advantages: {advantages_flat.shape}")
        print(f"               values: {values_flat.shape}, returns: {returns_flat.shape}")
        
        # Simple policy loss
        policy_loss = -(log_probs_flat * advantages_flat).mean()
        value_loss = F.mse_loss(values_flat, returns_flat)
        
        total_loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        # Log
        print(f"  Policy loss: {policy_loss.item():.3f}")
        print(f"  Value loss: {value_loss.item():.3f}")
        print(f"  Total loss: {total_loss.item():.3f}")
        
        pbar.update(rollout.batch_size[0])

    pbar.close()
    collector.shutdown()
    env.close()
    print("Training completed!")


if __name__ == "__main__":
    main() 