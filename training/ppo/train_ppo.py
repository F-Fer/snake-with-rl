import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Set CUDA memory configuration to help with fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Add CUDA debugging
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable device-side assertions
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from tqdm import tqdm
import torch
from torch import nn
import typing as tt
from training.lib.model import ImpalaModel
import multiprocessing
from snake_env.snake_env import SnakeEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import ToTensorImage, TransformedEnv, Resize, FrameSkipTransform, GrayScale, CatFrames
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
import torch.optim as optim
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.envs.transforms import Compose
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs import ParallelEnv
from torchrl.record.loggers.tensorboard import TensorboardLogger
from datetime import datetime
import imageio
import numpy as np


GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 16_384
LEARNING_RATE = 3e-4

PPO_EPS = 0.2
PPO_EPOCHS = 8
PPO_BATCH_SIZE = 256
NUM_ENVS = 16
SEQUENCE_LENGTH = 32  # Length of sequences for LSTM

is_fork = multiprocessing.get_start_method() == "fork"
if torch.cuda.is_available() and not is_fork:
    device = torch.device(0)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class ImpalaWrapper(nn.Module):
    """Wrapper to make IMPALA model compatible with TorchRL's expected interface"""
    def __init__(self, impala_model, num_envs):
        super().__init__()
        self.impala_model = impala_model
        self.num_envs = num_envs
        # Initialize hidden states for all environments
        self.hidden_states = self.impala_model.initial_state(num_envs)
        
    def forward(self, obs):
        # obs shape: (batch_size, channels, height, width)
        # Need to add sequence dimension: (1, batch_size, channels, height, width)
        obs_with_seq = obs.unsqueeze(0)
        
        # Forward pass through IMPALA
        logits, values, self.hidden_states = self.impala_model(obs_with_seq, self.hidden_states)
        
        # Remove sequence dimension: (batch_size, num_actions), (batch_size, 1)
        logits = logits.squeeze(0)
        values = values.squeeze(0)
        
        return logits, values
    
    def reset_states(self, env_indices=None):
        """Reset hidden states for specified environments (or all if None)"""
        if env_indices is None:
            self.hidden_states = self.impala_model.initial_state(self.num_envs)
        else:
            # Reset specific environments
            initial_states = self.impala_model.initial_state(len(env_indices))
            for i, env_idx in enumerate(env_indices):
                self.hidden_states[0][env_idx] = initial_states[0][i]
                self.hidden_states[1][env_idx] = initial_states[1][i]


class PolicyModule(nn.Module):
    """Policy module that extracts action parameters from IMPALA model"""
    def __init__(self, impala_wrapper):
        super().__init__()
        self.impala_wrapper = impala_wrapper
        
    def forward(self, obs):
        logits, _ = self.impala_wrapper(obs)
        # Split logits into loc and scale for continuous actions
        loc, scale = torch.chunk(logits, 2, dim=-1)
        scale = torch.nn.functional.softplus(scale) + 1e-4  # Ensure positive scale
        return {"loc": loc, "scale": scale}


class ValueModule(nn.Module):
    """Value module that extracts value from IMPALA model"""
    def __init__(self, impala_wrapper):
        super().__init__()
        self.impala_wrapper = impala_wrapper
        
    def forward(self, obs):
        _, values = self.impala_wrapper(obs)
        return {"state_value": values.squeeze(-1)}


def main():
    # Start with simpler environment for initial learning
    env = ParallelEnv(NUM_ENVS, lambda: GymWrapper(SnakeEnv(num_bots=20, num_foods=50, zoom_level=1.5)))

    # Define the transforms to apply to the environment
    frameskip_transform = FrameSkipTransform(4)
    resize_transform = Resize(84)
    grayscale_transform = GrayScale()

    env = TransformedEnv(env, Compose(
        frameskip_transform,
        ToTensorImage(in_keys=["pixels"], dtype=torch.uint8, from_int=False),  # Use uint8 to save memory
        # grayscale_transform,
        resize_transform,
        CatFrames(dim=-3, N=2, in_keys=["pixels"])
    ))

    # Test the env and get initial data spec
    initial_data = env.reset()
    # Remove batch dimension to get the actual observation shape
    transformed_obs_shape = initial_data["pixels"].shape[-3:]  # Take last 3 dimensions: (C, H, W)

    # Create the IMPALA model
    impala_model = ImpalaModel(
        in_channels=transformed_obs_shape[0], 
        num_actions=env.action_spec.shape[-1] * 2,  # *2 for loc and scale
        height=transformed_obs_shape[1],
        width=transformed_obs_shape[2],
        device=device
    ).to(device)
    
    # Wrap the IMPALA model
    impala_wrapper = ImpalaWrapper(impala_model, NUM_ENVS)
    
    # Create policy and value modules
    policy_net = PolicyModule(impala_wrapper)
    value_net = ValueModule(impala_wrapper)

    # This module will process "pixels" and output "loc" and "scale"
    policy_params_module = TensorDictModule(
        module=policy_net,
        in_keys=["pixels"], 
        out_keys=["loc", "scale"]
    )

    # ProbabilisticActor samples an action from "loc" and "scale" and outputs "action"
    policy_module = ProbabilisticActor(
        module=policy_params_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
    ).to(device)

    value_module = TensorDictModule(
        value_net,
        in_keys=["pixels"],
        out_keys=["state_value"]
    )

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=TRAJECTORY_SIZE,
        total_frames=1_000_000,
        split_trajs=False,
        device=device
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=TRAJECTORY_SIZE),
        sampler=SamplerWithoutReplacement()
    )

    advantage_module = GAE(
        gamma=GAMMA,
        lmbda=GAE_LAMBDA,
        value_network=value_module,
        average_gae=True,
        device=device,
        differentiable=True
    )

    ppo_loss = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=PPO_EPS,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    # Single optimizer
    optimizer = optim.Adam(impala_model.parameters(), lr=LEARNING_RATE)
    
    # For gradient clipping
    GRAD_CLIP_VALUE = 1.0

    # Initialize TensorBoard logger
    run_name = f"snake_ppo_training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TensorboardLogger(exp_name=run_name, log_dir="logs")
    
    # Create videos directory
    videos_dir = f"videos/{run_name}"
    os.makedirs(videos_dir, exist_ok=True)
    
    pbar = tqdm(total=collector.total_frames) # Use collector's actual total frames

    for i, tensordict_data in enumerate(collector):
        # Set the networks to training mode
        impala_model.train()

        # Print the shape of the data
        print(f"Before squeezing: {tensordict_data.shape}")

        # Reset LSTM states for environments that terminated
        if "done" in tensordict_data.keys():
            done_envs = tensordict_data["done"].any(dim=1)  # Check if any step in trajectory is done
            if done_envs.any():
                done_indices = torch.where(done_envs)[0].cpu().numpy()
                impala_wrapper.reset_states(done_indices)

        # Reshape the data to flatten environment and trajectory dimensions
        # From [num_envs, trajectory_length_per_env, ...] to [total_samples, ...]
        batch_size = tensordict_data.batch_size
        total_samples = batch_size[0] * batch_size[1]  # num_envs * trajectory_length_per_env
        tensordict_data_squeezed = tensordict_data.view(total_samples)
        print(f"After squeezing: {tensordict_data_squeezed.shape}")

        # Calculate advantage once for the entire trajectory
        with torch.no_grad():
            # Process advantage calculation in chunks to avoid memory issues
            chunk_size = 1024
            num_chunks = (total_samples + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_samples)
                
                chunk_data = tensordict_data_squeezed[start_idx:end_idx]
                advantage_module(chunk_data)
                
                # Copy the computed values back to the original tensor
                if chunk_idx == 0:
                    # Initialize the full tensors on first chunk
                    full_advantage = chunk_data["advantage"].clone()
                    full_value_target = chunk_data["value_target"].clone()
                else:
                    # Concatenate subsequent chunks
                    full_advantage = torch.cat([full_advantage, chunk_data["advantage"]], dim=0)
                    full_value_target = torch.cat([full_value_target, chunk_data["value_target"]], dim=0)
                
                # Clear chunk data to free memory
                del chunk_data
            
            # Set the computed values back to the original tensor
            tensordict_data_squeezed["advantage"] = full_advantage
            tensordict_data_squeezed["value_target"] = full_value_target
            # tensordict_data_squeezed now contains "advantage" and "value_target"

        # Normalize advantages
        advantages = tensordict_data_squeezed["advantage"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        tensordict_data_squeezed["advantage"] = advantages
        
        # PPO update loop: multiple epochs over the collected trajectory
        epoch_losses_actor = []
        epoch_losses_critic = []
        epoch_losses_entropy = []
        for _epoch_idx in range(PPO_EPOCHS):
            replay_buffer.empty() # Clear buffer before filling with new trajectory data for this epoch set
            replay_buffer.extend(tensordict_data_squeezed.cpu()) # Add all data from the current trajectory

            batch_losses_actor = []
            batch_losses_critic = []
            batch_losses_entropy = []
            for _batch_idx in range(total_samples // PPO_BATCH_SIZE):
                subdata = replay_buffer.sample(PPO_BATCH_SIZE)
                
                loss_vals = ppo_loss(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                
                batch_losses_actor.append(loss_vals["loss_objective"].cpu().item())
                batch_losses_critic.append(loss_vals["loss_critic"].cpu().item())
                batch_losses_entropy.append(loss_vals["loss_entropy"].cpu().item())

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(impala_model.parameters(), GRAD_CLIP_VALUE)
                optimizer.step()
                optimizer.zero_grad()
                
                # Clean up batch data immediately
                del subdata, loss_vals, loss_value
                
            
            epoch_losses_actor.extend(batch_losses_actor)
            epoch_losses_critic.extend(batch_losses_critic)
            epoch_losses_entropy.extend(batch_losses_entropy)

        # Log metrics to TensorBoard
        logger.log_scalar("train/reward_mean", tensordict_data_squeezed["next", "reward"].mean().item(), step=i)
        logger.log_scalar("train/reward_sum", tensordict_data_squeezed["next", "reward"].sum().item(), step=i)
        
        # Log training loss
        if epoch_losses_actor:
            logger.log_scalar("train/loss_mean_actor", np.mean(epoch_losses_actor), step=i)
            logger.log_scalar("train/loss_std_actor", np.std(epoch_losses_actor), step=i)
        if epoch_losses_critic:
            logger.log_scalar("train/loss_mean_critic", np.mean(epoch_losses_critic), step=i)
            logger.log_scalar("train/loss_std_critic", np.std(epoch_losses_critic), step=i)
        if epoch_losses_entropy:
            logger.log_scalar("train/loss_mean_entropy", np.mean(epoch_losses_entropy), step=i)
            logger.log_scalar("train/loss_std_entropy", np.std(epoch_losses_entropy), step=i)

        # Log learning rate
        logger.log_scalar("train/lr", LEARNING_RATE, step=i)

        pbar.update(total_samples)
        
        if i % 5 == 0:
            
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # Reset LSTM states for evaluation
                impala_wrapper.reset_states()
                
                # execute a rollout with the trained policy
                try:
                    eval_rollout = env.rollout(500, policy_module, break_when_any_done=False)

                    # Move to CPU immediately to avoid CUDA memory issues
                    eval_reward_mean = eval_rollout["next", "reward"].cpu().mean().item()
                    eval_reward_sum = eval_rollout["next", "reward"].cpu().sum().item()
                    
                    logger.log_scalar("eval/reward_mean", eval_reward_mean, step=i)
                    logger.log_scalar("eval/reward_sum", eval_reward_sum, step=i)
                    
                    # Save video every 10 iterations
                    if i % 10 == 0:
                        try:
                            # Extract pixel frames from rollout
                            # Shape should be [NUM_ENVS, sequence_length, channels, height, width]
                            pixel_frames = eval_rollout["pixels"].cpu().numpy()

                            print(pixel_frames.shape)
                            
                            # Take only the first environment's frames: [sequence_length, channels, height, width]
                            if pixel_frames.ndim == 5:
                                pixel_frames = pixel_frames[0]  # Select first environment
                            
                            # Convert from [seq_len, C, H, W] to [seq_len, H, W, C] for video writing
                            pixel_frames = np.transpose(pixel_frames, (0, 2, 3, 1))

                            # Handle frame stacking - take only the most recent frame (last 3 channels)
                            if pixel_frames.shape[-1] > 3:
                                pixel_frames = pixel_frames[:, :, :, -3:]  # Take last 3 channels (most recent frame)
                            
                            # Convert to proper format for video saving
                            if pixel_frames.dtype == np.uint8:
                                print("uint8")
                                # Data is already in 0-255 range as uint8, just ensure it's the right type
                                pixel_frames = pixel_frames.astype(np.uint8)
                            elif pixel_frames.max() <= 1.0:
                                # Data is in 0-1 range, convert to 0-255
                                pixel_frames = (pixel_frames * 255).astype(np.uint8)
                            else:
                                # Data might be in 0-255 range but as float, convert to uint8
                                pixel_frames = pixel_frames.astype(np.uint8)
                            
                            # If grayscale (single channel), convert to RGB
                            if pixel_frames.shape[-1] == 1:
                                pixel_frames = np.repeat(pixel_frames, 3, axis=-1)
                            
                            # Save video
                            video_path = f"{videos_dir}/eval_episode_{i:06d}.mp4"
                            
                            # Use imageio to write video (30 fps)
                            imageio.mimwrite(video_path, pixel_frames, fps=30, quality=8)
                            
                        except Exception as video_error:
                            print(f"Failed to save video: {video_error}")
                    
                except Exception as e:
                    print(f"Evaluation failed: {e}")

    collector.shutdown() # Ensure collector resources are released
    pbar.close()

if __name__ == "__main__":
    main()