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
from training.lib.model import ModelActor, ModelCritic, ModelResidual
import multiprocessing
from snake_env.envs.snake_env import SnakeEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import ToTensorImage, TransformedEnv, Resize, FrameSkipTransform, GrayScale
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

TRAJECTORY_SIZE = 8_192
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3

PPO_EPS = 0.2
PPO_EPOCHS = 10
PPO_BATCH_SIZE = 32
NUM_ENVS = 8

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

print(f"Using device: {device}")


def main():
    # Start with simpler environment for initial learning
    env = ParallelEnv(NUM_ENVS, lambda: GymWrapper(SnakeEnv(num_bots=5, num_foods=20)))  # Reduced from 20 bots, 50 foods

    # Define the transforms to apply to the environment
    frameskip_transform = FrameSkipTransform(4)
    resize_transform = Resize(84)
    grayscale_transform = GrayScale()
    
    env = TransformedEnv(env, Compose(
        frameskip_transform,
        ToTensorImage(in_keys=["pixels"], dtype=torch.float32),
        grayscale_transform,
        resize_transform
    ))

    # Test the env and get initial data spec
    initial_data = env.reset()
    # Remove batch dimension to get the actual observation shape
    transformed_obs_shape = initial_data["pixels"].shape[-3:]  # Take last 3 dimensions: (C, H, W)

    # actor_net_core = ModelActor(transformed_obs_shape, env.action_spec.shape[-1]).to(device)
    # critic_net = ModelCritic(transformed_obs_shape).to(device)
    actor_net_core = ModelResidual(channels_in=transformed_obs_shape[0], num_outputs=env.action_spec.shape[-1] * 2).to(device)
    critic_net = ModelResidual(channels_in=transformed_obs_shape[0], num_outputs=1).to(device)


    # Initialize the actor and critic networks
    with torch.no_grad():
        # Use the full tensor with batch dimension for proper initialization
        test_input = initial_data["pixels"]  # Keep the batch dimension [1, 3, 84, 84]
        # Initialize the lazy layers properly
        actor_output = actor_net_core(test_input.to(device))
        critic_output = critic_net(test_input.to(device))

    num_params = sum(p.numel() for p in actor_net_core.parameters())
    bytes_per_param = 4  # for float32, use 2 for float16, 1 for int8, etc.
    total_bytes = num_params * bytes_per_param
    total_mb = total_bytes / (1024 ** 2)
    print(f"Model size: {total_mb:.2f} MB")

    # Actor network produces raw parameters for the distribution
    actor_network_with_extractor = nn.Sequential(
        actor_net_core,
        NormalParamExtractor() # Splits the output of actor_net_core into loc and scale
    ).to(device)

    # This module will process "pixels" and output "loc" and "scale"
    policy_params_module = TensorDictModule(
        module=actor_network_with_extractor,
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
        critic_net,
        in_keys=["pixels"],
        out_keys=["state_value"],
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
        storage=LazyTensorStorage(max_size=PPO_BATCH_SIZE),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=GAMMA,
        lmbda=GAE_LAMBDA,
        value_network=value_module,
        average_gae=True,
        device=device
    )

    ppo_loss = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=PPO_EPS,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    # Use separate optimizers for better control
    optimizer_actor = optim.Adam(policy_module.parameters(), lr=LEARNING_RATE_ACTOR)
    optimizer_critic = optim.Adam(value_module.parameters(), lr=LEARNING_RATE_CRITIC)
    
    # For gradient clipping
    GRAD_CLIP_VALUE = 1.0

    # Initialize TensorBoard logger
    run_name = f"snake_ppo_training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TensorboardLogger(exp_name=run_name, log_dir="logs")
    
    # Create videos directory
    videos_dir = f"videos/{run_name}"
    os.makedirs(videos_dir, exist_ok=True)
    
    eval_str = ""
    pbar = tqdm(total=collector.total_frames) # Use collector's actual total frames

    for i, tensordict_data in enumerate(collector):
        # Reshape the data to flatten environment and trajectory dimensions
        # From [num_envs, trajectory_length_per_env, ...] to [total_samples, ...]
        batch_size = tensordict_data.batch_size
        total_samples = batch_size[0] * batch_size[1]  # num_envs * trajectory_length_per_env
        tensordict_data_squeezed = tensordict_data.view(total_samples)

        # Calculate advantage once for the entire trajectory
        with torch.no_grad():
            advantage_module(tensordict_data_squeezed)
            # tensordict_data_squeezed now contains "advantage" and "value_target"

        
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
                torch.nn.utils.clip_grad_norm_(ppo_loss.parameters(), GRAD_CLIP_VALUE)
                optimizer_actor.step()
                optimizer_critic.step()
                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()
                
                # Clean up batch data immediately
                del subdata, loss_vals, loss_value
                
                # Clear GPU cache every few batches
                if _batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
            
            epoch_losses_actor.extend(batch_losses_actor)
            epoch_losses_critic.extend(batch_losses_critic)
            epoch_losses_entropy.extend(batch_losses_entropy)
            # Clear GPU cache after each epoch
            torch.cuda.empty_cache()

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

        # Clean up trajectory data after all epochs
        del tensordict_data_squeezed, tensordict_data
        torch.cuda.empty_cache()

        pbar.update(total_samples)
        
        if i % 10 == 0:
            # Clear CUDA cache before evaluation
            torch.cuda.empty_cache()
            
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                try:
                    eval_rollout = env.rollout(1000, policy_module, break_when_any_done=False)

                    # Move to CPU immediately to avoid CUDA memory issues
                    eval_reward_mean = eval_rollout["next", "reward"].cpu().mean().item()
                    eval_reward_sum = eval_rollout["next", "reward"].cpu().sum().item()
                    
                    logger.log_scalar("eval/reward_mean", eval_reward_mean, step=i)
                    logger.log_scalar("eval/reward_sum", eval_reward_sum, step=i)
                    
                    # Save video every 20 iterations
                    if i % 20 == 0:
                        try:
                            # Extract pixel frames from rollout
                            # Shape should be [NUM_ENVS, sequence_length, channels, height, width]
                            pixel_frames = eval_rollout["pixels"].cpu().numpy()
                            
                            # Take only the first environment's frames: [sequence_length, channels, height, width]
                            if pixel_frames.ndim == 5:
                                pixel_frames = pixel_frames[0]  # Select first environment
                            
                            # Convert from [seq_len, C, H, W] to [seq_len, H, W, C] for video writing
                            pixel_frames = np.transpose(pixel_frames, (0, 2, 3, 1))
                            
                            # Ensure values are in 0-255 range
                            if pixel_frames.max() <= 1.0:
                                pixel_frames = (pixel_frames * 255).astype(np.uint8)
                            else:
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
                    
                    # Clean up evaluation rollout immediately
                    del eval_rollout
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Evaluation failed: {e}")
                    torch.cuda.empty_cache()

    collector.shutdown() # Ensure collector resources are released
    pbar.close()

if __name__ == "__main__":
    main()