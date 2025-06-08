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
import torch.nn.functional as F
import typing as tt
from training.lib.model import ImpalaCNN, PolicyValueHead
import multiprocessing
from snake_env.envs.snake_env import SnakeEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import (
    ToTensorImage, TransformedEnv, Resize, FrameSkipTransform, 
    GrayScale, CatFrames, InitTracker
)
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
import torch.optim as optim
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.envs.transforms import Compose
from torchrl.modules import ProbabilisticActor, TanhNormal, LSTMModule
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs import ParallelEnv
from torchrl.record.loggers.tensorboard import TensorboardLogger
from torchrl.modules.utils import get_primers_from_module
from datetime import datetime
import imageio
import numpy as np

GAMMA = 0.99
GAE_LAMBDA = 0.95

LEARNING_RATE = 3e-5
PPO_EPS = 0.2
PPO_EPOCHS = 4  # Reduced for quicker iteration
PPO_BATCH_SIZE = 128  # Reduced for memory
NUM_ENVS = 8  # Reduced for debugging
SEQUENCE_LENGTH = 64  # Reduced for better memory management
TOTAL_FRAMES = 1_000_000
GRAD_CLIP_VALUE = 1.0
LSTM_HIDDEN_SIZE = 256
CNN_FEATURE_DIM = 256

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

print(f"Using device: {device}")


def create_policy_network(obs_shape, action_spec, device):
    """Create the policy network using TorchRL's LSTMModule"""
    
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
    
    # LSTM module - using TorchRL's LSTMModule
    lstm_module = LSTMModule(
        input_size=CNN_FEATURE_DIM,
        hidden_size=LSTM_HIDDEN_SIZE,
        in_keys=["cnn_features", "recurrent_state_h", "recurrent_state_c"],
        out_keys=["lstm_features", ("next", "recurrent_state_h"), ("next", "recurrent_state_c")]
    ).to(device)
    
    # Policy and Value heads
    policy_value_head = PolicyValueHead(
        input_size=LSTM_HIDDEN_SIZE,
        num_actions=action_spec.shape[-1] * 2  # *2 for loc and scale
    ).to(device)
    
    policy_value_module = TensorDictModule(
        policy_value_head,
        in_keys=["lstm_features"],
        out_keys=["loc", "scale", "state_value"]
    )
    
    # Combine all modules for actor
    actor_module = TensorDictSequential(
        cnn_module,
        lstm_module,
        policy_value_module
    ).to(device)
    
    # Create separate value network (shares weights with actor)
    value_network = TensorDictSequential(
        cnn_module,
        lstm_module,
        TensorDictModule(
            lambda td: {"state_value": policy_value_head(td["lstm_features"])["state_value"]},
            in_keys=["lstm_features"],
            out_keys=["state_value"]
        )
    ).to(device)
    
    # Wrap with ProbabilisticActor
    policy_module = ProbabilisticActor(
        module=actor_module,
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
    
    return policy_module, value_network, lstm_module


def main():
    # Create environment
    env = ParallelEnv(NUM_ENVS, lambda: GymWrapper(SnakeEnv(num_bots=20, num_foods=50, zoom_level=1.5)))

    # Define transforms
    frameskip_transform = FrameSkipTransform(4)
    resize_transform = Resize(84)
    init_tracker = InitTracker()  # Add InitTracker for is_init key
    
    env = TransformedEnv(env, Compose(
        frameskip_transform,
        ToTensorImage(in_keys=["pixels"], dtype=torch.uint8, from_int=False),
        resize_transform,
        CatFrames(dim=-3, N=2, in_keys=["pixels"]),  # Stack 2 frames
        init_tracker  # Must be after frame stacking
    ))

    # Get observation shape
    initial_data = env.reset()
    obs_shape = initial_data["pixels"].shape[-3:]  # (C, H, W)
    print(f"Observation shape: {obs_shape}")

    # Create policy and value networks
    policy_module, value_module, lstm_module = create_policy_network(
        obs_shape, env.action_spec, device
    )
    
    # Add TensorDict primer for LSTM states
    env = env.append_transform(lstm_module.make_tensordict_primer())
    
    # Create data collector
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=NUM_ENVS * SEQUENCE_LENGTH,
        total_frames=TOTAL_FRAMES,
        device=device,
        compile_policy=False,  # Disable compilation for debugging
        reset_at_each_iter=False,
    )

    # Advantage calculation
    advantage_module = GAE(
        gamma=GAMMA,
        lmbda=GAE_LAMBDA,
        value_network=value_module,
        average_gae=True,
        device=device,
        differentiable=True
    )

    # PPO Loss
    ppo_loss = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=PPO_EPS,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    # Optimizer - optimize all parameters in the policy module
    optimizer = optim.Adam(policy_module.parameters(), lr=LEARNING_RATE)

    # Initialize TensorBoard logger
    run_name = f"snake_ppo_torchrl_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TensorboardLogger(exp_name=run_name, log_dir="logs")
    
    # Create videos directory
    videos_dir = f"videos/{run_name}"
    os.makedirs(videos_dir, exist_ok=True)

    frames_done = 0
    pbar = tqdm(total=TOTAL_FRAMES)
    iteration = 0

    for rollout in collector:
        frames_done += rollout.batch_size[0]
        
        # Calculate GAE advantage
        with torch.no_grad():
            advantage_module(rollout)
        
        # Normalize advantages
        rollout["advantage"] = (
            (rollout["advantage"] - rollout["advantage"].mean())
            / (rollout["advantage"].std() + 1e-8)
        )

        # Log rollout statistics
        logger.log_scalar("train/advantage_mean", rollout["advantage"].mean(), step=frames_done)
        logger.log_scalar("train/value_mean", rollout["state_value"].mean(), step=frames_done)
        logger.log_scalar("train/reward_mean", rollout["next", "reward"].mean(), step=frames_done)
        logger.log_scalar("train/reward_sum", rollout["next", "reward"].sum(), step=frames_done)

        # PPO Training loop - simple approach using entire rollout
        loss_values = {"loss_total": [], "loss_policy": [], "loss_value": [], "loss_entropy": []}
        
        # Convert rollout to proper shape for training
        # Flatten batch and time dimensions for training
        B, T = rollout.batch_size
        rollout_flat = rollout.view(-1)  # Flatten to [B*T]
        
        # Create mini-batches from flattened rollout
        indices = torch.randperm(B * T, device=device)
        
        for epoch in range(PPO_EPOCHS):
            for i in range(0, B * T, PPO_BATCH_SIZE):
                batch_indices = indices[i:i + PPO_BATCH_SIZE]
                batch = rollout_flat[batch_indices]
                
                # Compute PPO loss
                loss_dict = ppo_loss(batch)
                total_loss = loss_dict["loss_objective"] + loss_dict["loss_critic"] + loss_dict["loss_entropy"]
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_module.parameters(), GRAD_CLIP_VALUE)
                optimizer.step()
                
                # Store loss values
                loss_values["loss_total"].append(total_loss.item())
                loss_values["loss_policy"].append(loss_dict["loss_objective"].item())
                loss_values["loss_value"].append(loss_dict["loss_critic"].item())
                loss_values["loss_entropy"].append(loss_dict["loss_entropy"].item())

        # Log training metrics
        for key, values in loss_values.items():
            if values:
                logger.log_scalar(f"train/{key}", np.mean(values), step=frames_done)

        # Update progress bar
        pbar.update(rollout.batch_size[0])

        # Evaluation and video saving
        if iteration % 10 == 0:  # Evaluate less frequently
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                try:
                    # Create evaluation collector
                    eval_env = ParallelEnv(1, lambda: GymWrapper(SnakeEnv(num_bots=20, num_foods=50, zoom_level=1.5)))
                    eval_env = TransformedEnv(eval_env, Compose(
                        frameskip_transform,
                        ToTensorImage(in_keys=["pixels"], dtype=torch.uint8, from_int=False),
                        resize_transform,
                        CatFrames(dim=-3, N=2, in_keys=["pixels"]),
                        init_tracker
                    ))
                    eval_env = eval_env.append_transform(lstm_module.make_tensordict_primer())
                    
                    eval_collector = SyncDataCollector(
                        eval_env,
                        policy_module,
                        frames_per_batch=500,  # Shorter evaluation
                        total_frames=500,
                        device=device,
                        compile_policy=False,
                    )
                    
                    eval_rollout = next(iter(eval_collector))
                    
                    eval_reward_mean = eval_rollout["next", "reward"].cpu().mean().item()
                    eval_reward_sum = eval_rollout["next", "reward"].cpu().sum().item()
                    
                    logger.log_scalar("eval/reward_mean", eval_reward_mean, step=iteration)
                    logger.log_scalar("eval/reward_sum", eval_reward_sum, step=iteration)
                    
                    # Save video every 20 iterations
                    if iteration % 20 == 0:
                        try:
                            pixel_frames = eval_rollout["pixels"].cpu().numpy()
                            
                            # Handle dimensions: [batch, time, channels, height, width]
                            if pixel_frames.ndim == 5:
                                pixel_frames = pixel_frames[0]  # Take first environment
                            
                            # Convert to video format: [time, height, width, channels]
                            pixel_frames = np.transpose(pixel_frames, (0, 2, 3, 1))
                            
                            # Handle frame stacking - take last 3 channels
                            if pixel_frames.shape[-1] > 3:
                                pixel_frames = pixel_frames[:, :, :, -3:]
                            
                            # Convert to uint8 format
                            if pixel_frames.dtype != np.uint8:
                                if pixel_frames.max() <= 1.0:
                                    pixel_frames = (pixel_frames * 255).astype(np.uint8)
                                else:
                                    pixel_frames = pixel_frames.astype(np.uint8)
                            
                            # Ensure RGB format
                            if pixel_frames.shape[-1] == 1:
                                pixel_frames = np.repeat(pixel_frames, 3, axis=-1)
                            
                            # Save video
                            video_path = f"{videos_dir}/eval_episode_{iteration:06d}.mp4"
                            imageio.mimwrite(video_path, pixel_frames, fps=30, quality=8)
                            
                        except Exception as video_error:
                            print(f"Failed to save video: {video_error}")
                    
                    eval_collector.shutdown()
                    eval_env.close()
                    
                except Exception as e:
                    print(f"Evaluation failed: {e}")

        iteration += 1

    pbar.close()
    collector.shutdown()
    env.close()


if __name__ == "__main__":
    main()