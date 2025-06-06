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
from training.lib.model import ImpalaModel
import multiprocessing
from snake_env.envs.snake_env import SnakeEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.transforms import ToTensorImage, TransformedEnv, Resize, FrameSkipTransform, GrayScale, CatFrames
from tensordict import TensorDict
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
TOTAL_FRAMES = 1_000_000
NUM_MINIBATCHES = 4
MINIBATCH_SIZE = NUM_ENVS // NUM_MINIBATCHES

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

print(f"Using device: {device}")


class ImpalaBackbone(nn.Module):
    """
    Stateful wrapper that manages LSTM state internally
    """
    def __init__(self, impala_model, num_envs):
        super().__init__()
        self.impala = impala_model
        self.num_envs = num_envs
        self.reset_states()

    def reset_states(self):
        """Reset LSTM states for all environments"""
        self.lstm_state = self.impala.initial_state(self.num_envs)

    def forward(self, pixels):
        single_obs = len(pixels.shape) == 4
        
        if single_obs:
            # Rollout phase (stateful):
            # Use the managed LSTM state and add a time dimension.
            pixels_seq = pixels.unsqueeze(1)
            lstm_state = self.lstm_state
        else:
            # Training phase (stateless for each batch):
            # Create a new initial state for this batch.
            pixels_seq = pixels
            batch_size = pixels_seq.shape[0]
            lstm_state = self.impala.initial_state(batch_size)
        
        print(f"pixels_seq.shape: {pixels_seq.shape}")
        print(f"lstm_state: {lstm_state[0].shape}")
        logits, val, next_state = self.impala(pixels_seq, lstm_state)

        # Update internal state only during stateful rollout
        if single_obs:
            self.lstm_state = next_state
            # Remove the time dimension for single observations
            logits = logits.squeeze(1)
            val = val.squeeze(1)

        loc, scale = torch.chunk(logits, 2, dim=-1)
        
        return {
            "loc":  loc,
            "scale": F.softplus(scale) + 1e-4,          # keep >0
            "state_value": val,
        }
        
    def reset_lstm_for_done_envs(self, done_mask):
        """Reset LSTM state for environments that are done"""
        if isinstance(self.lstm_state, tuple):
            hx, cx = self.lstm_state
            
            done_mask = done_mask.to(hx.device).float().view(1, -1, 1)
                    
            hx = hx * (1 - done_mask)
            cx = cx * (1 - done_mask)
            self.lstm_state = (hx, cx)
    

def collect_rollout(env, policy, traj_len_per_env, backbone):
    """
    Returns:
        rollout     - TensorDict  [num_envs, traj_len, ...]
    """
    steps = []
    
    # Get initial observation
    obs = env.reset()
    
    for step_idx in range(traj_len_per_env):
        # Create input tensordict with current observation
        current_pixels = obs["pixels"]
        # Ensure pixels are on CPU before TensorDict conversion,
        # letting TensorDict handle the move to the target device.
        pixels_on_cpu = current_pixels.cpu()

        td_in = TensorDict(
            {"pixels": pixels_on_cpu},
            batch_size=[NUM_ENVS],
            device=device,
        )
        
        with torch.no_grad():
            td_out = policy(td_in)

        action = td_out["action"]

        # Step the environment
        next_obs = env.step(td_out)
        
        # Extract rewards and dones from the stepped environment
        reward = next_obs["next", "reward"]
        done = next_obs["next", "done"]
        
        # Reset LSTM state for done episodes
        backbone.reset_lstm_for_done_envs(done)

        # Get next state value for GAE calculation
        next_pixels_cpu = next_obs["next", "pixels"].cpu()
        next_td_in = TensorDict(
            {"pixels": next_pixels_cpu},
            batch_size=[NUM_ENVS],
            device=device,
        )
        
        with torch.no_grad():
            next_td_out = policy(next_td_in)
        
        # Store the step data
        step_data = TensorDict({
            "pixels": obs["pixels"].to(device),
            "action": action,
            "reward": reward.squeeze(-1).to(device),  # Remove extra dimension
            "done": done.squeeze(-1).to(device),      # Remove extra dimension
            "next": {
                "pixels": next_obs["next", "pixels"].to(device),
                "reward": reward.squeeze(-1).to(device),  # Remove extra dimension
                "done": done.squeeze(-1).to(device),      # Remove extra dimension
                "state_value": next_td_out["state_value"],
            },
            "loc": td_out["loc"],
            "scale": td_out["scale"],
            "sample_log_prob": td_out["sample_log_prob"],
            "state_value": td_out["state_value"],
        }, batch_size=[NUM_ENVS])
        
        steps.append(step_data)
        
        # Update current observation for next step
        obs = next_obs["next"]

    rollout = torch.stack(steps, dim=1)  # [NUM_ENVS, traj_len_per_env, ...]
    return rollout


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
    
    # Stateful backbone for data collection
    backbone = ImpalaBackbone(impala_model, NUM_ENVS).to(device)
    
    # Collection policy (stateful)
    actor_core = TensorDictModule(
        backbone,
        in_keys=["pixels"],
        out_keys=["loc", "scale", "state_value"],
    )

    policy_module = ProbabilisticActor(
        module=actor_core,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low":  env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
    ).to(device)

    # critic just re-uses the value inserted by backbone
    value_module = TensorDictModule(
        lambda td: td,              # identity – value already present
        in_keys=["state_value"],
        out_keys=["state_value"],
    ).to(device)

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
    
    pbar = tqdm(total=TOTAL_FRAMES) # Use collector's actual total frames

    frames_done = 0
    i = 0
    while frames_done < TOTAL_FRAMES:
        # Collect rollout
        rollout = collect_rollout(
            env, policy_module, traj_len_per_env=SEQUENCE_LENGTH, backbone=backbone)
        
        frames_done += NUM_ENVS * SEQUENCE_LENGTH

        # Calculate advantage on the un-flattened rollout
        print("Calculating advantage")
        advantage_module(rollout)

        # Normalize advantage
        rollout["advantage"] = (
            (rollout["advantage"] - rollout["advantage"].mean())
            / (rollout["advantage"].std() + 1e-8)
        )

        # Add a trailing unit dimension to advantage so PPO treats env and time dims correctly
        rollout["advantage"] = rollout["advantage"].unsqueeze(-1)

        print(f"rollout shape: {rollout.shape}")
        # PPO update loop: multiple epochs over the collected trajectory
        impala_model.train()
        
        loss_total = []
        loss_policy = []
        loss_value = []
        loss_entropy = []
        for _ in range(PPO_EPOCHS):
            # Randomly permute env dimension and iterate minibatches
            perm = torch.randperm(NUM_ENVS)
            for start in range(0, NUM_ENVS, MINIBATCH_SIZE):
                idx = perm[start:start+MINIBATCH_SIZE]
                sub_batch = rollout[idx].to(device)
                loss_dict = ppo_loss(sub_batch)
                loss = (
                    loss_dict["loss_objective"]
                    + loss_dict["loss_critic"]
                    + loss_dict["loss_entropy"]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(impala_model.parameters(), GRAD_CLIP_VALUE)
                optimizer.step()
                optimizer.zero_grad()

                loss_total.append(loss.item())
                loss_policy.append(loss_dict["loss_objective"].item())
                loss_value.append(loss_dict["loss_critic"].item())
                loss_entropy.append(loss_dict["loss_entropy"].item())

        # Log metrics to TensorBoard
        logger.log_scalar("train/loss_total", np.mean(loss_total), step=frames_done)
        logger.log_scalar("train/loss_policy", np.mean(loss_policy), step=frames_done)
        logger.log_scalar("train/loss_value", np.mean(loss_value), step=frames_done)
        logger.log_scalar("train/loss_entropy", np.mean(loss_entropy), step=frames_done)
            
        pbar.update(frames_done)

        if i % 5 == 0:
            
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # Reset LSTM states for evaluation
                backbone.reset_states()
                
                # execute a rollout with the trained policy
                try:
                    # Simple approach: just use our manual rollout collection for evaluation
                    # Reset states and collect a shorter evaluation rollout
                    eval_length = min(500, SEQUENCE_LENGTH * 5)  # Shorter evaluation
                    eval_rollout_dict = collect_rollout(env, policy_module, eval_length, backbone)
                    
                    # Convert to format expected by rest of code
                    eval_rollout = eval_rollout_dict

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
                    # Skip evaluation logging for this iteration
        i += 1
    pbar.close()

if __name__ == "__main__":
    main()