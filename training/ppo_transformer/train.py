import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import argparse
from typing import Tuple

from training.ppo_transformer.lib.config import Config
from training.ppo_transformer.lib.model import ViNTActorCritic
from training.ppo_transformer.lib.simple_model import SimpleModel
from training.ppo_transformer.lib.env_wrappers import make_env
    

class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, n_steps: int, n_envs: int, obs_shape: Tuple, action_dim: int, gamma: float, gae_lambda: float):
        self.n_steps = n_steps
        self.n_envs = n_envs
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Store observations as uint8 to save memory
        self.observations = torch.zeros((n_steps, n_envs, *obs_shape), dtype=torch.uint8)
        self.actions = torch.zeros((n_steps, n_envs, action_dim))
        self.logprobs = torch.zeros((n_steps, n_envs))
        self.rewards = torch.zeros((n_steps, n_envs))
        self.dones = torch.zeros((n_steps, n_envs))
        self.values = torch.zeros((n_steps, n_envs))
        
        self.ptr = 0
        self.full = False

        # Value predictions for the observation that follows the final action of the rollout.
        # Shape: [n_envs]
        self.last_values = torch.zeros(n_envs)
    
    def add(self, obs, action, logprob, reward, done, value):
        # Ensure observations are stored as uint8 (0-255) regardless of input dtype
        if obs.dtype == torch.uint8:
            obs_uint8 = obs
        else:
            # Assume observations are in the range [0, 1] if floating point
            obs_uint8 = torch.clamp((obs * 255.0).round(), 0, 255).to(torch.uint8)

        self.observations[self.ptr] = obs_uint8
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.n_steps
        if self.ptr == 0:
            self.full = True
    
    def set_last_values(self, last_values: torch.Tensor):
        """Store value predictions for the observations that follow the last action in the rollout."""
        self.last_values = last_values.clone()

    def get(self, device):
        """Get all data and compute advantages using the stored last_values for boot-strapping."""
        # Calculate advantages using GAE
        advantages = torch.zeros_like(self.rewards)
        last_advantage = 0
        
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                # Bootstrap with the value prediction of the observation following the last action
                next_value = self.last_values
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + self.values
        
        # Flatten batch dimensions and convert observations back to float32
        b_obs = self.observations.flatten(0, 1)
        b_actions = self.actions.flatten(0, 1)
        b_logprobs = self.logprobs.flatten(0, 1)
        b_advantages = advantages.flatten(0, 1)
        b_returns = returns.flatten(0, 1)
        b_values = self.values.flatten(0, 1)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        return b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values


class PPOTrainer:
    """PPO Trainer for Snake Environment"""
    
    def __init__(self, config: Config, model: torch.nn.Module):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup TensorBoard logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(config.log_dir, f"snake_ppo_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Create environments
        self.envs = [make_env(config)() for _ in range(config.n_envs)]

        self.model = model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Initialize buffer
        obs_shape = (config.frame_stack, config.output_height, config.output_width, config.n_channels)
        self.buffer = RolloutBuffer(config.n_steps, config.n_envs, obs_shape, config.action_dim, config.gamma, config.gae_lambda)
        
        # Initialize tracking
        self.global_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'clip_fraction': [],
            'explained_variance': []
        }
        
    def collect_rollouts(self):
        """Collect rollouts from environments"""
        # Get initial observations
        observations = []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
        observations = torch.from_numpy(np.array(observations)) # [n_env, seq_len, height, width, n_channels]
        
        # Track episode metrics
        episode_rewards = [0] * self.config.n_envs
        episode_lengths = [0] * self.config.n_envs

        self.model.eval()
        
        for step in range(self.config.n_steps):
            with torch.no_grad():
                actions, logprobs, entropy, values = self.model.get_action_and_value(observations.to(self.device))
            
            # Take actions in environments
            next_observations = []
            rewards = []
            dones = []
            
            for i, env in enumerate(self.envs):
                obs, reward, done, truncated, info = env.step(actions[i].cpu().numpy())
                
                # Track episode metrics
                episode_rewards[i] += reward
                episode_lengths[i] += 1
                
                if done or truncated:
                    # Log completed episode
                    self.episode_rewards.append(episode_rewards[i])
                    self.episode_lengths.append(episode_lengths[i])
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0
                    obs, info = env.reset()
                
                next_observations.append(obs)
                rewards.append(reward)
                dones.append(done or truncated)
            
            # Store in buffer
            self.buffer.add(
                observations.cpu(),
                actions.cpu(),
                logprobs.cpu(),
                torch.FloatTensor(rewards),
                torch.FloatTensor(dones),
                values.cpu().squeeze()
            )
            
            observations = torch.from_numpy(np.array(next_observations))
            self.global_step += self.config.n_envs

        # After collecting the rollout, compute the value prediction for the observations that follow the
        # final action. These are used to bootstrap the Generalised Advantage Estimation (GAE).
        with torch.no_grad():
            _, _, _, next_values = self.model.get_action_and_value(observations.to(self.device))

        self.model.train()

        # Store on CPU for the buffer (shape: [n_envs])
        self.buffer.set_last_values(next_values.cpu().squeeze())
    
    def update_policy(self):
        """Update policy using PPO"""
        b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = self.buffer.get(self.device)
        
        # Training metrics for this update
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropy_losses = []
        epoch_total_losses = []
        epoch_clip_fractions = []
        epoch_explained_variances = []

        self.model.train()
        
        # Training loop
        for epoch in range(self.config.ppo_epochs):
            # Create mini-batches
            batch_size = len(b_obs)
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                mb_indices = indices[start:end]
                
                # Get mini-batch data
                mb_obs = b_obs[mb_indices].to(self.device)
                mb_actions = b_actions[mb_indices].to(self.device)
                mb_logprobs = b_logprobs[mb_indices].to(self.device)
                mb_advantages = b_advantages[mb_indices].to(self.device)
                mb_returns = b_returns[mb_indices].to(self.device)
                mb_values = b_values[mb_indices].to(self.device)
                
                # Forward pass
                newlogprob, entropy, newvalue = self.model.evaluate_actions(mb_obs, mb_actions)
                
                # Calculate losses
                logratio = newlogprob - mb_logprobs
                # Clamp the log-ratio to a reasonable range before exponentiating to
                # avoid numerical overflow which can propagate NaNs through the loss
                # computation and corrupt the network weights.
                logratio_clamped = logratio.clamp(-10.0, 10.0)
                ratio = logratio_clamped.exp()
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                                       1 + self.config.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = ((newvalue.squeeze() - mb_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss + self.config.value_coef * v_loss - self.config.entropy_coef * entropy_loss
                
                # Calculate additional metrics
                clip_fraction = (abs(ratio - 1) > self.config.clip_epsilon).float().mean()
                explained_variance = 1 - torch.var(mb_returns - newvalue.squeeze()) / torch.var(mb_returns)
                
                # Store metrics
                epoch_policy_losses.append(pg_loss.item())
                epoch_value_losses.append(v_loss.item())
                epoch_entropy_losses.append(entropy_loss.item())
                epoch_total_losses.append(loss.item())
                epoch_clip_fractions.append(clip_fraction.item())
                epoch_explained_variances.append(explained_variance.item())
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
        
        # Store average metrics for this update
        self.training_metrics['policy_loss'].append(np.mean(epoch_policy_losses))
        self.training_metrics['value_loss'].append(np.mean(epoch_value_losses))
        self.training_metrics['entropy_loss'].append(np.mean(epoch_entropy_losses))
        self.training_metrics['total_loss'].append(np.mean(epoch_total_losses))
        self.training_metrics['clip_fraction'].append(np.mean(epoch_clip_fractions))
        self.training_metrics['explained_variance'].append(np.mean(epoch_explained_variances))
    
    def log_metrics(self, iteration):
        """Log metrics to TensorBoard"""
        # Log training metrics
        if len(self.training_metrics['policy_loss']) > 0:
            self.writer.add_scalar('Loss/Policy', self.training_metrics['policy_loss'][-1], iteration)
            self.writer.add_scalar('Loss/Value', self.training_metrics['value_loss'][-1], iteration)
            self.writer.add_scalar('Loss/Entropy', self.training_metrics['entropy_loss'][-1], iteration)
            self.writer.add_scalar('Loss/Total', self.training_metrics['total_loss'][-1], iteration)
            self.writer.add_scalar('Metrics/ClipFraction', self.training_metrics['clip_fraction'][-1], iteration)
            self.writer.add_scalar('Metrics/ExplainedVariance', self.training_metrics['explained_variance'][-1], iteration)
        
        # Log episode metrics (last 100 episodes for rolling average)
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]
            recent_lengths = self.episode_lengths[-100:]
            
            self.writer.add_scalar('Episode/Reward_Mean', np.mean(recent_rewards), iteration)
            self.writer.add_scalar('Episode/Reward_Std', np.std(recent_rewards), iteration)
            self.writer.add_scalar('Episode/Reward_Max', np.max(recent_rewards), iteration)
            self.writer.add_scalar('Episode/Reward_Min', np.min(recent_rewards), iteration)
            
            self.writer.add_scalar('Episode/Length_Mean', np.mean(recent_lengths), iteration)
            self.writer.add_scalar('Episode/Length_Std', np.std(recent_lengths), iteration)
        
        # Log learning rate
        self.writer.add_scalar('Training/LearningRate', self.config.learning_rate, iteration)
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.writer.add_scalar('Model/TotalParameters', total_params, iteration)
        self.writer.add_scalar('Model/TrainableParameters', trainable_params, iteration)
        
        # Log gradients norm
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.writer.add_scalar('Training/GradientNorm', total_norm, iteration)
        
        # Flush writer
        self.writer.flush()
    
    def train(self):
        """Main training loop"""
        iteration = 0
        
        try:
            while self.global_step < self.config.total_timesteps:
                # Collect rollouts
                print(f"Collecting rollouts...")
                self.collect_rollouts()
                
                # Update policy
                print(f"Updating policy...")
                self.update_policy()
                
                iteration += 1
                
                # Logging
                if iteration % self.config.log_interval == 0:
                    print(f"Iteration {iteration}, Global Step {self.global_step}")
                    self.log_metrics(iteration)
                
                # Save model
                if iteration % self.config.save_interval == 0:
                    torch.save(self.model.state_dict(), f"snake_ppo_model_{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        
        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            # Close TensorBoard writer
            self.writer.close()
            print("TensorBoard logging closed")

def main():
    parser = argparse.ArgumentParser(description="PPO Training for Snake Environment")
    parser.add_argument("--log_dir", type=str, default="logs/tensorboard")
    parser.add_argument("--model", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.log_dir = args.log_dir

    # model = SimpleModel(config)
    model = ViNTActorCritic(config)
    if args.model is not None:
        try:
            model.load_state_dict(torch.load(args.model, weights_only=True))
            print(f"Loaded model from {args.model}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Initialize trainer
    trainer = PPOTrainer(config, model)
    
    # Start training
    print("Starting PPO training...")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"TensorBoard logs will be saved to: {os.path.join(config.log_dir, 'snake_ppo_*')}")
    trainer.train()


if __name__ == "__main__":
    main()