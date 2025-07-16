import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import argparse
from typing import Tuple
import gymnasium as gym
import time

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
        self.writer = SummaryWriter(log_dir)
        # Log config
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
        )
        
        # Create environments
        self.envs = [make_env(config)() for _ in range(config.n_envs)]

        self.model = model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        
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
    parser = argparse.ArgumentParser(description="PPO Training for Snake Environment")
    parser.add_argument("--log_dir", type=str, default="logs/tensorboard")
    parser.add_argument("--model", type=str, default=None)
    
    args = parser.parse_args()

    run_name = f"snake_ppo_{int(time.time())}"
    
    # Create config
    config = Config()
    config.log_dir = args.log_dir

    # Setup TensorBoard logging
    writer = SummaryWriter(os.path.join(config.log_dir, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
    )

    # Seeding
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(config, config.seed + i, i, run_name) for i in range(config.n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )

    agent = SimpleModel(config).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((config.n_steps, config.n_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config.n_steps, config.n_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.n_steps, config.n_envs)).to(device)
    rewards = torch.zeros((config.n_steps, config.n_envs)).to(device)
    dones = torch.zeros((config.n_steps, config.n_envs)).to(device)
    values = torch.zeros((config.n_steps, config.n_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.n_envs).to(device)
    num_updates = config.total_timesteps // config.batch_size

    for update in range(1, num_updates + 1):
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollouts
        for step in range(config.n_steps):
            global_step += 1 * config.n_envs

            obs[step] = next_obs
            dones[step] = next_done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            print(type(info))
            print(info)
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)

            # Compute advantages (GAE)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.n_steps)):
                if t == config.n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        for epoch in range(config.ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.mini_batch_size):
                end = start + config.mini_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.clip_epsilon).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_epsilon,
                        config.clip_epsilon,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.entropy_coef * entropy_loss + v_loss * config.value_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None:
                if approx_kl > config.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
    print("Training complete")