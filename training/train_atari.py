import ale_py as ale
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
import logging

from lib.atari_config import Config
from lib.atari_model import SimpleModel, RNDModule
from lib.env_wrappers import make_atari_env

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def compute_gae(rewards, values, next_value, dones, gamma, gae_lambda, device):
    advantages = torch.zeros_like(rewards).to(device)
    lastgaelam = 0
    n_steps = rewards.shape[0]
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            nextnonterminal = 1.0 - dones[-1]  # Use last done state
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training for Snake Environment")
    parser.add_argument("--log_dir", type=str, default="logs/tensorboard")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5")
    
    args = parser.parse_args()

    gym.register_envs(ale)

    run_name = f"ppo_run_{int(time.time())}"
    logger.info(f"Run name: {run_name}")
    
    # Create config
    config = Config()
    config.log_dir = args.log_dir
    config.env_name = args.env_name if args.env_name is not None else config.env_name

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
        [make_atari_env(config) for i in range(config.n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
    )

    print(envs.single_observation_space.shape)
    print(envs.action_space)
 
    config.n_channels = envs.single_observation_space.shape[0]
    config.action_dim = envs.action_space.shape[0]

    agent = SimpleModel(config).to(device)
    if config.rnd_enabled:
        rnd_module = RNDModule(config).to(device)
        rnd_optimizer = optim.Adam(rnd_module.predictor_net.parameters(), lr=config.rnd_learning_rate)
    else:
        rnd_module = None
        rnd_optimizer = None

    if args.model is not None:
        agent.load_state_dict(torch.load(args.model))
    agent.train()

    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((config.n_steps, config.n_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config.n_steps, config.n_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.n_steps, config.n_envs)).to(device)
    rewards = torch.zeros((config.n_steps, config.n_envs)).to(device) # Extrinsic rewards
    dones = torch.zeros((config.n_steps, config.n_envs)).to(device)
    values = torch.zeros((config.n_steps, config.n_envs)).to(device)
    rewards_int = torch.zeros((config.n_steps, config.n_envs)).to(device)  # Intrinsic rewards (RND)
    if config.use_dual_value_heads:
        values_ext = torch.zeros((config.n_steps, config.n_envs)).to(device)
        values_int = torch.zeros((config.n_steps, config.n_envs)).to(device)

    # Observations of the first env for video recording
    first_obs = []

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.n_envs).to(device)
    num_updates = config.total_timesteps // config.batch_size

    for update in range(1, num_updates + 1):
        agent.reset_noise()
        
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Decay exploration parameters
        decay_frac = 1.0 - (update - 1.0) / num_updates
        
        # Anneal entropy coefficient
        if config.anneal_entropy_coef:
            current_entropy_coef = config.min_entropy_coef + (config.entropy_coef - config.min_entropy_coef) * decay_frac
        else:
            current_entropy_coef = config.entropy_coef
            
        # Anneal RND intrinsic coefficient
        if config.anneal_rnd_coef and config.rnd_enabled:
            current_rnd_coef = config.min_rnd_intrinsic_coef + (config.rnd_intrinsic_coef - config.min_rnd_intrinsic_coef) * decay_frac
        else:
            current_rnd_coef = config.rnd_intrinsic_coef

        # Collect rollouts
        for step in range(config.n_steps):
            global_step += 1 * config.n_envs

            obs[step] = next_obs
            dones[step] = next_done

            if config.record_video:
                # Extract the most recent frame from the stacked observation of the first environment
                obs_np = next_obs[0].cpu().numpy()

                # If the observation is a stack of frames (shape: [stack, H, W, C]), grab the last frame
                frame = obs_np[-1] if obs_np.ndim == 4 else obs_np  # Handles both stacked and single-frame cases

                # Convert grayscale to RGB if necessary
                if frame.shape[-1] == 1:  # Grayscale
                    frame = np.repeat(frame, 3, axis=-1)  # Convert to RGB by repeating the channel

                # Accumulate frames for this episode
                first_obs.append(frame)

                # Once the episode for env-0 terminates, write the video to TensorBoard
                if next_done[0].item():
                    if len(first_obs) > 0:
                        # Convert collected frames (T, H, W, C) -> (N=1, C, T, H, W)
                        video_frames = np.array(first_obs, dtype=np.uint8)
                        video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W)

                        # Add batch dimension (N=1) and log video to TensorBoard.
                        writer.add_video(
                            "video/episode_0",
                            video_tensor,  # (1, C, T, H, W)
                            global_step=global_step,
                            fps=3
                        )

                    # Clear stored frames for the next episode
                    first_obs.clear()

            # Action logic
            with torch.no_grad():
                if config.rnd_enabled and config.use_dual_value_heads:
                    action, logprob, _, value, value_ext, value_int = agent.get_action_and_value(next_obs)
                    values_ext[step] = value_ext.flatten()
                    values_int[step] = value_int.flatten()
                    values[step] = value.flatten() # ext + int
                else:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)

            next_obs_tensor = torch.Tensor(next_obs).to(device)

            # Compute intrinsic reward (RND)
            if config.rnd_enabled:
                with torch.no_grad():
                    intrinsic_reward = rnd_module(next_obs_tensor)
                    # Clip intrinsic rewards before scaling to prevent instability
                    if hasattr(config, 'rnd_clip_intrinsic_reward'):
                        intrinsic_reward = torch.clamp(intrinsic_reward, max=config.rnd_clip_intrinsic_reward)
                    intrinsic_reward = current_rnd_coef * intrinsic_reward
                    rewards_int[step] = intrinsic_reward.view(-1)

            next_obs, next_done = next_obs_tensor, torch.Tensor(done).to(device)

            # info is a dict of lists, each list entry containing the episode info for the corresponding environment\
            if "episode_done" in info.keys() and "episode_length" in info.keys() and "episode_return" in info.keys():
                done_idx = info["episode_done"]
                episode_lengths = info["episode_length"]
                episode_returns = info["episode_return"]
                if np.any(done_idx):
                    for i in range(config.n_envs):
                        if done_idx[i]:
                            # Only log the first episode that terminates
                            print(f"global_step={global_step}, episodic_return={episode_returns[i]}, episodic_length={episode_lengths[i]}")
                            writer.add_scalar("charts/episodic_return", episode_returns[i], global_step)
                            writer.add_scalar("charts/episodic_length", episode_lengths[i], global_step)
                            break
            

        # Advantages and returns
        if config.rnd_enabled and config.use_dual_value_heads:
            with torch.no_grad():
                next_value_ext = agent.get_value_ext(next_obs).reshape(1, -1)
                next_value_int = agent.get_value_int(next_obs).reshape(1, -1)
                
                # Extrinsic advantages (episodic)
                advantages_ext = compute_gae(rewards, values_ext, next_value_ext, dones, config.gamma, config.gae_lambda, device)
                returns_ext = advantages_ext + values_ext
                
                # Intrinsic advantages (non-episodic)
                advantages_int = compute_gae(rewards_int, values_int, next_value_int, torch.zeros_like(dones), config.rnd_gamma, config.gae_lambda, device)
                returns_int = advantages_int + values_int
                
                # Combined advantages
                advantages = advantages_ext + advantages_int
                returns = returns_ext + returns_int
        else:
            # Single value head case
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
            total_rewards = rewards + (rewards_int if config.rnd_enabled else 0)
            advantages = compute_gae(total_rewards, values, next_value, dones, config.gamma, config.gae_lambda, device)
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        if config.use_dual_value_heads:
            b_returns_ext = returns_ext.reshape(-1)
            b_returns_int = returns_int.reshape(-1)
            b_values_ext = values_ext.reshape(-1)
            b_values_int = values_int.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        for epoch in range(config.ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.mini_batch_size):
                end = start + config.mini_batch_size
                mb_inds = b_inds[start:end]

                if config.use_dual_value_heads:
                    _, newlogprob, entropy, _, newvalue_ext, newvalue_int = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    newvalue = newvalue_ext + newvalue_int
                else:
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
                if config.use_dual_value_heads:
                    if config.clip_vloss:
                        # For dual heads, compute separate losses but don't clip them separately
                        v_loss_ext = 0.5 * ((newvalue_ext.view(-1) - b_returns_ext[mb_inds]) ** 2).mean()
                        v_loss_int = 0.5 * ((newvalue_int.view(-1) - b_returns_int[mb_inds]) ** 2).mean()
                        v_loss = v_loss_ext + v_loss_int
                    else:
                        v_loss_ext = 0.5 * ((newvalue_ext.view(-1) - b_returns_ext[mb_inds]) ** 2).mean()
                        v_loss_int = 0.5 * ((newvalue_int.view(-1) - b_returns_int[mb_inds]) ** 2).mean()
                        v_loss = v_loss_ext + v_loss_int
                else:
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
                loss = pg_loss - current_entropy_coef * entropy_loss + v_loss * config.value_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

                # Train RND predictor
                if config.rnd_enabled:
                    # Sample subset of data for RND training
                    rnd_sample_size = int(len(mb_inds) * config.rnd_update_proportion)
                    rnd_inds = np.random.choice(mb_inds, rnd_sample_size, replace=False)

                    rnd_input = rnd_module.preproc(b_obs[rnd_inds])
                    rnd_input = rnd_module.normalize_obs(rnd_input)
                    
                    # Compute RND loss
                    predicted_features = rnd_module.predictor_net(rnd_input)
                    with torch.no_grad():
                        target_features = rnd_module.target_net(rnd_input)
                    
                    rnd_loss = 0.5 * (predicted_features - target_features).pow(2).mean()
                    
                    # Update RND predictor
                    rnd_optimizer.zero_grad()
                    rnd_loss.backward()
                    nn.utils.clip_grad_norm_(rnd_module.predictor_net.parameters(), config.max_grad_norm)
                    rnd_optimizer.step()

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
        writer.add_scalar("losses/entropy", - entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("charts/action_std", agent.actor_logstd.exp().mean().item(), global_step)
        writer.add_scalar("charts/entropy_coef", current_entropy_coef, global_step)
        if config.rnd_enabled:
            writer.add_scalar("charts/rnd_intrinsic_coef", current_rnd_coef, global_step)

        if config.rnd_enabled:
            writer.add_scalar("rnd/intrinsic_reward_mean", rewards_int.mean().item(), global_step)
            writer.add_scalar("rnd/intrinsic_reward_std", rewards_int.std().item(), global_step)
            writer.add_scalar("rnd/predictor_loss", rnd_loss.item(), global_step)
            writer.add_scalar("rnd/reward_running_std", torch.sqrt(rnd_module.reward_running_var).item(), global_step)

        if global_step % config.save_interval == 0:
            os.makedirs(config.save_dir, exist_ok=True)
            torch.save(agent.state_dict(), f"{config.save_dir}/run_{run_name}_step_{global_step}.pth")
            print(f"Saved model to {config.save_dir}/run_{run_name}_step_{global_step}.pth")

    envs.close()
    writer.close()
    print("Training complete")