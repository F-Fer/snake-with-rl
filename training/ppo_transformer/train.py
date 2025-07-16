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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training for Snake Environment")
    parser.add_argument("--log_dir", type=str, default="logs/tensorboard")
    parser.add_argument("--model", type=str, default=None)
    
    args = parser.parse_args()

    run_name = f"snake_ppo_{int(time.time())}"
    print(f"Run name: {run_name}")
    
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

    # Observations of the first env for video recording
    first_obs = []

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

            if config.record_video:
                # Extract the most recent frame from the stacked observation of the first environment
                obs_np = next_obs[0].cpu().numpy()

                # If the observation is a stack of frames (shape: [stack, H, W, C]), grab the last frame
                frame = obs_np[-1] if obs_np.ndim == 4 else obs_np  # Handles both stacked and single-frame cases

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
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

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
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
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
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if global_step % config.save_interval == 0:
            os.makedirs(config.save_dir, exist_ok=True)
            torch.save(agent.state_dict(), f"{config.save_dir}/run_{run_name}_step_{global_step}.pth")
            print(f"Saved model to {config.save_dir}/run_{run_name}_step_{global_step}.pth")

    envs.close()
    writer.close()
    print("Training complete")