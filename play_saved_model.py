import argparse
import os
import time
from collections import deque

import cv2
import imageio
import numpy as np
import torch
import pygame

from training.ppo_transformer.train import Config, ViNTActorCritic
from snake_env.envs.snake_env import SnakeEnv


def preprocess_frame(frame: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Resize the raw RGB frame to the (H, W) needed for the model."""
    # OpenCV expects (W, H) when giving size
    resized = cv2.resize(frame, output_size[::-1], interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def rollout(env: SnakeEnv, model: ViNTActorCritic, device: torch.device, cfg: Config, video_path: str):
    """Play a single episode using the trained model and save it to *video_path*."""
    writer = imageio.get_writer(video_path, fps=30)

    obs, _ = env.reset()
    # Initialise frame stack with the first observation resized
    processed = preprocess_frame(obs, (cfg.output_height, cfg.output_width))
    frame_stack = deque([processed for _ in range(cfg.frame_stack)], maxlen=cfg.frame_stack)

    done, truncated = False, False
    while not (done or truncated):
        # Prepare stacked observation for the model -> shape (1, stack, H, W, C)
        stacked_obs = np.stack(list(frame_stack), axis=0)
        stacked_obs = torch.tensor(stacked_obs, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(stacked_obs)
        action_np = action.squeeze(0).cpu().numpy()

        obs, _, done, truncated, _ = env.step(action_np)

        # Render high-resolution frame for video recording
        frame = env._render_frame()
        writer.append_data(frame)

        # Update frame stack
        processed = preprocess_frame(frame, (cfg.output_height, cfg.output_width))
        frame_stack.append(processed)

    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Play trained Snake RL agent and record videos.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model .pth file")
    parser.add_argument("--output_dir", type=str, default="videos", help="Directory to save rollout videos")
    parser.add_argument("--num_rollouts", type=int, default=3, help="Number of episodes to record")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pygame.init()

    # Prepare config (must match training hyper-parameters)
    cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViNTActorCritic(cfg).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Build environment (match training resolution, but we will down-sample for the model)
    env = SnakeEnv(screen_width=cfg.frame_width, screen_height=cfg.frame_height, zoom_level=1.0)

    for ep in range(args.num_rollouts):
        print(f"Starting rollout {ep + 1}/{args.num_rollouts} …")
        video_file = os.path.join(args.output_dir, f"rollout_{ep + 1}.mp4")
        rollout(env, model, device, cfg, video_file)
        print(f"Saved video to {video_file}")
        # Small delay between episodes to ensure video writer flushes
        time.sleep(1)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main() 