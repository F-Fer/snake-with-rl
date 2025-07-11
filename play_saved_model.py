import argparse
import os
import time
from collections import deque

import cv2
import imageio
import numpy as np
import torch
import pygame

from training.ppo_transformer.lib.config import Config
from training.ppo_transformer.lib.model import ViNTActorCritic
from training.ppo_transformer.lib.env_wrappers import make_env
from snake_env.envs.snake_env import SnakeEnv


def preprocess_frame(frame: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Resize the raw RGB frame to the (H, W) needed for the model."""
    # Keep uint8 dtype so that the model's ImagePreprocessor handles the conversion
    resized = cv2.resize(frame, output_size[::-1], interpolation=cv2.INTER_AREA)
    return resized  # dtype remains np.uint8


def rollout(env: SnakeEnv, model: ViNTActorCritic, device: torch.device, cfg: Config, video_path: str):
    """Play a single episode using the trained model and save it to *video_path*.

    The key here is to feed the model **exactly** the same observation format that was
    used during training:
        *  uint8 dtype
        *  shape (frame_stack, H, W, C) before adding the batch dimension
        *  resized via the ResizeObservation wrapper that is already applied in
           `make_env`
    """

    writer = imageio.get_writer(video_path, fps=30)

    # Reset environment – `obs` is already stacked & resized by the wrappers
    obs, _ = env.reset()  # shape: (frame_stack, H, W, C) – dtype: uint8

    done, truncated = False, False
    while not (done or truncated):
        # Convert to tensor keeping uint8 so that the model's ImagePreprocessor
        # performs the uint8 → float conversion exactly like during training.
        obs_tensor = torch.tensor(obs, dtype=torch.uint8).unsqueeze(0).to(device)

        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs_tensor)

        # Execute action in the environment
        action_np = action.squeeze(0).cpu().numpy()
        obs, _, done, truncated, _ = env.step(action_np)

        # Render high-resolution frame for the output video (does not affect the
        # model pipeline).
        # Use the underlying SnakeEnv's high-resolution render method
        frame = env.unwrapped._render_frame()
        writer.append_data(frame)

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
    env = make_env(cfg)()
    print(env)

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