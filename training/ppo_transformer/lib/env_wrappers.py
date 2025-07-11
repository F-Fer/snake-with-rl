import gymnasium as gym
from training.ppo_transformer.lib.config import Config
from snake_env.envs.snake_env import SnakeEnv

def make_env(config: Config):
    """Create environment factory"""
    def _init():
        env = gym.make('Snake-v0', screen_width=config.frame_width, screen_height=config.frame_height, zoom_level=1.0)
        env = gym.wrappers.ResizeObservation(env, (config.output_height, config.output_width))
        env = gym.wrappers.FrameStackObservation(env, config.frame_stack)
        return env
    return _init