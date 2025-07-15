import gymnasium as gym
from training.ppo_transformer.lib.config import Config
from snake_env.envs.snake_env import SnakeEnv

class FrameSkipWrapper(gym.Wrapper):
    """Repeat the same action for `skip` frames and accumulate rewards."""

    def __init__(self, env: gym.Env, skip: int):
        super().__init__(env)
        assert skip >= 1, "frameskip value must be >= 1"
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):  # type: ignore[override]
        return self.env.reset(**kwargs)


def make_env(config: Config, seed: int, idx: int, run_name: str):
    """Create environment factory"""
    def _init():
        env = gym.make('Snake-v0', screen_width=config.frame_width, screen_height=config.frame_height, zoom_level=1.0)
        # Apply frameskip before other wrappers so they operate on skipped observations
        env = FrameSkipWrapper(env, skip=config.frame_skip)
        env = gym.wrappers.ResizeObservation(env, (config.output_height, config.output_width))
        env = gym.wrappers.FrameStackObservation(env, config.frame_stack)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if config.record_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env
    return _init