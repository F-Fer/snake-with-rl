import gymnasium as gym
from training.ppo_transformer.lib.config import Config
from snake_env.envs.snake_env import SnakeEnv
from typing import Callable

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
    

class RandomActionResetWrapper(gym.Wrapper):
    """
    Sample a random number of random actions from the action space on environment reset.
    This is similar to `gymnasium.wrappers.NoopResetEnv`, but for continuous action spaces.
    """

    def __init__(self, env: gym.Env, max_random_steps: int = 30):
        """
        Args:
            env: The environment to wrap.
            max_random_steps: The maximum number of random actions to take on reset.
        """
        super().__init__(env)
        assert max_random_steps >= 0
        self.max_random_steps = max_random_steps

    def reset(self, **kwargs):
        """
        Resets the environment, then takes a random number of random actions.
        If the environment terminates during the random actions, it is reset again.
        """
        obs, info = self.env.reset(**kwargs)

        if self.max_random_steps > 0:
            num_random_steps = self.np_random.integers(self.max_random_steps + 1)
            for _ in range(num_random_steps):
                action = self.action_space.sample()
                obs, _, terminated, truncated, info = self.env.step(action)
                if terminated or truncated:
                    obs, info = self.env.reset(**kwargs)

        return obs, info


def make_env(config: Config, seed: int, idx: int, run_name: str) -> Callable:
    """Create environment factory"""
    def _init():
        env = gym.make(
            'Snake-v0', 
            screen_width=config.frame_width, 
            screen_height=config.frame_height, 
            zoom_level=1.0, 
            num_bots=config.num_bots, 
            num_foods=config.num_foods,
            world_size=config.world_size)

        if config.random_action_reset:
            env = RandomActionResetWrapper(env, config.max_random_steps)

        # Apply frameskip before other wrappers so they operate on skipped observations
        env = FrameSkipWrapper(env, skip=config.frame_skip)
        env = gym.wrappers.ResizeObservation(env, (config.output_height, config.output_width))
        if config.gray_scale:
            env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        env = gym.wrappers.FrameStackObservation(env, config.frame_stack)
        return env
    return _init