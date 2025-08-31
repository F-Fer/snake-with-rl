from gymnasium.envs.registration import register

register(
    id='Snake-v0',
    entry_point='snake_env.snake_env:SnakeEnv',
    max_episode_steps=100_000,
)
