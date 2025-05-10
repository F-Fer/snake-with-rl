from gymnasium.envs.registration import register

register(
    id='Snake-v0',
    entry_point='snake_env.envs:SnakeEnv',
    max_episode_steps=1000,
)
