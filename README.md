# Snake.io Gymnasium Environment

This project implements a Gymnasium environment for a Snake.io-style game, designed for training Reinforcement Learning models. It features continuous movement, enemy bots, dynamic camera, and a Slither.io-like gameplay experience.

## Installation

### Using `uv` (Recommended)

`uv` is a fast Python package installer and resolver.

1.  Create and activate a virtual environment:
    ```bash
    uv venv
    source .venv/bin/activate 
    # On Windows, use: .venv\Scripts\activate
    ```
2.  Install the package and its dependencies:
    ```bash
    uv pip install -e .
    ```

### Using `pip`

1.  Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate 
    # On Windows, use: .venv\Scripts\activate
    ```
2.  Install the package and its dependencies:
    ```bash
    pip install -e .
    ```

## Playing the Game

Run the `play.py` script to interact with the environment:

```bash
python play.py
```

### Controls
-   **Mouse Movement**: Controls the snake's target direction. The snake will smoothly turn towards the cursor.
-   **Q or ESC**: Quit the game.

## Environment Specifications

-   **Action Space**: Continuous, `Box(low=-1, high=1, shape=(2,), dtype=np.float32)`. Represents `[cosine_value, sine_value]` for the target direction.
-   **Observation Space**: Image, `Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8)`. Default is 84x84x3, but configurable via the `screen_size` parameter in `gym.make()`.
-   **Rewards**:
    -   Change in Length: `+C1 * (current_length - previous_length)`
    -   Eliminating an Opponent: `+C2 * (opponent_snake_length)`
    -   Death (collision with wall or longer/equal length snake): `-C3`
    -   (Constants `C1`, `C2`, `C3` are defined in `SnakeEnv` class, currently `0.1`, `10`, `100` respectively)

## Using the Environment with Gymnasium

```python
import gymnasium as gym
import numpy as np
import snake_env # Registers 'Snake-v0'

# Environment parameters can be customized:
env_params = {
    "world_size": 3000,
    "snake_segment_radius": 10,
    "num_bots": 3,
    "num_foods": 15,
    "screen_size": 84,  # For the observation image dimensions (HxW)
    "zoom_level": 1.0   # 1.0 = normal, >1.0 = zoomed out, <1.0 = zoomed in
}
env = gym.make('Snake-v0', **env_params)

observation, info = env.reset()

for _ in range(1000):
    # A simple random action:
    random_angle = np.random.uniform(0, 2 * np.pi)
    action = np.array([np.cos(random_angle), np.sin(random_angle)], dtype=np.float32)
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"Reward: {reward:.2f}, Length: {info['snake_length']}, Score: {info['score']}")

    if terminated or truncated:
        print("Episode finished!")
        observation, info = env.reset()

env.close()
```

## Key Game Mechanics

-   **Camera**: Always centered on the player's snake head. The `zoom_level` parameter affects how much of the world is captured in the observation.
-   **Movement**: Continuous. The snake attempts to turn towards the angle specified by the action vector, limited by its current agility (which depends on its length).
-   **Collisions**:
    -   Player snake dies if its head collides with a wall.
    -   Player snake dies if its head collides with an enemy snake's body.
    -   In head-to-head collisions between player and bot:
        -   If player is longer: bot dies, player gets reward.
        -   If bot is longer or equal length: player dies.
-   **Growth**: Eating food increases the snake's length and score. Bot snakes also eat and grow.
-   **Bots**: AI-controlled snakes that seek food, avoid walls, and can be eliminated. They respawn after a delay.