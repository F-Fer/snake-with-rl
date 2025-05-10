# Snake Environment for Gymnasium

A simple Snake game environment implementing the Gymnasium interface, similar to snake.io style games.

## Installation

Install the environment and its dependencies:

```bash
pip install -e .
```

## Playing the Game

Run the play script to play the game using keyboard controls:

```bash
python play.py
```

### Controls
- ↑: Move Up
- →: Move Right
- ↓: Move Down
- ←: Move Left
- Q: Quit Game

## Using the Environment

The environment can be used like any other Gymnasium environment:

```python
import gymnasium as gym
import snake_env

env = gym.make('Snake-v0')
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```