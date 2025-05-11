# Snake.io Environment for Gymnasium

A true Snake.io-style game environment implementing the Gymnasium interface with continuous movement, enemy bots, camera tracking, and mouse controls.

## Installation

Install the environment and its dependencies:

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Playing the Game

Run the play script to play the game using mouse controls:

```bash
python play.py
```

### Controls
- Mouse movement: Control snake direction with realistic turning radius (max 60° per frame)
- Q or ESC: Quit Game

## Features

- True Snake.io gameplay mechanics:
  - Camera follows the snake head (viewport centered on player)
  - You can cross over your own body without dying
  - Limited turning radius (max 60° per frame) like in real Snake.io
  - Large world size (3000x3000 pixels) with boundaries
  - Minimap showing your position in the world

- Multiple enemy bot snakes with AI behavior
- Multiple food items scattered throughout the arena
- Bots respawn after dying, creating constant challenges
- Continuous, smooth snake movement (not grid-based)
- Relative directional controls (15° increments)
- Snake grows longer when eating food
- Mouse-based control similar to snake.io games
- Realistic snake body physics - segments follow the path of the head
- Visual enhancements including color gradients, glowing food, and dynamic eyes
- Bot snakes with different colors to distinguish them

## Bot AI Behavior

Enemy snakes have the following behaviors:
- Seek nearby food within their detection radius
- Avoid walls when approaching boundaries
- Make random direction changes occasionally
- Grow when consuming food
- Respawn after hitting walls or being caught

## Using the Environment

The environment can be used like any other Gymnasium environment:

```python
import gymnasium as gym
import snake_env

# Customize with number of bots, food items, and world size
env = gym.make('Snake-v0', num_bots=5, num_foods=20, world_size=3000)
observation, info = env.reset()

for _ in range(1000):
    # Actions are now relative to current direction
    # 0-11: turn left (-180° to -15°)
    # 12: no turn (0°)
    # 13-23: turn right (+15° to +165°)
    action = 12  # No turn (continue straight)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## Action Space

Actions are integers from 0 to 23, representing angle changes relative to current heading:
- 0: -180° (reverse direction)
- 1: -165°
- ...
- 11: -15° (slight left turn)
- 12: 0° (no turn, continue straight)
- 13: +15° (slight right turn)
- ...
- 23: +165°

Note: The environment limits the maximum turn to 60° per frame, regardless of the action provided.

## Game Mechanics

- Camera always centered on player's snake head
- Player can cross over their own body without dying (like in real snake.io)
- Player dies when colliding with walls or other snakes
- Limited turning radius mimics the feel of real snake.io games
- Eating food grows your snake and increases your score
- Bot snakes also eat food and grow
- Bots die when hitting walls and respawn after a delay
- Minimap shows full world view with player position, food, and bots