import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import random

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 15}
    
    def __init__(self, render_mode=None, grid_size=20, cell_size=30):
        self.grid_size = grid_size  # Number of cells in each dimension
        self.cell_size = cell_size  # Size of each cell in pixels
        self.window_size = grid_size * cell_size  # Pixel dimensions of the window
        
        # Observation space: the entire grid state
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(self.grid_size, self.grid_size, 3), 
            dtype=np.uint8
        )
        
        # Action space: 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space = spaces.Discrete(4)
        
        # Render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # For rendering
        self.window = None
        self.clock = None
        
        # Initialize game state
        self.reset()
        
    def _get_obs(self):
        # Create an empty grid
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Draw snake body
        for body_part in self.snake_body[1:]:
            x, y = body_part
            # Check if body part is within grid bounds
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                obs[y, x] = [0, 255, 0]  # Green for body
        
        # Draw snake head
        head_x, head_y = self.snake_body[0]
        # Check if head is within grid bounds
        if 0 <= head_x < self.grid_size and 0 <= head_y < self.grid_size:
            obs[head_y, head_x] = [0, 200, 0]  # Darker green for head
        
        # Draw food
        food_x, food_y = self.food_position
        obs[food_y, food_x] = [255, 0, 0]  # Red for food
        
        return obs
    
    def _get_info(self):
        return {
            'snake_length': len(self.snake_body),
            'score': self.score
        }
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize snake position at the center (head is first element)
        center = self.grid_size // 2
        self.snake_body = [(center, center)]
        
        # Initialize snake direction: 0=up, 1=right, 2=down, 3=left
        self.direction = 1
        
        # Generate initial food position
        self._place_food()
        
        # Score and game status
        self.score = 0
        self.steps_without_food = 0
        self.game_over = False
        
        # Reset render components
        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Snake")
        if self.render_mode == "human" and self.clock is None:
            self.clock = pygame.time.Clock()
            
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _place_food(self):
        # Place food at a random position that is not occupied by the snake
        while True:
            food_x = self.np_random.integers(0, self.grid_size)
            food_y = self.np_random.integers(0, self.grid_size)
            food_pos = (food_x, food_y)
            
            if food_pos not in self.snake_body:
                self.food_position = food_pos
                break
    
    def step(self, action):
        # Check if game is already over
        if self.game_over:
            observation = self._get_obs()
            info = self._get_info()
            return observation, 0, True, False, info
        
        # Update direction based on action
        # Ensure snake can't reverse direction
        if action == 0 and self.direction != 2:  # Up and not currently going down
            self.direction = 0
        elif action == 1 and self.direction != 3:  # Right and not currently going left
            self.direction = 1
        elif action == 2 and self.direction != 0:  # Down and not currently going up
            self.direction = 2
        elif action == 3 and self.direction != 1:  # Left and not currently going right
            self.direction = 3
        
        # Move snake
        head_x, head_y = self.snake_body[0]
        if self.direction == 0:  # Up
            new_head = (head_x, head_y - 1)
        elif self.direction == 1:  # Right
            new_head = (head_x + 1, head_y)
        elif self.direction == 2:  # Down
            new_head = (head_x, head_y + 1)
        elif self.direction == 3:  # Left
            new_head = (head_x - 1, head_y)
        
        # Insert new head
        self.snake_body.insert(0, new_head)
        
        # Check for collisions with walls
        head_x, head_y = new_head
        if head_x < 0 or head_x >= self.grid_size or head_y < 0 or head_y >= self.grid_size:
            self.game_over = True
            reward = -10  # Penalty for hitting wall
            terminated = True
        # Check for collision with self
        elif new_head in self.snake_body[1:]:
            self.game_over = True
            reward = -10  # Penalty for hitting self
            terminated = True
        # Check for food collection
        elif new_head == self.food_position:
            self.score += 1
            reward = 10  # Reward for eating food
            self._place_food()
            self.steps_without_food = 0
            terminated = False
        else:
            # Remove tail if no food eaten
            self.snake_body.pop()
            reward = -0.01  # Small penalty for each step to encourage efficiency
            self.steps_without_food += 1
            
            # End episode if snake is wandering too long without food
            if self.steps_without_food > 100 * len(self.snake_body):
                terminated = True
            else:
                terminated = False
        
        # For compatibility with Gymnasium
        truncated = False
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
            
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))  # Fill with black
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake_body):
            # Skip drawing if outside grid
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                continue
                
            color = (0, 200, 0) if i == 0 else (0, 255, 0)  # Different color for head
            pygame.draw.rect(
                canvas, 
                color, 
                pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
            )
            
            # Draw eyes (only for head)
            if i == 0:
                eye_radius = self.cell_size // 8
                eye_offset = self.cell_size // 4
                
                # Calculate eye positions based on direction
                if self.direction == 0:  # Up
                    left_eye = (x * self.cell_size + eye_offset, y * self.cell_size + eye_offset)
                    right_eye = (x * self.cell_size + self.cell_size - eye_offset, y * self.cell_size + eye_offset)
                elif self.direction == 1:  # Right
                    left_eye = (x * self.cell_size + self.cell_size - eye_offset, y * self.cell_size + eye_offset)
                    right_eye = (x * self.cell_size + self.cell_size - eye_offset, y * self.cell_size + self.cell_size - eye_offset)
                elif self.direction == 2:  # Down
                    left_eye = (x * self.cell_size + eye_offset, y * self.cell_size + self.cell_size - eye_offset)
                    right_eye = (x * self.cell_size + self.cell_size - eye_offset, y * self.cell_size + self.cell_size - eye_offset)
                elif self.direction == 3:  # Left
                    left_eye = (x * self.cell_size + eye_offset, y * self.cell_size + eye_offset)
                    right_eye = (x * self.cell_size + eye_offset, y * self.cell_size + self.cell_size - eye_offset)
                
                pygame.draw.circle(canvas, (255, 255, 255), left_eye, eye_radius)
                pygame.draw.circle(canvas, (255, 255, 255), right_eye, eye_radius)
                pygame.draw.circle(canvas, (0, 0, 0), left_eye, eye_radius // 2)
                pygame.draw.circle(canvas, (0, 0, 0), right_eye, eye_radius // 2)
        
        # Draw food
        food_x, food_y = self.food_position
        pygame.draw.rect(
            canvas, 
            (255, 0, 0), 
            pygame.Rect(
                food_x * self.cell_size, 
                food_y * self.cell_size, 
                self.cell_size, 
                self.cell_size
            )
        )
        
        # Draw grid lines
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(canvas, (50, 50, 50), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(canvas, (50, 50, 50), (0, y), (self.window_size, y))
            
        # Display score
        if pygame.font:
            font = pygame.font.Font(None, 36)
            text = font.render(f"Score: {self.score}", True, (255, 255, 255))
            canvas.blit(text, (10, 10))
            
        if self.render_mode == "human":
            # Copy canvas to window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            
            # Ensure consistent frame rate
            self.clock.tick(self.metadata["render_fps"])
            
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None