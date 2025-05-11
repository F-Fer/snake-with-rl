import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import random
import math
import colorsys

class BotSnake:
    def __init__(self, width, height, segment_radius, np_random, color_hue):
        self.segment_radius = segment_radius
        self.speed = 2.5  # Slightly slower than player
        self.min_distance_between_segments = segment_radius * 1.5
        self.width = width
        self.height = height
        self.np_random = np_random
        
        # Random starting position, away from the edges
        margin = width * 0.2
        x = np_random.uniform(margin, width - margin)
        y = np_random.uniform(margin, height - margin)
        
        # Start with a single segment (head)
        self.body = [(float(x), float(y))]
        
        # Random initial direction
        self.direction_angle = np_random.uniform(0, 2 * math.pi)
        self.direction_vector = (math.cos(self.direction_angle), math.sin(self.direction_angle))
        
        # Add initial body segments
        self._add_initial_segments(3)
        
        # Set color based on the provided hue
        r, g, b = colorsys.hsv_to_rgb(color_hue, 0.8, 0.9)
        self.base_color = (int(r * 255), int(g * 255), int(b * 255))
        
        # Bot behavior settings
        self.direction_change_chance = 0.02  # Chance of randomly changing direction
        self.food_detection_radius = 150  # Radius to detect food
        self.wall_avoidance_distance = 60  # Distance to start avoiding walls
        self.last_direction_change = 0
    
    def _add_initial_segments(self, count):
        # Add initial body segments behind the head
        head_x, head_y = self.body[0]
        angle = self.direction_angle + math.pi  # Opposite direction of head
        
        for i in range(1, count + 1):
            offset = i * self.min_distance_between_segments
            x = head_x + math.cos(angle) * offset
            y = head_y + math.sin(angle) * offset
            self.body.append((float(x), float(y)))
    
    def update(self, food_positions, player_snake, other_bots):
        # Bot AI decision making
        head_x, head_y = self.body[0]
        
        # Check if it's time to change direction randomly
        if self.np_random.random() < self.direction_change_chance:
            # Random angle change between -45 and 45 degrees
            angle_change = self.np_random.uniform(-math.pi/4, math.pi/4)
            self.direction_angle += angle_change
            self.direction_vector = (math.cos(self.direction_angle), math.sin(self.direction_angle))
        
        # Food seeking behavior
        closest_food = None
        closest_distance = float('inf')
        
        for food_pos in food_positions:
            dist = math.sqrt((head_x - food_pos[0])**2 + (head_y - food_pos[1])**2)
            if dist < closest_distance:
                closest_distance = dist
                closest_food = food_pos
        
        if closest_food and closest_distance < self.food_detection_radius:
            # Calculate angle to food
            food_angle = math.atan2(closest_food[1] - head_y, closest_food[0] - head_x)
            
            # Gradually steer towards food
            # Convert both angles to [0, 2π) range
            current_angle = self.direction_angle % (2 * math.pi)
            target_angle = food_angle % (2 * math.pi)
            
            # Calculate the smallest angle difference
            angle_diff = target_angle - current_angle
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Adjust direction slightly towards food
            self.direction_angle += angle_diff * 0.1
            self.direction_vector = (math.cos(self.direction_angle), math.sin(self.direction_angle))
        
        # Wall avoidance
        # Check if approaching a wall and adjust direction
        if (head_x < self.wall_avoidance_distance or 
            head_x > self.width - self.wall_avoidance_distance or
            head_y < self.wall_avoidance_distance or 
            head_y > self.height - self.wall_avoidance_distance):
            
            # Calculate vector towards center of arena
            center_x, center_y = self.width / 2, self.height / 2
            to_center_angle = math.atan2(center_y - head_y, center_x - head_x)
            
            # Blend current direction with direction to center
            weight = 0.2  # How strongly to pull towards center
            self.direction_angle = (1 - weight) * self.direction_angle + weight * to_center_angle
            self.direction_vector = (math.cos(self.direction_angle), math.sin(self.direction_angle))
        
        # Move snake head
        new_head_x = head_x + self.direction_vector[0] * self.speed
        new_head_y = head_y + self.direction_vector[1] * self.speed
        
        # Update head position
        self.body[0] = (new_head_x, new_head_y)
        
        # Move each body segment towards the segment in front of it
        for i in range(len(self.body) - 1, 0, -1):
            target_x, target_y = self.body[i-1]
            current_x, current_y = self.body[i]
            
            # Calculate direction vector to target
            dx = target_x - current_x
            dy = target_y - current_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > self.min_distance_between_segments:
                # Move segment towards the one in front of it
                move_ratio = self.speed / distance
                new_x = current_x + dx * move_ratio
                new_y = current_y + dy * move_ratio
                self.body[i] = (new_x, new_y)
        
        # Check for collisions with walls and return True if bot should be removed
        head_x, head_y = self.body[0]
        if (head_x - self.segment_radius < 0 or 
            head_x + self.segment_radius > self.width or 
            head_y - self.segment_radius < 0 or 
            head_y + self.segment_radius > self.height):
            return True  # Bot hit wall, should be removed
        
        # Check for collision with player's head - bots die if they hit player's head
        player_head = player_snake[0]
        head_to_player_head_dist = math.sqrt((head_x - player_head[0])**2 + (head_y - player_head[1])**2)
        
        if head_to_player_head_dist < self.segment_radius * 1.8:
            # Head-to-head collision
            # If player's snake is longer, player wins
            if len(player_snake) >= len(self.body):
                return True  # Bot should be removed
        
        # Check for collisions with player's body (skipping head)
        for player_segment in player_snake[1:]:
            dist = math.sqrt((head_x - player_segment[0])**2 + (head_y - player_segment[1])**2)
            if dist < self.segment_radius * 1.5:
                return True  # Bot hit player body, should be removed
        
        # Check for collisions with other bots (only their bodies, not heads)
        for other_bot in other_bots:
            # Skip empty bots
            if not other_bot.body:
                continue
                
            # Check this bot's head against other bot's body segments (skipping head)
            for other_segment in other_bot.body[1:]:
                dist = math.sqrt((head_x - other_segment[0])**2 + (head_y - other_segment[1])**2)
                if dist < self.segment_radius * 1.5:
                    return True  # Bot hit another bot, should be removed
        
        # Check for food collection
        eaten_food_indices = []
        for i, food_pos in enumerate(food_positions):
            food_x, food_y = food_pos
            food_distance = math.sqrt((head_x - food_x)**2 + (head_y - food_y)**2)
            
            if food_distance < self.segment_radius + self.segment_radius * 0.8:  # Food radius is 0.8 * segment_radius
                eaten_food_indices.append(i)
                
                # Add a new segment at the end
                if len(self.body) > 1:
                    last_segment = self.body[-1]
                    second_last_segment = self.body[-2]
                    
                    # Direction from second-last to last segment
                    dx = last_segment[0] - second_last_segment[0]
                    dy = last_segment[1] - second_last_segment[1]
                    
                    # Normalize and multiply by segment distance
                    length = math.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        dx /= length
                        dy /= length
                    
                    # New segment position
                    new_segment_x = last_segment[0] + dx * self.min_distance_between_segments
                    new_segment_y = last_segment[1] + dy * self.min_distance_between_segments
                    
                    self.body.append((new_segment_x, new_segment_y))
        
        return False, eaten_food_indices  # Bot is alive, return which food it ate

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 15}
    
    def __init__(self, render_mode=None, screen_size=800, world_size=3000, snake_segment_radius=10, num_bots=3, num_foods=10):
        # Screen dimensions (viewport)
        self.screen_size = screen_size
        
        # World dimensions (larger than screen)
        self.world_width = world_size
        self.world_height = world_size
        
        # Camera/viewport settings
        self.camera_x = 0
        self.camera_y = 0
        
        self.snake_segment_radius = snake_segment_radius
        self.food_radius = snake_segment_radius * 0.8
        self.snake_speed = 3  # Pixels per step
        self.min_distance_between_segments = snake_segment_radius * 1.5
        
        # Maximum turning angle per frame (in radians)
        self.max_turn_per_frame = math.pi / 3  # 60 degrees
        
        # Bot settings
        self.num_bots = num_bots
        self.num_foods = num_foods
        self.bot_respawn_time = 300  # Number of steps before respawning a bot
        self.bot_respawn_counter = {}  # Maps bot_id to respawn countdown
        
        # Observation space: position of snake head, direction, and closest food
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([self.world_width, self.world_height, 2*math.pi, self.world_width, self.world_height]),
            dtype=np.float32
        )
        
        # Action space: 24 discrete directions (-180 to +165 degrees in 15-degree increments relative to current heading)
        # 0 = -180°, 1 = -165°, ..., 12 = 0° (no turn), ..., 23 = +165°
        self.action_space = spaces.Discrete(24)
        
        # Render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # For rendering
        self.window = None
        self.clock = None
        
        # Background for continuous space feel
        self.bg_color = (20, 20, 20)
        self.grid_spacing = 40
        self.grid_color = (30, 30, 30)
        
        # Minimap settings
        self.show_minimap = True
        self.minimap_size = 150  # Size of the minimap square
        self.minimap_padding = 10  # Padding from the edge
        
        # Initialize game state
        self.reset()
        
    def _get_obs(self):
        # Return head position, direction angle, and closest food position
        head_x, head_y = self.snake_body[0]
        
        # Find closest food
        closest_food = self.food_positions[0] if self.food_positions else (self.width/2, self.height/2)
        closest_distance = float('inf')
        
        for food_pos in self.food_positions:
            distance = math.sqrt((head_x - food_pos[0])**2 + (head_y - food_pos[1])**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_food = food_pos
        
        return np.array([
            head_x, 
            head_y, 
            self.direction_angle,
            closest_food[0],
            closest_food[1]
        ], dtype=np.float32)
    
    def _get_info(self):
        return {
            'snake_length': len(self.snake_body),
            'score': self.score,
            'direction_angle': self.direction_angle,
            'head_position': self.snake_body[0],
            'num_bots': len(self.bots),
            'num_foods': len(self.food_positions)
        }
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize snake position at the center of the world
        center_x = self.world_width // 2
        center_y = self.world_height // 2
        
        # Start with a single segment (head)
        self.snake_body = [(float(center_x), float(center_y))]
        
        # Initialize direction angle (0 = right, pi/2 = down, pi = left, 3pi/2 = up)
        self.direction_angle = 0.0
        
        # Calculate direction vector
        self.direction_vector = (math.cos(self.direction_angle), math.sin(self.direction_angle))
        
        # Initialize camera to center on snake head
        self.camera_x = center_x - self.screen_size // 2
        self.camera_y = center_y - self.screen_size // 2
        
        # Score and game status
        self.score = 0
        self.game_over = False
        
        # Add initial body segments behind the head
        self._add_initial_segments(3)  # Start with 4 segments total (1 head + 3 body)
        
        # Initialize bot snakes first (needed before placing food)
        self.bots = []
        self.bot_respawn_counter = {}
        for i in range(self.num_bots):
            # Create bots with different colors by using different hues
            hue = i / self.num_bots
            self.bots.append(BotSnake(self.world_width, self.world_height, self.snake_segment_radius, self.np_random, hue))
        
        # Initialize food
        self.food_positions = []
        for _ in range(self.num_foods):
            self._place_food()
        
        # Reset render components
        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Snake.io")
        if self.render_mode == "human" and self.clock is None:
            self.clock = pygame.time.Clock()
            
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _add_initial_segments(self, count):
        # Add initial body segments behind the head
        head_x, head_y = self.snake_body[0]
        angle = self.direction_angle + math.pi  # Opposite direction of head
        
        for i in range(1, count + 1):
            offset = i * self.min_distance_between_segments
            x = head_x + math.cos(angle) * offset
            y = head_y + math.sin(angle) * offset
            self.snake_body.append((float(x), float(y)))
    
    def _place_food(self):
        # Place food at a random position that is not too close to the snake
        min_distance = self.snake_segment_radius * 3
        
        for _ in range(100):  # Try up to 100 times to place food
            food_x = self.np_random.uniform(self.food_radius, self.world_width - self.food_radius)
            food_y = self.np_random.uniform(self.food_radius, self.world_height - self.food_radius)
            food_pos = (food_x, food_y)
            
            # Check if food is far enough from the player snake
            too_close = False
            for segment in self.snake_body:
                dist = math.sqrt((segment[0] - food_x)**2 + (segment[1] - food_y)**2)
                if dist < min_distance:
                    too_close = True
                    break
            
            # Check if food is far enough from all bot snakes
            if not too_close:
                for bot in self.bots:
                    for segment in bot.body:
                        dist = math.sqrt((segment[0] - food_x)**2 + (segment[1] - food_y)**2)
                        if dist < min_distance:
                            too_close = True
                            break
                    if too_close:
                        break
            
            # Check if food is far enough from other food
            if not too_close:
                for other_food in self.food_positions:
                    dist = math.sqrt((other_food[0] - food_x)**2 + (other_food[1] - food_y)**2)
                    if dist < min_distance:
                        too_close = True
                        break
            
            if not too_close:
                self.food_positions.append(food_pos)
                return
        
        # If we can't find a good spot after many tries, just place it randomly
        food_x = self.np_random.uniform(self.food_radius, self.world_width - self.food_radius)
        food_y = self.np_random.uniform(self.food_radius, self.world_height - self.food_radius)
        self.food_positions.append((food_x, food_y))
    
    def step(self, action):
        # Check if game is already over
        if self.game_over:
            observation = self._get_obs()
            info = self._get_info()
            return observation, 0, True, False, info
        
        # Convert action to angle change relative to current direction
        # Action is now relative to current direction (-180 to +165 degrees)
        # 0-11 = turn left (-180 to -15 degrees)
        # 12 = no turn (0 degrees)
        # 13-23 = turn right (+15 to +165 degrees)
        angle_change = (action - 12) * (math.pi / 12.0)  # 15 degree increments
        
        # Limit angle change to max_turn_per_frame
        angle_change = max(-self.max_turn_per_frame, min(self.max_turn_per_frame, angle_change))
        
        # Update direction angle and vector
        self.direction_angle = (self.direction_angle + angle_change) % (2 * math.pi)
        self.direction_vector = (math.cos(self.direction_angle), math.sin(self.direction_angle))
        
        # Move snake head
        head_x, head_y = self.snake_body[0]
        new_head_x = head_x + self.direction_vector[0] * self.snake_speed
        new_head_y = head_y + self.direction_vector[1] * self.snake_speed
        
        # Insert new head position
        self.snake_body[0] = (new_head_x, new_head_y)
        
        # Update camera position to center on snake head
        self.camera_x = new_head_x - self.screen_size // 2
        self.camera_y = new_head_y - self.screen_size // 2
        
        # Move each body segment towards the segment in front of it
        for i in range(len(self.snake_body) - 1, 0, -1):
            target_x, target_y = self.snake_body[i-1]
            current_x, current_y = self.snake_body[i]
            
            # Calculate direction vector to target
            dx = target_x - current_x
            dy = target_y - current_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > self.min_distance_between_segments:
                # Move segment towards the one in front of it
                move_ratio = self.snake_speed / distance
                new_x = current_x + dx * move_ratio
                new_y = current_y + dy * move_ratio
                self.snake_body[i] = (new_x, new_y)
                
        # Initialize reward and terminated status
        reward = 0
        terminated = False
        
        # Check for collisions with walls
        head_x, head_y = self.snake_body[0]
        wall_collision = (
            head_x - self.snake_segment_radius < 0 or 
            head_x + self.snake_segment_radius > self.world_width or 
            head_y - self.snake_segment_radius < 0 or 
            head_y + self.snake_segment_radius > self.world_height
        )
        
        if wall_collision:
            self.game_over = True
            reward = -10  # Penalty for hitting wall
            terminated = True
        
        # Check for collisions with bot snakes
        head_segment = self.snake_body[0]
        bot_collision = False
        killed_bots = []
        
        # Check for head-to-segment collisions (player's head hitting bot segments)
        for bot_idx, bot in enumerate(self.bots):
            # Skip if bot is empty
            if not bot.body:
                continue
                
            bot_head = bot.body[0]
            
            # Check for head-to-head collisions (special case)
            head_to_head_dist = math.sqrt((head_segment[0] - bot_head[0])**2 + (head_segment[1] - bot_head[1])**2)
            
            if head_to_head_dist < self.snake_segment_radius * 1.8:
                # Head-to-head collision
                # If player's snake is longer, player wins; otherwise, bot wins
                if len(self.snake_body) > len(bot.body):
                    # Player kills bot
                    killed_bots.append(bot_idx)
                    reward += 5  # Bonus for killing a bot
                else:
                    # Bot kills player
                    bot_collision = True
                    break
            
            # Check player head against bot body segments (skipping bot head)
            for bot_segment in bot.body[1:]:
                dist = math.sqrt((head_segment[0] - bot_segment[0])**2 + (head_segment[1] - bot_segment[1])**2)
                if dist < self.snake_segment_radius * 1.5:
                    bot_collision = True
                    break
            
            if bot_collision:
                break
        
        # Remove killed bots and add their remnants as food
        for idx in sorted(killed_bots, reverse=True):
            # Convert some of the bot segments to food
            if idx < len(self.bots) and self.bots[idx].body:
                # Add food for every few segments of the killed bot
                for i in range(1, len(self.bots[idx].body), 3):  # Every 3rd segment becomes food
                    segment = self.bots[idx].body[i]
                    if self.food_positions is not None:
                        self.food_positions.append(segment)
                
                # Remove the bot and schedule respawn
                self.bot_respawn_counter[idx] = self.bot_respawn_time
                self.bots.pop(idx)
        
        if bot_collision:
            self.game_over = True
            reward = -10  # Penalty for hitting a bot
            terminated = True
        else:
            terminated = False
            reward = 0
            
        # Check for food collection by player
        head_x, head_y = self.snake_body[0]
        food_eaten = False
        
        for i, (food_x, food_y) in enumerate(self.food_positions[:]):
            food_distance = math.sqrt((head_x - food_x)**2 + (head_y - food_y)**2)
            
            if food_distance < self.snake_segment_radius + self.food_radius:
                self.score += 1
                reward += 10  # Reward for eating food
                food_eaten = True
                
                # Remove the eaten food
                self.food_positions.pop(i)
                
                # Add a new segment at the end of the snake
                last_segment = self.snake_body[-1]
                second_last_segment = self.snake_body[-2]
                
                # Direction from second-last to last segment
                dx = last_segment[0] - second_last_segment[0]
                dy = last_segment[1] - second_last_segment[1]
                
                # Normalize and multiply by segment distance
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    dx /= length
                    dy /= length
                
                # New segment position
                new_segment_x = last_segment[0] + dx * self.min_distance_between_segments
                new_segment_y = last_segment[1] + dy * self.min_distance_between_segments
                
                self.snake_body.append((new_segment_x, new_segment_y))
                break  # Only eat one food per step
        
        # Make sure we maintain the desired number of food items
        while len(self.food_positions) < self.num_foods:
            self._place_food()
            
        # Update bot snakes - but don't end the game if they collide
        bot_food_eaten = []  # Track food eaten by bots
        bots_to_remove = []  # Track bots that need to be removed
        
        for i, bot in enumerate(self.bots):
            result = bot.update(self.food_positions, self.snake_body, [b for b in self.bots if b != bot])
            
            if result:
                # Bot hit a wall or another snake
                if result is True:
                    bots_to_remove.append(i)
                    self.bot_respawn_counter[i] = self.bot_respawn_time
                # Bot ate food
                else:
                    should_die, eaten_food_indices = result
                    if should_die:
                        bots_to_remove.append(i)
                        self.bot_respawn_counter[i] = self.bot_respawn_time
                    
                    # Track food eaten by bots
                    if eaten_food_indices:
                        bot_food_eaten.extend(eaten_food_indices)
        
        # Remove food eaten by bots (after all bots have been updated)
        if bot_food_eaten:
            # Sort in reverse order to avoid index issues when removing
            for food_idx in sorted(set(bot_food_eaten), reverse=True):
                if food_idx < len(self.food_positions):
                    self.food_positions.pop(food_idx)
        
        # Handle dead bots - convert some segments to food and remove them
        for i in sorted(bots_to_remove, reverse=True):
            if i < len(self.bots):
                # Convert some of the bot segments to food
                if self.bots[i].body and len(self.bots[i].body) > 3:
                    # Add food for every few segments of the killed bot
                    for j in range(1, len(self.bots[i].body), 3):  # Every 3rd segment becomes food
                        if j < len(self.bots[i].body):
                            segment = self.bots[i].body[j]
                            self.food_positions.append(segment)
                
                # Remove the bot
                self.bots.pop(i)
        
        # Update respawn counters and respawn bots
        for bot_id in list(self.bot_respawn_counter.keys()):
            self.bot_respawn_counter[bot_id] -= 1
            if self.bot_respawn_counter[bot_id] <= 0:
                hue = bot_id / self.num_bots
                self.bots.append(BotSnake(self.world_width, self.world_height, self.snake_segment_radius, self.np_random, hue))
                del self.bot_respawn_counter[bot_id]
        
        # Make sure we always have the minimum number of food items
        while len(self.food_positions) < self.num_foods:
            self._place_food()
        
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
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.screen_size, self.screen_size))
        canvas.fill(self.bg_color)  # Dark background
        
        # Calculate visible area boundaries in world coordinates
        view_left = self.camera_x
        view_right = self.camera_x + self.screen_size
        view_top = self.camera_y
        view_bottom = self.camera_y + self.screen_size
        
        # Draw grid for continuous space feel - adjusting for camera position
        grid_start_x = (int(view_left) // self.grid_spacing) * self.grid_spacing - int(view_left)
        grid_start_y = (int(view_top) // self.grid_spacing) * self.grid_spacing - int(view_top)
        
        for x in range(grid_start_x, self.screen_size + self.grid_spacing, self.grid_spacing):
            pygame.draw.line(canvas, self.grid_color, (x, 0), (x, self.screen_size))
        for y in range(grid_start_y, self.screen_size + self.grid_spacing, self.grid_spacing):
            pygame.draw.line(canvas, self.grid_color, (0, y), (self.screen_size, y))
        
        # Draw world boundaries
        bound_left = max(0, -view_left)
        bound_right = min(self.screen_size, self.world_width - view_left)
        bound_top = max(0, -view_top)
        bound_bottom = min(self.screen_size, self.world_height - view_top)
        
        # Draw world border (a bit thicker to be visible)
        border_color = (60, 60, 80)  # Bluish-gray
        border_thickness = 3
        
        # Left border
        if view_left <= 0:
            pygame.draw.line(canvas, border_color, (bound_left, 0), (bound_left, self.screen_size), border_thickness)
        
        # Right border
        if view_right >= self.world_width:
            pygame.draw.line(canvas, border_color, (bound_right, 0), (bound_right, self.screen_size), border_thickness)
        
        # Top border
        if view_top <= 0:
            pygame.draw.line(canvas, border_color, (0, bound_top), (self.screen_size, bound_top), border_thickness)
        
        # Bottom border
        if view_bottom >= self.world_height:
            pygame.draw.line(canvas, border_color, (0, bound_bottom), (self.screen_size, bound_bottom), border_thickness)
        
        # Draw bot snakes
        for bot in self.bots:
            # Draw bot body segments
            for i, (x, y) in enumerate(reversed(bot.body)):
                # Convert world coordinates to screen coordinates
                screen_x = int(x - view_left)
                screen_y = int(y - view_top)
                
                # Skip drawing if outside screen
                if (screen_x < -bot.segment_radius or screen_x > self.screen_size + bot.segment_radius or
                    screen_y < -bot.segment_radius or screen_y > self.screen_size + bot.segment_radius):
                    continue
                
                # Gradient color from tail to head
                color_intensity = 0.5 + min(0.5, (i * 0.5) / len(bot.body))
                color = tuple(int(c * color_intensity) for c in bot.base_color)
                
                # Draw circle for segment
                pygame.draw.circle(canvas, color, (screen_x, screen_y), bot.segment_radius)
                
                # Draw eyes on head (first segment in reversed list)
                if i == len(bot.body) - 1:
                    # Eye positions based on direction angle
                    eye_offset = bot.segment_radius * 0.6
                    eye_radius = bot.segment_radius * 0.3
                    
                    # Left eye
                    left_angle = bot.direction_angle - math.pi/4
                    left_eye_x = screen_x + math.cos(left_angle) * eye_offset
                    left_eye_y = screen_y + math.sin(left_angle) * eye_offset
                    
                    # Right eye
                    right_angle = bot.direction_angle + math.pi/4
                    right_eye_x = screen_x + math.cos(right_angle) * eye_offset
                    right_eye_y = screen_y + math.sin(right_angle) * eye_offset
                    
                    # Draw eyes
                    pygame.draw.circle(canvas, (255, 255, 255), (int(left_eye_x), int(left_eye_y)), eye_radius)
                    pygame.draw.circle(canvas, (255, 255, 255), (int(right_eye_x), int(right_eye_y)), eye_radius)
                    
                    # Draw pupils
                    pupil_radius = eye_radius * 0.6
                    pygame.draw.circle(canvas, (0, 0, 0), (int(left_eye_x), int(left_eye_y)), pupil_radius)
                    pygame.draw.circle(canvas, (0, 0, 0), (int(right_eye_x), int(right_eye_y)), pupil_radius)
        
        # Draw player snake body segments
        for i, (x, y) in enumerate(reversed(self.snake_body)):
            # Convert world coordinates to screen coordinates
            screen_x = int(x - view_left)
            screen_y = int(y - view_top)
            
            # Skip drawing if outside screen
            if (screen_x < -self.snake_segment_radius or screen_x > self.screen_size + self.snake_segment_radius or
                screen_y < -self.snake_segment_radius or screen_y > self.screen_size + self.snake_segment_radius):
                continue
            
            # Gradient color from tail to head (darker to brighter green)
            color_intensity = 150 + min(105, (i * 105) // len(self.snake_body))
            color = (0, color_intensity, 0)
            
            # Draw circle for segment
            pygame.draw.circle(canvas, color, (screen_x, screen_y), self.snake_segment_radius)
            
            # Draw eyes on head (first segment)
            if i == len(self.snake_body) - 1:  # This is the head (reversed list)
                # Eye positions based on direction angle
                eye_offset = self.snake_segment_radius * 0.6
                eye_radius = self.snake_segment_radius * 0.3
                
                # Left eye
                left_angle = self.direction_angle - math.pi/4
                left_eye_x = screen_x + math.cos(left_angle) * eye_offset
                left_eye_y = screen_y + math.sin(left_angle) * eye_offset
                
                # Right eye
                right_angle = self.direction_angle + math.pi/4
                right_eye_x = screen_x + math.cos(right_angle) * eye_offset
                right_eye_y = screen_y + math.sin(right_angle) * eye_offset
                
                # Draw eyes
                pygame.draw.circle(canvas, (255, 255, 255), (int(left_eye_x), int(left_eye_y)), eye_radius)
                pygame.draw.circle(canvas, (255, 255, 255), (int(right_eye_x), int(right_eye_y)), eye_radius)
                
                # Draw pupils
                pupil_radius = eye_radius * 0.6
                pygame.draw.circle(canvas, (0, 0, 0), (int(left_eye_x), int(left_eye_y)), pupil_radius)
                pygame.draw.circle(canvas, (0, 0, 0), (int(right_eye_x), int(right_eye_y)), pupil_radius)
        
        # Draw all food in view
        for food_x, food_y in self.food_positions:
            # Convert world coordinates to screen coordinates
            screen_x = int(food_x - view_left)
            screen_y = int(food_y - view_top)
            
            # Skip if outside screen
            if (screen_x < -self.food_radius or screen_x > self.screen_size + self.food_radius or
                screen_y < -self.food_radius or screen_y > self.screen_size + self.food_radius):
                continue
            
            # Draw a glowing effect for food
            for radius in range(int(self.food_radius), 1, -2):
                alpha = 255 - (self.food_radius - radius) * 15
                alpha = max(0, min(255, alpha))
                food_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                color = (255, 0, 0, alpha)
                pygame.draw.circle(food_surface, color, (radius, radius), radius)
                canvas.blit(food_surface, (int(screen_x - radius), int(screen_y - radius)))
            
            # Main food body
            pygame.draw.circle(canvas, (255, 50, 50), (screen_x, screen_y), int(self.food_radius))
        
        # Draw minimap if enabled
        if self.show_minimap:
            minimap_rect = pygame.Rect(
                self.screen_size - self.minimap_size - self.minimap_padding,
                self.minimap_padding,
                self.minimap_size,
                self.minimap_size
            )
            
            # Background for minimap
            pygame.draw.rect(canvas, (0, 0, 0), minimap_rect)
            pygame.draw.rect(canvas, (80, 80, 80), minimap_rect, 1)  # Border
            
            # Scale factors for minimap
            scale_x = self.minimap_size / self.world_width
            scale_y = self.minimap_size / self.world_height
            
            # Draw player position on minimap (green dot)
            head_x, head_y = self.snake_body[0]
            minimap_x = int(minimap_rect.x + head_x * scale_x)
            minimap_y = int(minimap_rect.y + head_y * scale_y)
            pygame.draw.circle(canvas, (0, 255, 0), (minimap_x, minimap_y), 3)
            
            # Draw viewport rectangle on minimap
            viewport_rect = pygame.Rect(
                minimap_rect.x + int(view_left * scale_x),
                minimap_rect.y + int(view_top * scale_y),
                int(self.screen_size * scale_x),
                int(self.screen_size * scale_y)
            )
            pygame.draw.rect(canvas, (255, 255, 255), viewport_rect, 1)
            
            # Draw food positions on minimap (red dots)
            for food_x, food_y in self.food_positions:
                minimap_food_x = int(minimap_rect.x + food_x * scale_x)
                minimap_food_y = int(minimap_rect.y + food_y * scale_y)
                pygame.draw.circle(canvas, (255, 0, 0), (minimap_food_x, minimap_food_y), 1)
            
            # Draw bot positions on minimap (dots in their color)
            for bot in self.bots:
                if bot.body:
                    bot_x, bot_y = bot.body[0]  # Head position
                    minimap_bot_x = int(minimap_rect.x + bot_x * scale_x)
                    minimap_bot_y = int(minimap_rect.y + bot_y * scale_y)
                    pygame.draw.circle(canvas, bot.base_color, (minimap_bot_x, minimap_bot_y), 2)
        
        # Display score and stats
        if pygame.font:
            font = pygame.font.Font(None, 28)
            text = font.render(f"Score: {self.score}", True, (255, 255, 255))
            canvas.blit(text, (10, 10))
            
            # Display length
            length_text = font.render(f"Length: {len(self.snake_body)}", True, (255, 255, 255))
            canvas.blit(length_text, (10, 40))
            
            # Display number of bots
            bots_text = font.render(f"Bots: {len(self.bots)}/{self.num_bots}", True, (255, 255, 255))
            canvas.blit(bots_text, (10, 70))
            
            # Display world position
            head_x, head_y = self.snake_body[0]
            pos_text = font.render(f"Pos: ({int(head_x)}, {int(head_y)})", True, (200, 200, 200))
            canvas.blit(pos_text, (10, 100))
            
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