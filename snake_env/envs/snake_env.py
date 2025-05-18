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
    metadata = {'render_modes': ['rgb_array']}

    # Reward constants
    C1 = 0.1  # For change in length
    C2 = 10   # For eliminating an opponent
    C3 = 100  # For death
    
    def __init__(self, world_size=3000, snake_segment_radius=10, num_bots=3, num_foods=10, screen_size=84, zoom_level=1.0):
        self.screen_size = screen_size  
        self.render_mode = "rgb_array" 
        self.zoom_level = zoom_level 
        
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
        
        # Turning agility parameters
        self.initial_snake_length = 4  # 1 head + 3 initial segments
        self.base_max_turn_per_frame = math.pi / 24 
        self.turn_agility_decay_factor = 0.05 # Determines how fast agility drops with length
        self.min_allowable_turn_per_frame = math.pi / 36 # Min turn of 5 degrees, regardless of length
        
        # Bot settings
        self.num_bots = num_bots
        self.num_foods = num_foods
        self.bot_respawn_time = 300  # Number of steps before respawning a bot
        self.bot_respawn_counter = {}  # Maps bot_id to respawn countdown
        
        # Observation space: screen_size x screen_size x 3 image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(screen_size, screen_size, 3),
            dtype=np.uint8
        )
        
        # Action space: two continuous values (sine and cosine) for direction
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Background for continuous space feel
        self.bg_color = (20, 20, 20)
        self.grid_spacing = 40
        self.grid_color = (30, 30, 30)

        # Calculate the effective span of the world that the camera will see for the intermediate render surface
        self.effective_view_span = int(self.screen_size * self.zoom_level)
        
        # Initialize game state
        self.reset()

    
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
        self.camera_x = center_x - self.effective_view_span // 2
        self.camera_y = center_y - self.effective_view_span // 2
        
        # Score and game status
        self.score = 0
        self.game_over = False
        self.previous_length = 3 # Initial length
        
        # Add initial body segments behind the head
        self._add_initial_segments(3)
        
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
        
        observation = self._render_frame()
        info = self._get_info()
        
        return observation, info
    
    
    def _add_initial_segments(self, count):
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
            observation = self._render_frame()
            info = self._get_info()
            return observation, 0, True, False, info
        
        # Action is [cos_val, sin_val]
        cos_val = action[0]
        sin_val = action[1]

        # Directly use atan2 to get the target angle
        target_angle = math.atan2(sin_val, cos_val)

        # Smoothly turn towards the target_angle
        current_angle_rad = self.direction_angle % (2 * math.pi)
        target_angle_rad = target_angle % (2 * math.pi)

        angle_diff = target_angle_rad - current_angle_rad
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Calculate dynamic max turn per frame based on snake length
        current_length = len(self.snake_body)
        length_effect = max(0, current_length - self.initial_snake_length)
        dynamic_max_turn = self.base_max_turn_per_frame / (1 + self.turn_agility_decay_factor * length_effect)
        effective_max_turn_per_frame = max(self.min_allowable_turn_per_frame, dynamic_max_turn)
        
        # Apply max turn limit
        turn_this_frame = max(-effective_max_turn_per_frame, min(effective_max_turn_per_frame, angle_diff))
        
        self.direction_angle = (self.direction_angle + turn_this_frame) % (2 * math.pi)
        self.direction_vector = (math.cos(self.direction_angle), math.sin(self.direction_angle))
        
        # Move snake head
        head_x, head_y = self.snake_body[0]
        new_head_x = head_x + self.direction_vector[0] * self.snake_speed
        new_head_y = head_y + self.direction_vector[1] * self.snake_speed
        
        # Update head position
        self.snake_body[0] = (new_head_x, new_head_y)
        
        # Update camera position to center on snake head
        self.camera_x = new_head_x - self.effective_view_span // 2
        self.camera_y = new_head_y - self.effective_view_span // 2
        
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
        current_length = len(self.snake_body)

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
            reward -= self.C3 
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
                    reward += self.C2 * len(bot.body)  # Reward for eliminating opponent
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
            reward -= self.C3 
            terminated = True
        else:
            terminated = False
            reward = 0
            
        # Calculate length change reward (C1)
        length_change = current_length - self.previous_length
        reward += self.C1 * length_change
        self.previous_length = current_length
            
        # Check for food collection by player
        head_x, head_y = self.snake_body[0]
        food_eaten = False
        
        for i, (food_x, food_y) in enumerate(self.food_positions[:]):
            food_distance = math.sqrt((head_x - food_x)**2 + (head_y - food_y)**2)
            
            if food_distance < self.snake_segment_radius + self.food_radius:
                self.score += 1
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
            
        # Update bot snakes 
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

        observation = self._render_frame()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    

    def _render_frame(self):
        # Determine dimensions of the intermediate rendering surface based on zoom
        render_surface_dim = self.effective_view_span
        temp_canvas = pygame.Surface((render_surface_dim, render_surface_dim))
        temp_canvas.fill(self.bg_color)  # Dark background
        
        # Calculate visible area boundaries in world coordinates for the temp_canvas
        view_left = self.camera_x # World coordinate of the left edge of the zoomed view
        view_top = self.camera_y  # World coordinate of the top edge of the zoomed view
        
        # Draw grid for continuous space feel - adjusting for camera position on temp_canvas
        # grid_start_x is the pixel offset on temp_canvas for the first vertical grid line
        grid_start_x_on_temp_canvas = (int(view_left) // self.grid_spacing) * self.grid_spacing - int(view_left)
        grid_start_y_on_temp_canvas = (int(view_top) // self.grid_spacing) * self.grid_spacing - int(view_top)
        
        for x_on_temp_canvas in range(grid_start_x_on_temp_canvas, render_surface_dim + self.grid_spacing, self.grid_spacing):
            pygame.draw.line(temp_canvas, self.grid_color, (x_on_temp_canvas, 0), (x_on_temp_canvas, render_surface_dim))
        for y_on_temp_canvas in range(grid_start_y_on_temp_canvas, render_surface_dim + self.grid_spacing, self.grid_spacing):
            pygame.draw.line(temp_canvas, self.grid_color, (0, y_on_temp_canvas), (render_surface_dim, y_on_temp_canvas))
        
        # Draw world boundaries on temp_canvas
        # These are pixel coordinates on the temp_canvas
        bound_left_on_temp = max(0, -view_left)
        bound_right_on_temp = min(render_surface_dim, self.world_width - view_left)
        bound_top_on_temp = max(0, -view_top)
        bound_bottom_on_temp = min(render_surface_dim, self.world_height - view_top)
        
        border_color = (60, 60, 80)  # Bluish-gray
        border_thickness = 3 # This will be 3 pixels on temp_canvas, then scaled down
        
        # Left border
        if view_left <= 0:
            pygame.draw.line(temp_canvas, border_color, (bound_left_on_temp, 0), (bound_left_on_temp, render_surface_dim), border_thickness)
        
        # Right border
        # Condition for drawing right border: right edge of view (view_left + render_surface_dim) >= world_width
        if view_left + render_surface_dim >= self.world_width:
            pygame.draw.line(temp_canvas, border_color, (bound_right_on_temp, 0), (bound_right_on_temp, render_surface_dim), border_thickness)
        
        # Top border
        if view_top <= 0:
            pygame.draw.line(temp_canvas, border_color, (0, bound_top_on_temp), (render_surface_dim, bound_top_on_temp), border_thickness)
        
        # Bottom border
        # Condition for drawing bottom border: bottom edge of view (view_top + render_surface_dim) >= world_height
        if view_top + render_surface_dim >= self.world_height:
            pygame.draw.line(temp_canvas, border_color, (0, bound_bottom_on_temp), (render_surface_dim, bound_bottom_on_temp), border_thickness)
        
        # Draw bot snakes on temp_canvas
        for bot in self.bots:
            # Draw bot body segments
            for i, (x, y) in enumerate(reversed(bot.body)):
                # Convert world coordinates to temp_canvas coordinates
                screen_x = int(x - view_left)
                screen_y = int(y - view_top)
                
                # Skip drawing if outside temp_canvas
                if (screen_x < -bot.segment_radius or screen_x > render_surface_dim + bot.segment_radius or
                    screen_y < -bot.segment_radius or screen_y > render_surface_dim + bot.segment_radius):
                    continue
                
                # Gradient color from tail to head
                color_intensity = 0.5 + min(0.5, (i * 0.5) / len(bot.body))
                color = tuple(int(c * color_intensity) for c in bot.base_color)
                
                # Draw circle for segment on temp_canvas (radius is world unit, drawn as pixels)
                pygame.draw.circle(temp_canvas, color, (screen_x, screen_y), bot.segment_radius)
                
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
                    pygame.draw.circle(temp_canvas, (255, 255, 255), (int(left_eye_x), int(left_eye_y)), eye_radius)
                    pygame.draw.circle(temp_canvas, (255, 255, 255), (int(right_eye_x), int(right_eye_y)), eye_radius)
                    
                    # Draw pupils
                    pupil_radius = eye_radius * 0.6
                    pygame.draw.circle(temp_canvas, (0, 0, 0), (int(left_eye_x), int(left_eye_y)), pupil_radius)
                    pygame.draw.circle(temp_canvas, (0, 0, 0), (int(right_eye_x), int(right_eye_y)), pupil_radius)
        
        # Draw player snake body segments on temp_canvas
        for i, (x, y) in enumerate(reversed(self.snake_body)):
            # Convert world coordinates to temp_canvas coordinates
            screen_x = int(x - view_left)
            screen_y = int(y - view_top)
            
            # Skip drawing if outside temp_canvas
            if (screen_x < -self.snake_segment_radius or screen_x > render_surface_dim + self.snake_segment_radius or
                screen_y < -self.snake_segment_radius or screen_y > render_surface_dim + self.snake_segment_radius):
                continue
            
            # Gradient color from tail to head (darker to brighter green)
            color_intensity = 150 + min(105, (i * 105) // len(self.snake_body))
            color = (0, color_intensity, 0)
            
            # Draw circle for segment on temp_canvas
            pygame.draw.circle(temp_canvas, color, (screen_x, screen_y), self.snake_segment_radius)
            
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
                pygame.draw.circle(temp_canvas, (255, 255, 255), (int(left_eye_x), int(left_eye_y)), eye_radius)
                pygame.draw.circle(temp_canvas, (255, 255, 255), (int(right_eye_x), int(right_eye_y)), eye_radius)
                
                # Draw pupils
                pupil_radius = eye_radius * 0.6
                pygame.draw.circle(temp_canvas, (0, 0, 0), (int(left_eye_x), int(left_eye_y)), pupil_radius)
                pygame.draw.circle(temp_canvas, (0, 0, 0), (int(right_eye_x), int(right_eye_y)), pupil_radius)
        
        # Draw all food in view on temp_canvas
        for food_x, food_y in self.food_positions:
            # Convert world coordinates to temp_canvas coordinates
            screen_x = int(food_x - view_left)
            screen_y = int(food_y - view_top)
            
            # Skip if outside temp_canvas
            if (screen_x < -self.food_radius or screen_x > render_surface_dim + self.food_radius or
                screen_y < -self.food_radius or screen_y > render_surface_dim + self.food_radius):
                continue
            
            # Draw a glowing effect for food on temp_canvas
            for radius in range(int(self.food_radius), 1, -2):
                alpha = 255 - (self.food_radius - radius) * 15
                alpha = max(0, min(255, alpha))
                food_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                color = (255, 0, 0, alpha)
                pygame.draw.circle(food_surface, color, (radius, radius), radius)
                temp_canvas.blit(food_surface, (int(screen_x - radius), int(screen_y - radius)))
            
            # Main food body on temp_canvas
            pygame.draw.circle(temp_canvas, (255, 50, 50), (screen_x, screen_y), int(self.food_radius))

        # Scale the temp_canvas down to the final observation size (self.screen_size x self.screen_size, e.g., 84x84)
        final_observation_canvas = pygame.transform.scale(temp_canvas, (self.screen_size, self.screen_size))

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(final_observation_canvas)), axes=(1, 0, 2) # Transpose for (height, width, channel)
        )

    def close(self):
        pygame.quit() 