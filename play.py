import gymnasium as gym
import pygame
import snake_env
import time
import math
import numpy as np

def play_snake():
    # Create environment. render_mode is now fixed in the env.
    # screen_size for observation is 84x84. world_size can still be large.
    env = gym.make('Snake-v0', num_bots=3, num_foods=20, world_size=3000, screen_size=256, zoom_level=1.0)

    pygame.init()
    display_width = 600  # Width of the window to show the game
    display_height = 600 # Height of the window to show the game
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Snake RL Environment Viewer")
    clock = pygame.time.Clock()
    # --- End Pygame setup ---

    observation, info = env.reset() # Observation is now an 84x84x3 image

    # Game loop
    running = True
    
    # For controlling the snake with keyboard (continuous actions)
    # Target angle for the snake, controlled by arrow keys
    target_direction_angle = 0.0 # Initial angle (right)
    angle_increment = math.pi / 18 # Increment for turning (10 degrees)

    print("Welcome to Snake.io RL Viewer!")
    print("Use LEFT/RIGHT arrow keys to change snake direction.")
    print("Press Q or ESC to quit.")

    # Add a slight delay before starting game
    time.sleep(1)

    last_step_time = time.time()
    step_delay = 0.05 # Control game speed for playability

    while running:
        current_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_LEFT:
                    target_direction_angle -= angle_increment
                if event.key == pygame.K_RIGHT:
                    target_direction_angle += angle_increment
        
        target_direction_angle %= (2 * math.pi) # Normalize angle

        # Convert target_direction_angle to [cos, sin] action
        action = np.array([math.cos(target_direction_angle), math.sin(target_direction_angle)], dtype=np.float32)

        if current_time - last_step_time >= step_delay:
            observation, reward, terminated, truncated, info = env.step(action)
            last_step_time = current_time

            # Display score and stats in console
            current_snake_angle_rad = info['direction_angle']
            heading_degrees = int(current_snake_angle_rad * 180 / math.pi) % 360
            
            # Calculate target degrees for display
            target_degrees = int(target_direction_angle * 180 / math.pi) % 360

            print(f"\rScore: {info['score']} | Length: {info['snake_length']} | " +
                  f"Bots: {info['num_bots']} | Reward: {reward:.2f} | " +
                  f"Target Dir: {target_degrees}° | Actual Heading: {heading_degrees}°",
                  end="")

            if terminated or truncated:
                print(f"\nGame Over! Final Score: {info['score']}")
                time.sleep(2)
                observation, info = env.reset()
                target_direction_angle = 0.0 # Reset target angle

        # --- Display the 84x84 observation from the environment ---
        if observation is not None:
            # Pygame expects (width, height, channel) or (width, height)
            # The observation is (height, width, channel) from env.render()
            # So it's already in a good format for pygame.surfarray.make_surface
            # but pygame.transform.scale expects a surface.
            
            # Create a Pygame surface from the observation
            obs_surface = pygame.surfarray.make_surface(observation) # observation is HxWxC
            
            # Scale the 84x84 surface to fit the display window
            scaled_surface = pygame.transform.scale(obs_surface, (display_width, display_height))
            
            screen.blit(scaled_surface, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for the display window

    env.close()
    pygame.quit()

if __name__ == "__main__":
    play_snake()