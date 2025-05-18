import gymnasium as gym
import pygame
import snake_env
import time
import math
import numpy as np

def play_snake():
    # Create environment. render_mode is now fixed in the env.
    # screen_size for observation is defined in env. world_size can still be large.
    env = gym.make('Snake-v0', num_bots=3, num_foods=20, world_size=3000, screen_size=256, zoom_level=1.0)

    # --- Pygame setup for displaying the observation ---
    pygame.init()
    display_width = 600  # Width of the window to show the game
    display_height = 600 # Height of the window to show the game
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Snake RL Environment Viewer")
    clock = pygame.time.Clock()
    # Center of the display window (for mouse control)
    display_center_x = display_width // 2
    display_center_y = display_height // 2
    # --- End Pygame setup ---

    observation, info = env.reset() # Observation is an image

    # Game loop
    running = True
    
    target_direction_angle = 0.0 # Initial angle, will be updated by mouse

    print("Welcome to Snake.io RL Viewer!")
    print("Use MOUSE to control snake direction.")
    print("Press Q or ESC to quit.")


    frame_count = 0
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Calculate angle from display center to mouse position
        dx = mouse_x - display_center_x
        dy = mouse_y - display_center_y
        target_direction_angle = math.atan2(dx, dy)

        target_direction_angle %= (2 * math.pi) # Normalize angle

        # Convert target_direction_angle to [cos, sin] action
        action = np.array([math.cos(target_direction_angle), math.sin(target_direction_angle)], dtype=np.float32)

        observation, reward, terminated, truncated, info = env.step(action)

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
            print(f"Frame count: {frame_count}")
            print(f"terminated: {terminated}, truncated: {truncated}")
            time.sleep(2)
            observation, info = env.reset()
            # target_direction_angle will be updated by mouse on next frame
            

        # --- Display the observation from the environment ---
        if observation is not None:
            obs_surface = pygame.surfarray.make_surface(observation) 
            scaled_surface = pygame.transform.scale(obs_surface, (display_width, display_height))
            screen.blit(scaled_surface, (0, 0))
        
        pygame.display.flip()
        clock.tick(60) # Limit FPS for the display window for smoother mouse input processing
        frame_count += 1

    env.close()
    pygame.quit()

if __name__ == "__main__":
    play_snake()