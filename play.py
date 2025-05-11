import gymnasium as gym
import pygame
import snake_env
import time
import math

def play_snake():
    # Create environment with multiple bot snakes and multiple food items
    # Using a larger world size (3000x3000 pixels)
    env = gym.make('Snake-v0', render_mode='human', num_bots=5, num_foods=20, world_size=3000)
    
    # Get the window dimensions from PyGame once it's been initialized
    observation, info = env.reset()
    
    # Modify the render FPS for smoother gameplay
    env.metadata['render_fps'] = 60  # Higher FPS for smoother mouse control
    
    # Initialize pygame for capturing mouse events
    pygame.init()
    
    # Get the actual window dimensions after the window is created
    # Using pygame.display.get_surface().get_size() to get actual screen dimensions
    pygame.event.pump()  # Process events to make sure the window is created
    screen_width, screen_height = pygame.display.get_surface().get_size()
    
    # Center of screen to calculate screen coordinate of head
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2
    
    # Game loop
    running = True
    last_action = 12  # Start with no turn (action 12 = 0 degrees relative to current heading)
    
    print("Welcome to Snake.io! Use your mouse to control direction.")
    print("Move mouse to change direction (limited to 60° turn per frame).")
    print("Unlike classic Snake, you can run over yourself like in snake.io!")
    print("Avoid hitting other snakes and the walls.")
    print("Press Q or ESC to quit.")
    
    # Add a slight delay before starting game
    time.sleep(1)
    
    # Add a variable to control manual step timing
    last_step_time = time.time()
    step_delay = 0.01  # 10ms between steps (for smooth movement)
    
    # Hide the mouse cursor
    pygame.mouse.set_visible(False)
    
    while running:
        current_time = time.time()
        
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get mouse position
        mouse_pos = pygame.mouse.get_pos()
        
        # Get head world position from info
        head_world_x, head_world_y = info['head_position']
        
        # Calculate target angle from mouse position relative to screen center (where head is)
        dx = mouse_pos[0] - screen_center_x
        dy = mouse_pos[1] - screen_center_y
        target_angle = math.atan2(dy, dx)  # angle in radians
        
        # Normalize both angles to [0, 2π)
        current_angle = info['direction_angle'] % (2 * math.pi)
        target_angle = target_angle % (2 * math.pi)
        
        # Calculate the shortest angle difference
        angle_diff = target_angle - current_angle
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # Convert angle difference to action (relative turn)
        # 0 = -180°, 12 = 0° (no turn), 23 = +165°
        relative_action = int(round(angle_diff / (math.pi / 12))) + 12
        relative_action = max(0, min(23, relative_action))  # Clamp to valid range
        
        # Only take a step if enough time has passed
        if current_time - last_step_time >= step_delay:
            # Step environment
            observation, reward, terminated, truncated, info = env.step(relative_action)
            last_step_time = current_time
            last_action = relative_action
            
            # Display score and stats in console
            heading_degrees = int(info['direction_angle'] * 180 / math.pi) % 360
            relative_degrees = (relative_action - 12) * 15
            print(f"\rScore: {info['score']} | Length: {info['snake_length']} | " +
                  f"Bots: {info['num_bots']} | Turn: {relative_degrees}° | Heading: {heading_degrees}°", 
                  end="")
            
            # Check if game is over
            if terminated or truncated:
                print(f"\nGame Over! Final Score: {info['score']}")
                time.sleep(2)  # Pause briefly to see final state
                observation, info = env.reset()
                last_action = 12  # Reset to no turn
        
    # Clean up
    pygame.mouse.set_visible(True)
    env.close()
    pygame.quit()

if __name__ == "__main__":
    play_snake()