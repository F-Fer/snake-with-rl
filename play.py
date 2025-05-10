import gymnasium as gym
import pygame
import snake_env
import time

def play_snake():
    # Create environment - using a custom slower render fps
    env = gym.make('Snake-v0', render_mode='human')
    
    # Modify the render FPS to slow down the game
    env.metadata['render_fps'] = 5  # Reduced from 15 to 5 fps
    
    observation, info = env.reset()
    
    # Initialize pygame for capturing key events
    pygame.init()
    
    # Game loop
    running = True
    action = 1  # Start moving right
    
    print("Welcome to Snake! Use arrow keys to move.")
    print("Controls: ↑(up), →(right), ↓(down), ←(left), Q(quit)")
    
    # Add a slight delay before starting game
    time.sleep(1)
    
    # Add a variable to control manual step timing
    last_step_time = time.time()
    step_delay = 0.2  # 200ms between steps (additional delay)
    
    while running:
        current_time = time.time()
        
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3
                elif event.key == pygame.K_q:
                    running = False
        
        # Only take a step if enough time has passed (additional timing control)
        if current_time - last_step_time >= step_delay:
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            last_step_time = current_time
            
            # Display score in console
            print(f"\rScore: {info['score']}", end="")
            
            # Check if game is over
            if terminated or truncated:
                print(f"\nGame Over! Final Score: {info['score']}")
                time.sleep(2)  # Pause briefly to see final state
                observation, info = env.reset()
                action = 1  # Reset to moving right
    
    # Clean up
    env.close()
    pygame.quit()

if __name__ == "__main__":
    play_snake()