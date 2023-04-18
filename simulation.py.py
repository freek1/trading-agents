import pygame
import random

# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()
dt = 0


# Set the dimensions of the screen
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up the grid
CELL_SIZE = 8
GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

grid = [[0 for x in range(GRID_WIDTH)] for y in range(GRID_HEIGHT)]

# Set up the agents
NUM_AGENTS = 10
agents = []

for i in range(NUM_AGENTS):
    agent = {
        "x": GRID_WIDTH // 2,
        "y": GRID_HEIGHT // 2,
        "color": (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)),
    }
    agents.append(agent)

# Run the simulation
running = True

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the agents
    for agent in agents:
        # Move the agent randomly
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        agent["x"] += dx
        agent["y"] += dy

        # Keep the agent on the grid
        agent["x"] = max(0, min(GRID_WIDTH - 1, agent["x"]))
        agent["y"] = max(0, min(GRID_HEIGHT - 1, agent["y"]))

    # Clear the screen
    screen.fill(BLACK)

    # Draw the grid
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            if grid[y][x] == 1:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, WHITE, rect)

    # Draw the agents
    for agent in agents:
        rect = pygame.Rect(agent["x"] * CELL_SIZE, agent["y"] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, agent["color"], rect)

    # Update the display
    pygame.display.flip()

    clock.tick(10)  # limits FPS to 60
    dt = clock.tick(60)/1000

# Clean up
pygame.quit()
