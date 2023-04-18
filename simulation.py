import pygame
import random

def draw_rect_alpha(surface, color, rect):
    shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
    pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
    surface.blit(shape_surf, rect)


# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()
dt = 0
fps = 200

# Set the dimensions of the screen
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (153, 102, 0)
DARK_BROWN = (90, 70, 0)
LIGHT_BROWN = (148, 113, 0)
GREEN = (0, 153, 51)
DARK_GREEN = (0, 102, 34)
LIGHT_GREEN = (102, 204, 102)
RED  = (255, 25, 25)
BLUE = (25, 25, 255)


# Set up the grid
CELL_SIZE = 80
GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE

wood = [[random.uniform(0, 1) for x in range(GRID_WIDTH)] for y in range(GRID_HEIGHT)]
food = [[random.uniform(0, 1) for x in range(GRID_WIDTH)] for y in range(GRID_HEIGHT)]

resources = {
    'wood': wood,
    'food': food,
}

# Set up the agents
NUM_AGENTS = 2
agents = []
agent_colours = [RED, BLUE]

for i in range(NUM_AGENTS):
    agent = {
        "x": GRID_WIDTH // 2,
        "y": GRID_HEIGHT // 2,
        "id": i,
        "color": agent_colours[i],
        "wood_capacity":1,
        "food_capacity":1,
        "current_stock": {
            "wood": 0, 
            "food": 0,
        }
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


    
    # Update the resources
    for agent in agents:
        for resource in resources:
            if agent[f'{resource}_capacity'] > agent['current_stock'][f'{resource}']+0.1 and resources[resource][agent['y']][agent['x']] >= 0.1:
                agent['current_stock'][f'{resource}'] += 0.1
                resources[resource][agent['y']][agent['x']] -= 0.1

        rect = pygame.Rect(agent["x"] * CELL_SIZE, agent["y"] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, agent["color"], rect)


    # Clear the screen
    screen.fill(WHITE)

    # Draw the grid
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            wood_value = wood[row][col]
            food_value = food[row][col]
            # Map the resource value to a shade of brown or green
            wood_color = DARK_BROWN if wood_value > 0.75 else LIGHT_BROWN if wood_value > 0.5 else BROWN if wood_value > 0.1 else WHITE
            food_color = DARK_GREEN if food_value > 0.75 else LIGHT_GREEN if food_value > 0.5 else GREEN if food_value > 0.1 else WHITE

            blended_color = (
                int((wood_color[0] * wood_value + food_color[0] * food_value) / (wood_value + food_value)),
                int((wood_color[1] * wood_value + food_color[1] * food_value) / (wood_value + food_value)),
                int((wood_color[2] * wood_value + food_color[2] * food_value) / (wood_value + food_value)),
                int(max(wood_value, food_value)*255)
            ) 
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            draw_rect_alpha(screen, blended_color, rect)

    # Draw the grid
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (SCREEN_WIDTH, y))


    # Draw the agents
    for agent in agents:
        rect = pygame.Rect(agent["x"] * CELL_SIZE, agent["y"] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, agent["color"], rect)

    # Update the display
    pygame.display.flip()

    clock.tick(fps)  # limits FPS to 60
    dt = clock.tick(fps)/1000

# Clean up
pygame.quit()


