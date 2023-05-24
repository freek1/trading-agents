from matplotlib import pyplot as plt
import pygame
import random
import numpy as np
import seaborn as sns
import copy
from lifelines import KaplanMeierFitter

# Functions file
from funcs import *
# Agent class
from agent import Agent

# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()
dt = 0
fps = 120
time = 1

GRID_WIDTH, GRID_HEIGHT, CELL_SIZE = get_grid_params()

# Set the dimensions of the screen
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (153, 102, 0)
DARK_BROWN = (60, 40, 0)
LIGHT_BROWN = (148, 113, 0)
GREEN = (0, 153, 51)
DARK_GREEN = (0, 102, 34)
LIGHT_GREEN = (102, 204, 102)
RED  = (255, 25, 25)
BLUE = (25, 25, 255)
FOOD_COLOR = (200,100,0)

MAX_WOOD = 10
MAX_FOOD = 10
# Uniform random distribution of resources
# wood = [[random.uniform(0, MAX_WOOD) for x in range(4,12)] for y in range(4,12)]
# food = [[random.uniform(0, MAX_FOOD) for x in range(4,12)] for y in range(4,12)]

# Resources in non-random positions
wood = np.zeros((GRID_HEIGHT, GRID_WIDTH))
for i in range(0,GRID_WIDTH):
    for j in range(0,8):
        wood[i][j] = random.uniform(5, MAX_WOOD)

food = np.zeros((GRID_HEIGHT, GRID_WIDTH))
for i in range(0,GRID_WIDTH):
    for j in range(32,GRID_HEIGHT):
        food[i][j] = random.uniform(5, MAX_FOOD)

maximum_resources = {
    'wood': wood,
    'food': food,
}

resources = copy.deepcopy(maximum_resources)

# Set up the agents
NUM_AGENTS = 100
agents = []
agent_colours = sns.color_palette('bright', n_colors=NUM_AGENTS)

regen_amount = 10
regen_active = True

transaction_cost = 0.1
gather_amount = 1

agent_positions = np.zeros([NUM_AGENTS, 2])

# Creating agents
for i in range(NUM_AGENTS):
    x = random.randint(0, GRID_WIDTH-2)
    y = random.randint(0, GRID_HEIGHT-2)
    color = (255.0,0.0,0.0) if y < GRID_HEIGHT/2 else (0.0,255.0,0.0)
    agent = Agent(i, x, y, color, GRID_WIDTH, GRID_HEIGHT) #color = np.array(agent_colours[i])*255
    agents.append(agent)

    # Save agent position for the KD-tree
    agent_positions[i] = [x, y]
    # Initialize KDTree
    positions_tree = KDTree(agent_positions)

# Run the simulation
running = True

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Clear the screen
    screen.fill(WHITE)

    # Draw the grid
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            wood_value = wood[row][col]
            food_value = food[row][col]
            # Map the resource value to a shade of brown or green
            
            wood_color = DARK_BROWN if wood_value > 7.5 else LIGHT_BROWN if wood_value > 5 else BROWN if wood_value > 1 else WHITE
            food_color = DARK_GREEN if food_value > 7.5 else LIGHT_GREEN if food_value > 5 else GREEN if food_value > 1 else WHITE

            blended_color = (
                max(min(255, int((wood_color[0] * wood_value + food_color[0] * food_value) / (wood_value + food_value + 1))), 0),
                max(min(255, int((wood_color[1] * wood_value + food_color[1] * food_value) / (wood_value + food_value + 1))), 0),
                max(min(255, int((wood_color[2] * wood_value + food_color[2] * food_value) / (wood_value + food_value + 1))), 0),
                max(min(255, int(max(wood_value, food_value)*25)), 0)
            )
            '''
            #food_color = FOOD_COLOR * (food_value/MAX_FOOD)
            food_color = tuple((food_value/MAX_FOOD) * elem for elem in FOOD_COLOR)
            #wood_color = DARK_BROWN * (wood_value/MAX_WOOD)
            wood_color = tuple((wood_value/MAX_WOOD) * elem for elem in DARK_BROWN)
            blended_color = tuple(map(lambda x, y: (x + y)/2, food_color, wood_color))'''

            rect = pygame.Rect(row * CELL_SIZE, col * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            draw_rect_alpha(screen, blended_color, rect)

    # Draw the grid
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (SCREEN_WIDTH, y))

    # DEBUG
    # if len(agents) < 10:
    #     fps = 5
    #     for agent in agents:
    #         print('pos',agent.getPos(),'wood',agent.getCurrentStock('wood'),'food',agent.getCurrentStock('food'))

    # Counting the nr of alive agents for automatic stopping
    nr_agents = 0

    # Update the agents
    for agent in agents:
        if agent.isAlive():
            nr_agents += 1
            agent.update_time_alive()

            x, y = agent.getPos()
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, agent.getColor(), rect)
            
            # Do agent behaviour
            if agent.getBehaviour() == 'trade_wood' or agent.getBehaviour() == 'trade_food':
                traded = False
                neighboring_cells = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
                neighboring_cells.remove((0,0))
                while not traded and neighboring_cells:
                    dx, dy = random.choice(neighboring_cells)
                    neighboring_cells.remove((dx, dy))
                    if 0 <= x+dx < GRID_WIDTH and 0 <= y+dy < GRID_HEIGHT:
                        x_check = agent.getPos()[0] + dx
                        y_check = agent.getPos()[1] + dy
                        occupied, agent_B = cellAvailable(x_check, y_check, agents)
                        if agent_B is None:
                            continue
                        if agent.compatible(agent_B):
                            print(f"TRADE at {agent.getPos()} at pos={agent_B.getPos()}")
                            print(f"  Agent A = {agent.current_stock}, {agent.behaviour}")
                            print(f"  Agent B = {agent_B.current_stock}, {agent_B.behaviour}")
                            traded_qty = agent.trade(agent_B, transaction_cost)
                            traded = True
                            print(f"  Qty traded: {traded_qty}")
                            agent.clearBlacklistedAgents()
                        else:
                            # If not compatible, find next nearest neighbor
                            agent.removeClosestNeighbor()

            # Update the resource gathering
            else:
                chosen_resource = choose_resource(agent, resources, gather_amount) # make agent choose which resource to gather based on it's predisposition
                if able_to_take_resource(agent, chosen_resource, resources):
                    take_resource(agent, chosen_resource, resources, gather_amount)
            
            # Upkeep of agents and check if agent can survive
            agent.upkeep()

            # Choose behaviour
            agent.updateBehaviour() # Agent brain

            # Choose step
            preferred_direction = agent.chooseStep()
            moveAgent(preferred_direction, agent, agents)

            # Distance and indices of 5 nearest neighbors
            dist, idx = positions_tree.query([[x, y]], k=5)
            # Coordinates of 5 nearest neighbors as param
            agent.setNearestNeighbors(agent_positions[idx][0])

            # Update agent position for the KD-tree
            agent_positions[i] = [x, y]
        
        # Updating KD-tree
        positions_tree = KDTree(agent_positions)

    if regen_active:
        for maximum_resource in maximum_resources:
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if maximum_resources[maximum_resource][x][y] > resources[maximum_resource][x][y]:
                        resources[f'{maximum_resource}'][x][y] += regen_amount
                        
    # Update the display
    pygame.display.flip()

    clock.tick(fps)
    dt = clock.tick(fps)/100
    time += 1
    
    if nr_agents == 0:
        print('No agents left, ending simulation')
        running=False
        
# Clean up
pygame.quit()

# Time alive of agents distribution
alive_times = np.zeros(NUM_AGENTS)
for agent in agents:
    alive_times[agent.id] = agent.time_alive

# List of time-steps
duration = np.arange(time)
# List of when agents died
events = np.zeros(len(duration))
for ev in alive_times:
    # If agent died before the final timestep (otherwise it was still alive at the end)
    if ev < time - 1:
        ev = int(ev)
        events[ev] = 1

# Result figures
kmf = KaplanMeierFitter()
kmf.fit(duration, events)
km_graph = plt.figure()
kmf.plot()
plt.title('Kaplan-Meier curve of agent deaths')
plt.ylabel('Survival probability')

time_alive_fig = plt.figure()
plt.bar(np.arange(NUM_AGENTS), np.sort(alive_times))
plt.plot(np.arange(NUM_AGENTS), np.sort(alive_times), 'k')
plt.xlabel('Agents')
plt.ylabel('Time alive [timesteps]')
plt.title('Time alive distribution of the agents')

# Keep images open
plt.show()
