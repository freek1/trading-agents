import pygame
import random
import numpy as np
import seaborn as sns
import copy
import pandas as pd
import os

# Functions file
from funcs import *
# Agent class
from agent import Agent

# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()
dt = 0
fps = 144
time = 1

SAVE_TO_FILE = True

RUN_NR = 2

MOVE_PROB = 0.8

# Market, Baseline, 
SCENARIO = 'Baseline'
# 'random', 'pathfind_neighbor', 'pathfind_market'
AGENT_TYPE = 'random'
# trading switch
TRADING = True

# Resource distribution parameters
DISTRIBUTION = 'RandomGrid' # Sides, RandomGrid, Uniform
regen_active = True
MAX_WOOD = 2
MAX_FOOD = 2
#MIN_WOOD = 0  # Not used anymore
#MIN_FOOD = 0

GRID_WIDTH, GRID_HEIGHT, CELL_SIZE = get_grid_params()

# place and size of market
MARKET_PLACE = 'Middle'
market_size = 6

# Grid distribution parameters
BLOB_SIZE = 3

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
YELLOW = (255, 255, 0)

market = np.full((GRID_HEIGHT, GRID_WIDTH), False, dtype=bool)
if SCENARIO == 'Market':
    if MARKET_PLACE == 'Middle':
        for x in range(int((GRID_WIDTH/2)-market_size), int((GRID_WIDTH/2)+market_size)):
            for y in range(int((GRID_HEIGHT/2)-market_size), int((GRID_HEIGHT/2)+market_size)):
                market[x][y] = True

wood = np.zeros((GRID_HEIGHT, GRID_WIDTH))
food = np.zeros((GRID_HEIGHT, GRID_WIDTH))

# number of resource cells
wood_cell_count = 0 
food_cell_count = 0

if DISTRIBUTION == 'Sides':
    # Resources in non-random positions
    for x in range(0,GRID_WIDTH):
        for y in range(0,8):
            if not market[x][y]:
                wood_cell_count += 1
                wood[x][y] = MAX_WOOD  #random.uniform(MIN_WOOD, MAX_WOOD)
    for x in range(0,GRID_WIDTH):
        for y in range(32,GRID_HEIGHT):
            if not market[x][y]:
                food_cell_count += 1
                food[x][y] = MAX_FOOD  #random.uniform(MIN_FOOD, MAX_FOOD)

elif DISTRIBUTION == 'Uniform':
    #TODO: uniform distibution, but low resources such that it supports a certain number of agents
    for x in range(0,GRID_WIDTH):
        for y in range(0,GRID_HEIGHT):
            if not market[x][y]:
                wood_cell_count += 1
                food_cell_count += 1
                wood[x][y] = MAX_WOOD  #random.uniform(MIN_WOOD, MAX_WOOD)
                food[x][y] = MAX_FOOD  #random.uniform(MIN_FOOD, MAX_FOOD)

elif DISTRIBUTION == 'RandomGrid':
    for x in range(0,GRID_WIDTH):
        for y in range(0,GRID_HEIGHT):
            if not market[x][y] and int(x/BLOB_SIZE) % 2 == 0 and int(y/BLOB_SIZE) % 2 == 0:
                if random.random() > 0.5:
                    wood_cell_count += 1
                    wood[x][y] = MAX_WOOD  #random.uniform(MIN_WOOD, MAX_WOOD)
                else:
                    food_cell_count += 1
                    food[x][y] = MAX_FOOD  #random.uniform(MIN_FOOD, MAX_FOOD)

total_food_regen = 2.1
total_wood_regen = 2.1
initial_food_qty = 420
initial_wood_qty = 420
initial_food_qty_cell = initial_food_qty / food_cell_count
initial_wood_qty_cell = initial_wood_qty / wood_cell_count
# normalize resource regeneration such that the total regen. is same regardless of number of resource cells)
food_regen_rate = total_food_regen / food_cell_count
wood_regen_rate = total_wood_regen / wood_cell_count
# normalize initial quantities
for x in range(0,GRID_WIDTH):
    for y in range(0,GRID_HEIGHT):
        if food[x][y] > 0:
            food[x][y] = initial_food_qty_cell
        if wood[x][y] > 0:
            wood[x][y] = initial_wood_qty_cell

resources = {
    'wood': wood,
    'food': food,
}
max_resources = copy.deepcopy(resources)

# Set up the agents
NUM_AGENTS = 200
agents = []
agent_colours = sns.color_palette('bright', n_colors=NUM_AGENTS)

gather_amount = 1.0

agent_positions = np.zeros([NUM_AGENTS, 2])

# Creating agents
for i in range(NUM_AGENTS):
    x = random.randint(0, GRID_WIDTH-2)
    y = random.randint(0, GRID_HEIGHT-2)
    color = (255.0,0.0,0.0) if y < GRID_HEIGHT/2 else (0.0,255.0,0.0)
    agent = Agent(i, x, y, AGENT_TYPE, color, market) #color = np.array(agent_colours[i])*255
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

    # Counting the nr of alive agents for automatic stopping
    nr_agents = 0

    # Update the agents
    for agent in agents:
        if agent.isAlive():
            agent.updateBehaviour()

            nr_agents += 1
            agent.update_time_alive()
            x, y = agent.getPos()
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, agent.getColor(), rect)
            
            # Do agent behaviour
            if TRADING:
                if (agent.getBehaviour() == 'trade_wood' or agent.getBehaviour() == 'trade_food') and SCENARIO == 'Baseline':
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
                                # print(f"TRADE at {agent.getPos()} at pos={agent_B.getPos()}")
                                # print(f"  Agent A = {agent.current_stock}, {agent.behaviour}")
                                # print(f"  Agent B = {agent_B.current_stock}, {agent_B.behaviour}")
                                traded_qty = agent.trade(agent_B)
                                traded = True
                                # print(f"  Qty traded: {traded_qty}")
                                agent.clearBlacklistedAgents()
                            else:
                                # If not compatible, find next nearest neighbor
                                if AGENT_TYPE == 'pathfind_neighbor':
                                    agent.removeClosestNeighbor()
                elif (agent.getBehaviour() == 'trade_wood' or agent.getBehaviour() == 'trade_food') and SCENARIO == 'Market':
                    market_idx = np.argwhere(market)
                    if [x, y] in market_idx:
                        traded = False
                        neighboring_cells = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
                        neighboring_cells.remove((0,0))
                        
                        while not traded and bool(neighboring_cells):
                            dx, dy = random.choice(neighboring_cells)
                            neighboring_cells.remove((dx, dy))
                            if 0 <= x+dx < GRID_WIDTH and 0 <= y+dy < GRID_HEIGHT and [x+dx, y+dy] in market_idx:
                                x_check = agent.getPos()[0] + dx
                                y_check = agent.getPos()[1] + dy
                                occupied, agent_B = cellAvailable(x_check, y_check, agents)
                                if agent_B is None:
                                    continue
                                if agent.compatible(agent_B):
                                    # print(f"TRADE at {agent.getPos()} at pos={agent_B.getPos()}")
                                    # print(f"  Agent {agent.getID()} = {agent.current_stock}, {agent.behaviour}")
                                    # print(f"  Agent {agent_B.getID()} = {agent_B.current_stock}, {agent_B.behaviour}")
                                    traded_qty = agent.trade(agent_B)               
                                    agent.updateBehaviour()
                                    agent_B.updateBehaviour()
                                    # print(f"  Qty traded: {traded_qty}")
                                    # print(f"  Agent {agent.getID()} = {agent.current_stock}, {agent.behaviour}")
                                    # print(f"  Agent {agent_B.getID()} = {agent_B.current_stock}, {agent_B.behaviour}")
                                    traded = True
                            


                        if traded:
                            agent.set_movement = 'random' 

                # Update the resource gathering
                chosen_resource = choose_resource(agent, resources, gather_amount) # make agent choose which resource to gather 
                #if able_to_take_resource(agent, chosen_resource, resources):
                take_resource(agent, chosen_resource, resources, gather_amount)
            
            # Upkeep of agents and check if agent can survive
            agent.upkeep()

            # closest distance to market
            agent.setClosestMarketPos(findClosestMarketPos(agent, market))

            # # Choose behaviour
            # agent.updateBehaviour() # Agent brain

            # closest distance to market
            agent.setClosestMarketPos(findClosestMarketPos(agent, market))

            # Probabalistic movement
            if random.uniform(0, 1) < MOVE_PROB:
                # Choose step
                preferred_direction = agent.chooseStep(market)
                moveAgent(preferred_direction, agent, agents)

            # Update market bool
            agent.setInMarket(in_market(agent, market))

            # Distance and indices of 5 nearest neighbors within view radius
            view_radius = 20
            dist, idx = positions_tree.query([[x, y]], k=5)
            for i, d in enumerate(dist[0]):
                if d > view_radius:
                    # neighbors_too_far += 1
                    np.delete(dist, i)
                    np.delete(idx, i)
            
            if len(idx) > 0:
                agent.setNearestNeighbors(agent_positions[idx][0])

            # closest distance to market
            agent.setClosestMarketPos(findClosestMarketPos(agent, market))

            # Update agent position for the KD-tree
            agent_positions[i] = [x, y]
        
        # Updating KD-tree
        positions_tree = KDTree(agent_positions)

    if regen_active:
        for resource in resources:
            regen_rate = food_regen_rate if resource == 'food' else wood_regen_rate  # get regen_rate for specific resource
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if resources[resource][x][y] < max_resources[resource][x][y] - regen_rate:
                        resources[resource][x][y] += regen_rate
                    else:
                        resources[resource][x][y] = max_resources[resource][x][y]  # Set to max
    
    # Clear the screen
    screen.fill(WHITE)

    # Draw resources
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            wood_value = wood[row][col]
            food_value = food[row][col]
            # Map the resource value to a shade of brown or green
            old_rendering = False
            if old_rendering:
                wood_color = DARK_BROWN if wood_value > 7.5 else LIGHT_BROWN if wood_value > 5 else BROWN if wood_value > 0 else WHITE
                food_color = DARK_GREEN if food_value > 7.5 else LIGHT_GREEN if food_value > 5 else GREEN if food_value > 0 else WHITE
                blended_color = (
                    max(min(255, int((wood_color[0] * wood_value + food_color[0] * food_value) / (wood_value + food_value + 1))), 0),
                    max(min(255, int((wood_color[1] * wood_value + food_color[1] * food_value) / (wood_value + food_value + 1))), 0),
                    max(min(255, int((wood_color[2] * wood_value + food_color[2] * food_value) / (wood_value + food_value + 1))), 0),
                    max(min(255, int(max(wood_value, food_value)*25)), 0)
                )
            else:
                if market[row][col]:
                    blended_color = YELLOW
                else:
                    inv_food_color = tuple(map(lambda i, j: i - j, WHITE, DARK_GREEN))
                    food_percentage = (food_value/initial_food_qty_cell)
                    #if row == 0 and col == 0:
                    #    print(f"{food_percentage=} = {food_value=} {initial_food_qty_cell=}")
                    inv_food_color = tuple(map(lambda i: i * food_percentage, inv_food_color))
                    food_color = tuple(map(lambda i, j: i - j, WHITE, inv_food_color))
                    inv_wood_color = tuple(map(lambda i, j: i - j, WHITE, BROWN))
                    wood_percentage = (wood_value/initial_wood_qty_cell)
                    inv_wood_color = tuple(map(lambda i: i * wood_percentage, inv_wood_color))
                    wood_color = tuple(map(lambda i, j: i - j, WHITE, inv_wood_color))
                    blended_color = tuple(map(lambda x, y: (x + y)/2, food_color, wood_color))

            rect = pygame.Rect(row * CELL_SIZE, col * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            draw_rect_alpha(screen, blended_color, rect)
    
    # Draw agents
    for agent in agents:
        if agent.isAlive():
            x, y = agent.getPos()
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, agent.getColor(), rect)

            # Draw wood and food bars
            # agent.wood_bar(screen)
            # agent.food_bar(screen)

    # Draw the grid
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (SCREEN_WIDTH, y))
                        
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


if SAVE_TO_FILE:
    # Time alive of agents distribution
    alive_times = np.zeros(NUM_AGENTS)
    for agent in agents:
        alive_times[agent.id] = agent.time_alive
    
    # only record deaths, kaplan meier and cox model both can deal with this 
    events = np.ones(len(alive_times))


    # Saving data to file
    file_path = f'outputs/{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}-{NUM_AGENTS}-{TRADING}.csv'

    if not os.path.exists(file_path):
        empty = pd.DataFrame({'ignore': [0]*time})
        empty.to_csv(file_path, index=False)

    data = pd.read_csv(file_path)

    data = pd.DataFrame({'T': alive_times, 'E': events, 'Agent_type': AGENT_TYPE, 'Scenario': SCENARIO, 'Trading': TRADING, 'RUN_NUMBER': RUN_NR})

    # Assign the adjusted events and alive_times to DataFrame columns
    data.to_csv(file_path, index=False)
