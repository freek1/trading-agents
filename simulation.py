import pygame
import random
import numpy as np
import math
import seaborn as sns
import copy
from agent import Agent

def draw_rect_alpha(surface, color, rect):
    ''' Draws a rectangle with an alpha channel
    Input: 
        surface: object
        color: tuple, RGB
        rect: tuple, (x, y, w, h)
    Output:
        None
    '''
    shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
    pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
    surface.blit(shape_surf, rect)

def choose_resource(agent, resources):
    ''' Returns a random resource from the list of resources
    Input: 
        agent: object
        resources: dict
    Output:
        chosen_resource: string, name of chosen resource
    '''
    return list(resources.keys())[np.random.choice(len(resources), p=agent.getPredisposition())]

def take_resource(agent: Agent, chosen_resource, resources):
    ''' Takes a resource from the chosen resource
    Input: 
        agent: object
        chosen_resource: string, name of chosen resource
        resources: dict
    Output:
        None
    '''
    y, x = agent.getPos()
    agent.gatherResource(chosen_resource) 
    resources[chosen_resource][y][x] -= agent.getSpecificSpecialization(chosen_resource)


def able_to_take_resource(agent, chosen_resource, resources):
    ''' Checks if the agent is able to take a resource
    Input: 
        agent: object
        chosen_resource: string, name of chosen resource
        resources: dict
    Output:
        bool, True if able to take resource, False if not
    '''
    y, x = agent.getPos()
    return agent.getCapacity(chosen_resource) > agent.getCurrentStock(chosen_resource) and resources[chosen_resource][y][x] >= 1

def find_nearest_resource(agent, resource):
    y_agent, x_agent = agent.getPos()
    closest_loc = (-np.inf, -np.inf)
    closest_dist = np.inf
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if resources[resource][y][x]>=1:
                if math.dist((y_agent, x_agent), (y, x))<closest_dist:
                    closest_dist = math.dist((y_agent, x_agent), (y, x))
                    closest_loc = y, x
    return closest_loc

def cellAvailable(x, y):
    """
    Returns True and agent if occupied
    """
    for agent in agents:
        if agent.isAt(x, y):
            return (False, agent)
    return (True, None)

def moveAgent(preferred_direction):
    # move agent to preferred direction if possible, otherwise move randomly
    y, x = agent.getPos()
    dy, dx = preferred_direction
    if 0 <= x+dx < GRID_WIDTH and  0 <= y+dy < GRID_HEIGHT:
        new_x = x + dx
        new_y = y + dy
        if cellAvailable(new_x, new_y)[0]:
            agent.move(dy, dx)

    else:
        found = False # available grid cell found
        possible_moves = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
        possible_moves.remove((0,0))
        while not found and possible_moves:
            dy,dx = random.choice(possible_moves)
            possible_moves.remove((dy,dx))
            if 0 <= x+dx < GRID_WIDTH and 0 <= y+dy < GRID_HEIGHT:
                new_x = x + dx
                new_y = y + dy
                if cellAvailable(new_x, new_y)[0]:
                    agent.move(dy, dx)
                    found = True


# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()
dt = 0
fps = 60
time = 1


# Set up the grid
CELL_SIZE = 20
GRID_WIDTH = 40 
GRID_HEIGHT = 40

# Set the dimensions of the screen
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE
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


wood = [[random.uniform(0, 10) for x in range(4,12)] for y in range(4,12)]
food = [[random.uniform(0, 10) for x in range(4,12)] for y in range(4,12)]

# Wood and food in non-random positions
wood = np.zeros((GRID_HEIGHT, GRID_WIDTH))
for i in range(0,8):
    for j in range(0,8):
        wood[i][j] = random.uniform(5, 10)

food = np.zeros((GRID_HEIGHT, GRID_WIDTH))
for i in range(32,40):
    for j in range(32,40):
        food[i][j] = random.uniform(5, 10)

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



for i in range(NUM_AGENTS):
    # create predispotion for resources 
    predispotions = np.random.uniform(0.4, 0.6, len(resources)) # probability of chosing that particular resource
    predispotions /= predispotions.sum()
    specialization = np.random.uniform(1, 3, len(resources)) # multiplier

    agent = Agent(i, np.array(agent_colours[i])*255, predispotions, specialization, GRID_WIDTH, GRID_HEIGHT)
    agents.append(agent)

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
                max(min(255, int((wood_color[0] * wood_value + food_color[0] * food_value) / (wood_value + food_value + 1))), 0), # I added +1 to prevent / 0 errors
                max(min(255, int((wood_color[1] * wood_value + food_color[1] * food_value) / (wood_value + food_value + 1))), 0),
                max(min(255, int((wood_color[2] * wood_value + food_color[2] * food_value) / (wood_value + food_value + 1))), 0),
                max(min(255, int(max(wood_value, food_value)*25)), 0)
            ) 
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            draw_rect_alpha(screen, blended_color, rect)

    # Draw the grid
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (SCREEN_WIDTH, y))

    # Update the agents
    for agent in agents:
        if agent.isAlive():
            y, x = agent.getPos()
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, agent.getColor(), rect)

            # Draw wood and food bars
            agent.wood_bar(screen)
            agent.food_bar(screen)

            # Check in surrounding area (9 cells) for resources
            # And update agent beliefs of their locations
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if y + dy > 0 and y + dy < GRID_HEIGHT and x + dx > 0 and x + dx < GRID_WIDTH:
                        y_check = (y + dy)
                        x_check = (x + dx)
                        if resources['wood'][y_check][x_check] > 0:
                            agent.addWoodLocation((y_check, x_check))
                        else: 
                            agent.removeWoodLocation((y_check, x_check))
                        if resources['food'][y_check][x_check] > 0:
                            agent.addFoodLocation((y_check, x_check))
                        else:
                            agent.removeFoodLocation((y_check, x_check))
            
            # Do agent behaviour
            if agent.getBehaviour() == 'trade_wood' or agent.getBehaviour() == 'trade_food':
                traded = False
                neighboring_cells = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
                neighboring_cells.remove((0,0))
                while not traded and neighboring_cells:
                    dy, dx = random.choice(neighboring_cells)
                    neighboring_cells.remove((dy, dx))
                    if 0 <= x+dx < GRID_WIDTH and 0 <= y+dy < GRID_HEIGHT:
                        y_check = agent.getPos()[0] + dy
                        x_check = agent.getPos()[1] + dx
                        occupied, agent_B = cellAvailable(y_check, x_check)
                        if agent_B is None:
                            continue
                        if agent.compatible(agent_B):
                            print(f"TRADE at {agent.getPos()} at pos={agent_B.getPos()}")
                            print(f"  Agent A = {agent.current_stock}, {agent.behaviour}")
                            print(f"  Agent B = {agent_B.current_stock}, {agent_B.behaviour}")
                            traded_qty = agent.trade(agent_B, transaction_cost)
                            traded = True
                            print(f"  Qty traded: {traded_qty}")
            # Update the resource gathering
            else:
                chosen_resource = choose_resource(agent, resources) # make agent choose which resource to gather based on it's predisposition
                if able_to_take_resource(agent, chosen_resource, resources):
                    take_resource(agent, chosen_resource, resources)
            
                # Upkeep of agents and check if agent can survive
                agent.upkeep()
                

            # Choose behaviour
            agent.updateBehaviour() # Agent brain
            preferred_direction = agent.chooseStep()
            moveAgent(preferred_direction)

        
    if regen_active:
        for maximum_resource in maximum_resources:
            for row in range(GRID_HEIGHT):
                for column in range(GRID_WIDTH):
                    if maximum_resources[maximum_resource][row][column] > resources[maximum_resource][row][column]:
                        resources[f'{maximum_resource}'][row][column] += regen_amount
                        


    
    # Update the display
    pygame.display.flip()

    clock.tick(fps)
    dt = clock.tick(fps)/100
    time += 1
    

# Clean up
pygame.quit()

