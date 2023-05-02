import pygame
import random
import numpy as np
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

# def pos_agent(agent):
#     ''' Returns y, x position tuple of the agent
#     Input: 
#         agent: object
#     Output:
#         y: int, y-pos
#         x: int, x-pos
#     '''
#     x = agent['x']
#     y = agent['y']
#     return (y, x)

def choose_resource(agent, resources):
    ''' Returns a random resource from the list of resources
    Input: 
        agent: object
        resources: dict
    Output:
        chosen_resource: string, name of chosen resource
    '''
    return list(resources.keys())[np.random.choice(len(resources), p=agent['predispostion'])]

def take_resource(agent, chosen_resource, resources):
    ''' Takes a resource from the chosen resource
    Input: 
        agent: object
        chosen_resource: string, name of chosen resource
        resources: dict
    Output:
        None
    '''
    #y, x = pos_agent(agent)
    y, x = agent.getPos()
    #agent['current_stock'][f'{chosen_resource}'] += 1
    agent.updateStock(chosen_resource, 1)
    resources[chosen_resource][y][x] -= 1
    #agent['gathered_resource_backlog'].append(chosen_resource)
    agent.addResBacklog(chosen_resource)

def able_to_take_resource(agent, chosen_resource, resources):
    ''' Checks if the agent is able to take a resource
    Input: 
        agent: object
        chosen_resource: string, name of chosen resource
        resources: dict
    Output:
        bool, True if able to take resource, False if not
    '''
    return agent[f'{chosen_resource}_capacity'] > agent['current_stock'][f'{chosen_resource}'] and resources[chosen_resource][y][x] >= 1

def find_nearest_resource(agent, resource):
    y_agent, x_agent = pos_agent(agent)
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
    for agent in agents:
        if agent.isAt(x, y):
            return False
    return True

def moveAgent(preferred_direction):
    # move agent to preferred direction if possible, otherwise move randomly
    y, x = agent.getPos()
    dy, dx = preferred_direction
    new_x = x
    new_y = y
    if 0 <= x+dx < GRID_WIDTH:
        new_x += dx
    if 0 <= y+dy < GRID_HEIGHT:
        new_y += dy
    if cellAvailable(new_x, new_y):
        agent.move(dy, dx)
    else:
        found = False # available grid cell found
        possible_moves = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
        possible_moves.remove((0,0))
        while not found and possible_moves:
            dy,dx = random.choice(possible_moves)
            possible_moves.remove((dy,dx))
            new_x = x
            new_y = y
            if 0 <= x+dx < GRID_WIDTH:
                new_x += dx
            if 0 <= y+dy < GRID_HEIGHT:
                new_y += dy
            if cellAvailable(new_x, new_y):
                agent.move(dy, dx)
                found = True
        
    if True: # TODO: check if cell is occupied/not outside world
        # Keep the agent on the grid
        agent["x"] = max(0, min(GRID_WIDTH - 1, agent["x"]))
        agent["y"] = max(0, min(GRID_HEIGHT - 1, agent["y"]))
        agent.move(dy, dx)
    else:
        pass
        # TODO: move randomly to available cell
    

# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()
dt = 0
fps = 10
time = 1

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

wood = [[random.uniform(0, 10) for x in range(GRID_WIDTH)] for y in range(GRID_HEIGHT)]
food = [[random.uniform(0, 10) for x in range(GRID_WIDTH)] for y in range(GRID_HEIGHT)]

# Wood and food in non-random positions
wood = np.zeros((GRID_HEIGHT, GRID_WIDTH))
for i in range(GRID_HEIGHT):
    for j in range(GRID_WIDTH):
        if i in [0,1,2,3] and j in [0,1,2,3]:
            wood[i][j] = random.uniform(0, 10)

food = np.zeros((GRID_HEIGHT, GRID_WIDTH))
for i in range(GRID_HEIGHT):
    for j in range(GRID_WIDTH):
        if i in [5,6,7,8] and j in [5,6,7,8]:
            food[i][j] = random.uniform(0, 10)

resources = {
    'wood': wood,
    'food': food,
}

# Set up the agents
NUM_AGENTS = 2
agents = []
agent_colours = [RED, BLUE]

regen_rate = 5
regen_amount = 1
regen_active = False

upkeep_rate = 5
upkeep_cost = 1


for i in range(NUM_AGENTS):
    # create predispotion for resources 
    for j in range(len(resources)):
        predispotions = np.random.random(len(resources))
        predispotions /= predispotions.sum()
    # agent = {
    #     "x": random.randint(0, GRID_WIDTH-1),
    #     "y": random.randint(0, GRID_HEIGHT-1),
    #     "id": i,
    #     'alive' : True,
    #     "color": agent_colours[i],
    #     "wood_capacity":30,
    #     "food_capacity":30,
    #     "current_stock": {
    #         "wood": 10, 
    #         "food": 10,
    #     },
    #     "predispostion": predispotions, 
    #     "pos_backlog" : [],
    #     "gathered_resource_backlog" : [],
    #     "movement" : "pathfinding", # ["pathfinding", "random"]
    #     "goal_position" : (8,8), # y, x
    # }
    agent = Agent(i, agent_colours[i], predispotions, GRID_WIDTH, GRID_HEIGHT)
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
                int((wood_color[0] * wood_value + food_color[0] * food_value) / (wood_value + food_value + 1)), # I added +1 to prevent / 0 errors
                int((wood_color[1] * wood_value + food_color[1] * food_value) / (wood_value + food_value + 1)),
                int((wood_color[2] * wood_value + food_color[2] * food_value) / (wood_value + food_value + 1)),
                int(max(wood_value, food_value)*25)
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
        #if agent['alive']==True:
        if agent.isAlive():
            y, x = agent.getPos()
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            #pygame.draw.rect(screen, agent["color"], rect)
            pygame.draw.rect(screen, agent.getColor(), rect)

            # Update the resource gathering
            chosen_resource =  choose_resource(agent, resources) # make agent choose which resource to gather based on it's predisposition
            if able_to_take_resource(agent, chosen_resource, resources):
                take_resource(agent, chosen_resource, resources)
            else:
                #agent['gathered_resource_backlog'].append(None)
                agent.addResBacklog(None)
            
            # Upkeep of agents and check if agent can survive
            if time % upkeep_rate == 0:
                for resource in resources:
                    agent.upkeep()
                    # agent['current_stock'][f'{resource}'] -= upkeep_cost
                    # if agent['current_stock'][f'{resource}'] < 0:
                    #     agent['alive'] = False
                    #     pass
            
            # Add position to backlog, before agent moves
            #agent['pos_backlog'].append((y, x))
            agent.addPosBacklog((y, x))

            # Choose behaviour
            agent.updateBehaviour() # Agent brain
            preferred_direction = agent.chooseStep()
            moveAgent(preferred_direction)

# TODO: update using Agent class
            # regenerate resources on the field based on the backlog of the agents
            if len(agent["pos_backlog"]) >= regen_rate and regen_active:
                if (agent['gathered_resource_backlog'][0]) != None:
                    y, x = agent['pos_backlog'][0]
                    regen_resource = agent['gathered_resource_backlog'][0]
                    resources[regen_resource][y][x] += 1
                agent['pos_backlog'].pop(0)
                agent['gathered_resource_backlog'].pop(0)

    
    # Update the display
    pygame.display.flip()

    clock.tick(fps)
    dt = clock.tick(fps)/100
    time += 1

# Clean up
pygame.quit()

