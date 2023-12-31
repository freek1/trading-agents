import pygame
from agent import Agent
import numpy as np
import math
import random
from sklearn.neighbors import KDTree

# Set up the grid
CELL_SIZE = 20
GRID_WIDTH = 40 
GRID_HEIGHT = 40

def get_grid_params():
    ''' Returns the size of the grid
    Input: 
        None
    Output:
        tuple, (width, height)
    '''
    return GRID_WIDTH, GRID_HEIGHT, CELL_SIZE

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

def choose_resource(agent:Agent, resources, gather_amount):
    ''' Returns resource based on which resource is available
    Input: 
        agent: object
        resources: dict
    Output:
        chosen_resource: string, name of chosen resource
    '''
    x, y = agent.getPos()
    preferred, ratio = agent.preferredResource()
    if resources[preferred][x][y] >= gather_amount:
        return preferred
    elif resources[other_resource(preferred)][x][y] >= gather_amount:
        return other_resource(preferred)
    elif resources[preferred][x][y] < resources[other_resource(preferred)][x][y]*ratio:
        return other_resource(preferred)
    return preferred

def other_resource(resource: str):
    # Return the opposing resource name
    if resource == 'wood':
        return 'food'
    return 'wood'

def take_resource(agent: Agent, chosen_resource, resources, gather_amount):
    ''' Takes a resource from the chosen resource
    Input: 
        agent: object
        chosen_resource: string, name of chosen resource
        resources: dict
    Output:
        None
    '''
    x, y = agent.getPos()
    gathered = min(resources[chosen_resource][x][y], gather_amount)
    resources[chosen_resource][x][y] -= gathered
    agent.gatherResource(chosen_resource, gathered) 


def able_to_take_resource(agent, chosen_resource, resources):
    ''' Checks if the agent is able to take a resource
    Input: 
        agent: object
        chosen_resource: string, name of chosen resource
        resources: dict
    Output:
        bool, True if able to take resource, False if not
    '''
    if chosen_resource == None:
        return False
    x, y = agent.getPos()
    return agent.getCapacity(chosen_resource) > agent.getCurrentStock(chosen_resource) and resources[chosen_resource][x][y] >= 1

def find_nearest_resource(agent, resource, resources):
    x_agent, y_agent = agent.getPos()
    closest_loc = (-np.inf, -np.inf)
    closest_dist = np.inf
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if resources[resource][x][y]>=1:
                if math.dist((x_agent, y_agent), (x, y)) < closest_dist:
                    closest_dist = math.dist((x_agent, y_agent), (x, y))
                    closest_loc = x, y
    return closest_loc

def cellAvailable(x, y, agents):
    """
    Returns True and agent if occupied
    """
    for agent in agents:
        if agent.isAt(x, y):
            return (False, agent)
    return (True, None)

def moveAgent(preferred_direction, agent, agents):
    # move agent to preferred direction if possible, otherwise move randomly
    x, y = agent.getPos()
    dx, dy = preferred_direction
    # check if preffered direction is possible 
    if 0 <= x + dx < GRID_WIDTH and  0 <= y + dy < GRID_HEIGHT:
        new_x = x + dx
        new_y = y + dy
        if cellAvailable(new_x, new_y, agents)[0]:
            agent.move(dx, dy)
        else:
            found = False # available grid cell found
            possible_moves = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
            possible_moves.remove((0,0))
            while not found and possible_moves:
                dx,dy = random.choice(possible_moves)
                possible_moves.remove((dx, dy))
                if 0 <= x+dx < GRID_WIDTH and 0 <= y+dy < GRID_HEIGHT:
                    new_x = x + dx
                    new_y = y + dy
                    if cellAvailable(new_x, new_y, agents)[0]:
                        agent.move(dx, dy)
                        found = True

def findClosestMarketPos(agent: Agent, market):
    x, y = agent.getPos()
    idx_market_true = np.argwhere(market)
    smallest_distance = np.inf
    x_cmp, y_cmp = 0, 0
    for x_market, y_market in idx_market_true:
        distance = math.dist([x_market, y_market], [x, y])
        if distance < smallest_distance:
            smallest_distance = distance
            x_cmp, y_cmp = x_market, y_market
    return x_cmp, y_cmp

def in_market(agent: Agent, market):
    x, y = agent.getPos()
    return market[x][y]

def getSetClosestNeighbor(positions_tree, agents, agent:Agent, k, view_radius):
    # Update agent position for the KD-tree
    x, y = agent.getPos()
    
    # Distance and indices of 5 nearest neighbors within view radius
    view_radius = 20
    dist, idx = positions_tree.query([[x, y]], k=k)
    for i, d in enumerate(dist[0]):
        if d > view_radius:
            # neighbors_too_far += 1
            np.delete(dist, i)
            np.delete(idx, i)
    if len(idx) > 0:
        idx = idx[0]
        neighboring_agents = []
        for ids in idx:
            if agent != agents[ids]:
                neighboring_agents.append(agents[ids])
        agent.setNearestNeighbors(neighboring_agents)