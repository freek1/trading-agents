import random
import pygame
import numpy as np

resources = ['wood', 'food']
AGENT_TYPE = 'random' # 'random', 'pathfind_neighbor', 'pathfind_market'
TRADE_THRESHOLD = 1.5
TRADE_QTY = 1.0
UPKEEP_COST = 0.02

class Agent:
    def __init__(self, id, x, y, color, GRID_WIDTH, GRID_HEIGHT):
        self.GRID_WIDTH = GRID_WIDTH
        self.GRID_HEIGHT = GRID_HEIGHT
        self.x = x
        self.y = y
        self.id = id
        self.alive = True
        self.color = color
        self.time_alive = 0
        self.wood_capacity = 30
        self.food_capacity = 30
        self.current_stock = {
            "wood": 15,
            "food": 15,
        }
        self.upkeep_cost = {
            "wood": UPKEEP_COST,
            "food": UPKEEP_COST,
        }
        self.behaviour = '',  # 'trade_wood', 'trade_food'
        self.movement = 'random' # initialize as random for all agent types, since their movement changes only when wanting to trade
        self.goal_position = (None, None)  # x, y
        self.nearest_neighbors = [] # List of (x,y) of the nearest neighbors
        self.blacklisted_agents = [[]] # List of (x,y) of the blacklisted agents
        
    def update_time_alive(self):
        self.time_alive += 1

    def get_time_alive(self):
        return self.time_alive
    
    def set_movement(self, movement):
        self.movement = movement

    def updateBehaviour(self):
        # Update trade behaviour
        ratio = self.current_stock['wood']/self.current_stock['food']
        if ratio > TRADE_THRESHOLD and sum(self.current_stock.values()) > 5:
            self.behaviour = 'trade_wood' # means selling wood
            # adapt movement behaviour
            match AGENT_TYPE:
                case 'random':
                    self.movement = "random"
                case 'pathfind_neighbor':
                    self.movement = "pathfind_neighbor"
                case 'pathfind_market':
                    self.movement = "pathfind_market"

        elif 1/ratio > TRADE_THRESHOLD and sum(self.current_stock.values()) > 5:
            self.behaviour = 'trade_food' # means selling food

            match AGENT_TYPE:
                case 'random':
                    self.movement = "random"
                case 'pathfind_neighbor':
                    self.movement = "pathfind_neighbor"
                case 'pathfind_market':
                    self.movement = "pathfind_market"
    
    def chooseStep(self):
        ''' Pick the next direction to walk in for the agent
        Input:
            self: agent
        Output:
            dx
            dy
        '''
        dx, dy = 0, 0
        # compute where to
        if self.movement == 'pathfind_neighbor':
            # TODO: find out how it can end up with unequal amount of values (it should be 5x2=10, sometimes its 9)
            if len(self.nearest_neighbors.reshape(-1)) % 2 != 0:
                return dx, dy
            if len(self.blacklisted_agents.reshape(-1)) % 2 != 0:
                return dx, dy
            
            self.nearest_neighbors.reshape(-1, 2)

            print(self.nearest_neighbors, self.blacklisted_agents)
            set1 = set(tuple(x) for x in self.nearest_neighbors)
            set2 = set(tuple(x) for x in self.blacklisted_agents)
            not_blacklisted_neighbors = list(set1 - set2)
            
            x_nn, y_nn = not_blacklisted_neighbors[0]
            self.goal_position = (x_nn, y_nn)

        if self.movement == 'pathfind_market':
            self.goal_position = find_closest_market_position()
        
        # move
        if 'pathfind' in self.movement:
            goal_x, goal_y = self.goal_position
            if goal_y < self.y:
                dy = -1
            elif goal_y > self.y:
                dy = 1
            if goal_x < self.x:
                dx = -1
            elif goal_x > self.x:
                dx = 1
        elif self.movement == 'random':
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)

        return dx, dy
    
    def compatible(self, agent_B):
        if self.behaviour == 'trade_wood' and agent_B.getBehaviour() == 'trade_food' \
        or self.behaviour == 'trade_food' and agent_B.getBehaviour() == 'trade_wood':
            return True

    def trade(self, agent_B, transaction_cost):
        old_color = self.color
        traded_quantity = 0.0
        if self.behaviour == 'trade_wood':
            # Sell wood for food 
            while not (self.tradeFinalized() or agent_B.tradeFinalized()):
                self.color = (0,0,0)
                self.current_stock['wood'] -= TRADE_QTY
                agent_B.current_stock['wood'] += TRADE_QTY #- transaction_cost
                agent_B.current_stock['food'] -= TRADE_QTY
                self.current_stock['food'] += TRADE_QTY #- transaction_cost
                traded_quantity += TRADE_QTY
        elif self.behaviour == 'trade_food':
            # Sell food for wood
            while not (self.tradeFinalized() or agent_B.tradeFinalized()):
                self.color = (0,0,0)
                self.current_stock['food'] -= TRADE_QTY
                agent_B.current_stock['food'] += TRADE_QTY #- transaction_cost
                agent_B.current_stock['wood'] -= TRADE_QTY
                self.current_stock['wood'] += TRADE_QTY #- transaction_cost
                traded_quantity += TRADE_QTY
            
        # Return to not trading
        self.behaviour = ''

        return traded_quantity
    
    def removeClosestNeighbor(self):
        self.blacklisted_agents.append(self.nearest_neighbors[0])
        self.nearest_neighbors = np.delete(self.nearest_neighbors, 0)
    
    def tradeFinalized(self):
        # Finalize trade if resource equilibrium is reached (diff < TRADE_QTY)
        return abs(self.current_stock['wood'] - self.current_stock['food']) <= TRADE_QTY

    def addWoodLocation(self, pos):
        if pos not in self.wood_locations:
            self.wood_locations.append(pos)
        
    def removeWoodLocation(self, pos):
        if pos in self.wood_locations:
            self.wood_locations.remove(pos)

    def addFoodLocation(self, pos):
        if pos not in self.food_locations:
            self.food_locations.append(pos)

    def removeFoodLocation(self, pos):
        if pos in self.food_locations:
            self.food_locations.remove(pos)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def upkeep(self):
        for resource in self.current_stock.keys():
            self.current_stock[resource] -= self.upkeep_cost[resource]
            if self.current_stock[resource] < 0:
                self.alive = False
        
    def gatherResource(self, chosen_resource, gather_amount):
        self.current_stock[chosen_resource] += gather_amount
    
    def isAt(self, x, y):
        return self.x == x and self.y == y

    def getPos(self):
        return self.x, self.y
       
    def getBehaviour(self):
        return self.behaviour
    
    def getColor(self):
        return self.color
        
    def isAlive(self):
        return self.alive
    
    def getCapacity(self, chosen_resource):
        if chosen_resource == 'wood':
            return self.wood_capacity
        elif chosen_resource == 'food':
            return self.food_capacity
        
    def getCurrentStock(self, chosen_resource):
        return self.current_stock[chosen_resource]
    
    def setPos(self, x, y):
        self.x = x
        self.y = y
    
    def getID(self):
        return self.id
    
    def setNearestNeighbors(self, nearest_neighbors):
        self.nearest_neighbors = nearest_neighbors

    def clearBlacklistedAgents(self):
        self.blacklisted_agents = []