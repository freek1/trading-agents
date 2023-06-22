import random
import pygame
import numpy as np
import math

resources = ["wood", "food"]
TRADE_THRESHOLD = 1.5
UPKEEP_COST = 0.035  # was 0.02

DARK_BROWN = (60, 40, 0)
DARK_GREEN = (0, 102, 34)
PINK = (255, 192, 203)
ORANGE = (240, 70, 0)


class Agent:
    def __init__(self, id, x, y, agent_type, color, market):
        self.x = x
        self.y = y
        self.id = id
        self.alive = True
        self.color = color
        self.time_alive = 0
        self.wood_capacity = 30
        self.food_capacity = 30
        self.current_stock = {
            "wood": random.uniform(4, 8),
            "food": random.uniform(4, 8),
        }
        self.upkeep_cost = {
            "wood": UPKEEP_COST,
            "food": UPKEEP_COST,
        }
        self.behaviour = ("",)  # 'trade_wood', 'trade_food'
        self.movement = "random"  # initialize as random for all agent types, since their movement changes only when wanting to trade
        self.goal_position = (None, None)  # x, y
        self.nearest_neighbors = []  # List of (x,y) of the nearest neighbors
        self.closest_market_pos = (None, None)
        self.agent_type = agent_type
        self.in_market = False
        self.market = market
        self.treshold_new_neighbours = 0

    def print_info(self):
        if self.alive:
            print(f"{self.current_stock=} \n{self.behaviour=} \n{self.getPos()=}")

    def setInMarket(self, in_market):
        self.in_market = in_market

    def update_time_alive(self):
        self.time_alive += 1

    def get_time_alive(self):
        return self.time_alive

    def set_movement(self, movement):
        self.movement = movement

    def updateBehaviour(self, positions_tree, agent_positions, agents, k, view_radius):
        # Update trade behaviour
        ratio = self.calculateResourceRatio("wood", "food")
        if (
            ratio > TRADE_THRESHOLD
        ):
            self.color = DARK_BROWN
            self.behaviour = "trade_wood"  # means selling wood
            # adapt movement behaviour
            self.movement = self.agent_type
            if len(self.nearest_neighbors)==0 and self.treshold_new_neighbours==0:
                self.getSetClosestNeighbor(positions_tree, agents, min(k, len(agent_positions)), view_radius)
                self.treshold_new_neighbours=50
        elif (
            1 / ratio > TRADE_THRESHOLD
        ):
            self.color = DARK_GREEN
            self.behaviour = "trade_food"  # means selling food
            self.movement = self.agent_type
            if len(self.nearest_neighbors)==0 and self.treshold_new_neighbours==0:
                self.getSetClosestNeighbor(positions_tree, agents, min(k, len(agent_positions)), view_radius)
                self.treshold_new_neighbours=50
        else:
            self.color = ORANGE
            self.behaviour = ""
            self.movement = "random"
            
    def getSetClosestNeighbor(self, positions_tree, agents, k, view_radius):
        # Update agent position for the KD-tree
        x, y = self.getPos()
        
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
                if self != agents[ids]:
                    neighboring_agents.append(agents[ids])
            self.setNearestNeighbors(neighboring_agents)
    
    def chooseStep(self, agent_positions):
        """Pick the next direction to walk in for the agent
        Input:
            self: agent
        Output:
            dx
            dy
        """
        dx, dy = 0, 0
        # compute where to
        if self.movement == "pathfind_neighbor" and len(self.nearest_neighbors)>0: 
            # If it could not find any suitable neighbors, move randomly for 100 timesteps
            # -> Force it to move randomly
            x_nn, y_nn = self.nearest_neighbors[0].getPos()
            self.goal_position = [x_nn, y_nn]
            
        if len(self.nearest_neighbors)==0 and self.movement == "pathfind_neighbor":
            self.movement = "random"

        if self.movement == "pathfind_market":
            self.goal_position = self.closest_market_pos
            if self.in_market:
                self.movement = "random"
            else:
                self.movement = "pathfind_market"

        # move
        if "pathfind" in self.movement:
            goal_x, goal_y = self.goal_position
            if goal_y < self.y:
                dy = -1
            elif goal_y > self.y:
                dy = 1
            if goal_x < self.x:
                dx = -1
            elif goal_x > self.x:
                dx = 1
                
        elif self.movement == "random":
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)

        return dx, dy

    def compatible(self, agent_B):
        """Compatible if both agents are in market when this is the simulation situation."""
        if self.agent_type == "pathfind_market":
            if self.in_market and agent_B.in_market:
                if (
                    self.behaviour == "trade_wood"
                    and agent_B.getBehaviour() == "trade_food"
                ) or (
                    self.behaviour == "trade_food"
                    and agent_B.getBehaviour() == "trade_wood"
                ):
                    return True
        elif self.agent_type != "pathfind_market" and (
            (self.behaviour == "trade_wood" and agent_B.getBehaviour() == "trade_food")
            or (
                self.behaviour == "trade_food"
                and agent_B.getBehaviour() == "trade_wood"
            )
        ):
            return True
        else:
            return False

    def trade(self, agent_B):
        traded_quantity = 0.0
        minimum_difference = min(abs(self.current_stock["wood"] - self.current_stock["food"]), abs(agent_B.current_stock["wood"] - agent_B.current_stock["food"]))
        traded_quantity = minimum_difference/2.0
        # Sell wood for food
        if self.behaviour == "trade_wood":
            self.color = PINK
            agent_B.setColor = PINK
            self.current_stock["wood"] -= traded_quantity
            agent_B.current_stock["wood"] += traded_quantity
            agent_B.current_stock["food"] -= traded_quantity
            self.current_stock["food"] += traded_quantity
        # Sell food for wood  
        elif self.behaviour == "trade_food":
            self.color = PINK
            agent_B.setColor = PINK
            self.current_stock["food"] -= traded_quantity
            agent_B.current_stock["food"] += traded_quantity
            agent_B.current_stock["wood"] -= traded_quantity
            self.current_stock["wood"] += traded_quantity

        return traded_quantity

    def removeClosestNeighbor(self):
        """Removes closest neighbor from list"""
        self.nearest_neighbors.pop(0)
        
    def findNonMarketSquare(self):
        idx_market_false = np.argwhere(np.invert(self.market))
        smallest_distance = np.inf
        x_nmp, y_nmp = 0, 0
        for x_market, y_market in idx_market_false:
            distance = math.dist([x_market, y_market], [self.x, self.y])
            if distance < smallest_distance:
                smallest_distance = distance
                x_nmp, y_nmp = x_market, y_market
        return x_nmp, y_nmp

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

    def setColor(self, color):
        self.color = color

    def isAlive(self):
        return self.alive

    def preferredResource(self):
        if self.current_stock["food"] < self.current_stock["wood"]:
            return "food", self.calculateResourceRatio("food", "wood")
        else:
            return "wood", self.calculateResourceRatio("wood", "food")

    def calculateResourceRatio(self, resource_1: str, resource_2: str):
        return self.current_stock[resource_1] / self.current_stock[resource_2]

    def getCapacity(self, chosen_resource):
        if chosen_resource == "wood":
            return self.wood_capacity
        elif chosen_resource == "food":
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

    def setClosestMarketPos(self, closest_market_pos):
        self.closest_market_pos = closest_market_pos
        
    def getNearestNeigbors(self):
        return self.nearest_neighbors
    
    def getTresholdNewNeighbours(self):
        return self.treshold_new_neighbours
    
    def updateTresholdNewNeighbours(self):
        self.treshold_new_neighbours -= 1
