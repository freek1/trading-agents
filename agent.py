import random
import pygame

resources = ['wood', 'food']
RANDOM_AGENTS = True
TRADE_THRESHOLD = 1.2
TRADE_QTY = 1.0
UPKEEP_COST = 0.0

class Agent:
    def __init__(self, id, color, predispositions, specialization, GRID_WIDTH, GRID_HEIGHT):
        self.GRID_WIDTH = GRID_WIDTH
        self.GRID_HEIGHT = GRID_HEIGHT
        self.x = random.randint(0, GRID_WIDTH-1)
        self.y = random.randint(0, GRID_HEIGHT-1)
        self.id = id
        self.alive = True
        self.color = color
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
        self.wood_locations = [] # [(y_1, x_1), (y_2, x_2), ...] list
        self.food_locations = []
        self.predisposition = predispositions
        self.specialization = specialization
        self.movement = "random"  # "pathfinding" or "random"
        self.behaviour = 'gather'
        self.goal_position = (None, None)  # y, x
        # For wood and food bars
        self.bar_length = 200
        self.bar_ratio_wood = self.wood_capacity / self.bar_length
        self.bar_ratio_food = self.food_capacity / self.bar_length

    # TODO: make this responsive (SCREEN_WIDTH = 800) or remove
    def wood_bar(self, screen):    
        pygame.draw.rect(screen, self.color, (800 - self.bar_length - 10, (self.id * 3) * 10, self.current_stock['wood'] / self.bar_ratio_wood, 25))
        pygame.draw.rect(screen, (0, 0, 0), (800 - self.bar_length - 10, (self.id * 3) * 10, self.bar_length, 25), 4)

    def food_bar(self, screen):
        pygame.draw.rect(screen, self.color, (800 - self.bar_length - 10, (self.id * 3) * 10 + 70, self.current_stock['food'] / self.bar_ratio_food, 25))
        pygame.draw.rect(screen, (0, 0, 0), (800 - self.bar_length - 10, (self.id * 3) * 10 + 70, self.bar_length, 25), 4)
        

    def updateBehaviour(self):
        # If agent has no knowledge of wood or food locations, random walk.
        if not RANDOM_AGENTS:
            if len(self.wood_locations) == 0 or len(self.food_locations) == 0:
                self.movement = 'random'
            if self.current_stock['wood'] < 10:
                # If agent doesnt know a location, random walk.
                # If he does, pathfind to it
                if len(self.wood_locations) == 0:
                    self.movement = 'random'
                else:
                    self.movement = 'pathfinding'
                    self.goal_position = self.wood_locations.pop(0)
            elif self.current_stock['food'] < 10:
                if len(self.food_locations) == 0:
                    self.movement = 'random'
                else:
                    self.movement = 'pathfinding'
                    self.goal_position = self.food_locations.pop(0)
            else:
                self.movement = 'random'
                # TODO: trade?
                # self.movement = 'trade'
        # Update gather/trade behaviour
        if self.current_stock['wood']/self.current_stock['food'] > TRADE_THRESHOLD:
            self.behaviour = 'trade_wood' # means selling wood
        elif self.current_stock['food']/self.current_stock['wood'] > TRADE_THRESHOLD:
            self.behaviour = 'trade_food' # means selling food
        else: self.behaviour = 'gather'
    
    def chooseStep(self):
        dx, dy = 0, 0
        if self.movement == 'pathfinding':
            goal_y, goal_x = self.goal_position
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
        return dy, dx
    
    def compatible(self, agent_B):
        if self.behaviour == 'trade_wood' and agent_B.getBehaviour() == 'trade_food' \
        or self.behaviour == 'trade_food' and agent_B.getBehaviour() == 'trade_wood':
            return True

    def trade(self, agent_B, transaction_cost):
        traded_quantity = 0.0
        if self.behaviour == 'trade_wood':
            # Sell wood for food
            while not (self.tradeFinalized() or agent_B.tradeFinalized()):
                self.current_stock['wood'] -= TRADE_QTY
                agent_B.current_stock['wood'] += TRADE_QTY - transaction_cost
                agent_B.current_stock['food'] -= TRADE_QTY
                self.current_stock['food'] += TRADE_QTY - transaction_cost
                traded_quantity += TRADE_QTY
        else:
            # Sell food for wood
            while not (self.tradeFinalized() or agent_B.tradeFinalized()):
                self.current_stock['food'] -= TRADE_QTY
                agent_B.current_stock['food'] += TRADE_QTY - transaction_cost
                agent_B.current_stock['wood'] -= TRADE_QTY
                self.current_stock['wood'] += TRADE_QTY - transaction_cost
                traded_quantity += TRADE_QTY
        print(traded_quantity)        
        return traded_quantity
    
    def tradeFinalized(self):
        # Finalize trade if resource equilibrium is reached (diff < TRADE_QTY/2)
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

    def move(self, dy, dx):
        self.y += dy
        self.x += dx

    def upkeep(self):
        for resource in self.current_stock.keys():
            self.current_stock[resource] -= self.upkeep_cost[resource]
            if self.current_stock[resource] < 0:
                self.alive = False
        
    def gatherResource(self, chosen_resource):
        self.current_stock[chosen_resource] += self.getSpecificSpecialization(chosen_resource) # calculates amount based on specialization
    
    def isAt(self, x, y):
        return self.x == x and self.y == y

    def getPos(self):
        return self.y, self.x
    
    def getPredisposition(self):
        return self.predisposition
    
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
    
    def setPos(self, y, x):
        self.y = y
        self.x = x

    def getSpecificSpecialization(self, resource):
        return self.specialization[resources.index(resource)]
    
    def getID(self):
        return self.id