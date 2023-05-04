import random

resources = ['wood', 'food']

class Agent:
    def __init__(self, id, color, predispositions, specialization, GRID_WIDTH, GRID_HEIGHT):
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
        self.upkeepCost = {
            "wood": 1,
            "food": 1,
        }
        self.wood_locations = [] # [(y_1, x_1), (y_2, x_2), ...] list
        self.food_locations = []
        self.predisposition = predispositions
        self.specialization = specialization
        self.pos_backlog = []
        self.gathered_resource_backlog = []
        self.movement = "random"  # ["pathfinding", "random"]
        self.goal_position = (8,8)  # y, x
    
    def updateBehaviour(self):
        # If agent has no knowledge of wood or food locations, random walk.
        if len(self.wood_locations) == 0:
            self.movement = 'random'
        if len(self.food_locations) == 0:
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
    
    def addWoodLocation(self, pos):
        if pos not in self.wood_locations:
            self.wood_locations.append(pos)
            print('Agent', self.id, 'wood locations:', self.wood_locations)
        
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

    def getResBacklog(self):
        return self.gathered_resource_backlog

    def addResBacklog(self, resource):
        self.gathered_resource_backlog.append(resource)

    def setResBacklog(self, res_backlog):
        self.res_backlog = res_backlog

    def getPosBacklog(self):
        return self.pos_backlog
    
    def addPosBacklog(self, pos):
        self.pos_backlog.append(pos)

    def setPosBacklog(self, pos_backlog):
        self.pos_backlog = pos_backlog
    
    def upkeep(self):
        for resource in self.current_stock.keys():
            self.current_stock[resource] -= self.upkeepCost[resource]
            if self.current_stock[resource] < 0:
                self.alive = False
        
    def updateStock(self, chosen_resource):
        self.current_stock[chosen_resource] += self.getSpecificSpecialization(chosen_resource) # calculates amount based on specialization
    
    def isAt(self, x, y):
        return self.x == x and self.y == y

    def getPos(self):
        return self.y, self.x
    
    def getPredisposition(self):
        return self.predisposition
    
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