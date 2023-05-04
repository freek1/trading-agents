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
            "wood": 10,
            "food": 10,
        }
        self.upkeepCost = {
            "wood": 1,
            "food": 1,
        }
        self.predisposition = predispositions
        self.specialization = specialization
        self.pos_backlog = []
        self.gathered_resource_backlog = []
        self.movement = "pathfinding"  # ["pathfinding", "random"]
        self.goal_position = (8,8)  # y, x
    
    def updateBehaviour(self):
        if self.current_stock['wood'] < 10:
            self.movement = 'pathfinding'
            self.goal_position = (1,1)
        elif self.current_stock['food'] < 10:
            self.movement = 'pathfinding'
            self.goal_position = (7,7) # How does it know where food is?
        # How does it know where wood is?
        # - Make goal location a radius
        # - Implement that agents cannot be on the same square 
        # - What to do when the goal is the same square? Go one next to it?
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