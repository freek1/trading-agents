import random

class Agent:
    def __init__(self, id, color, predispotions, GRID_WIDTH, GRID_HEIGHT):
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
        self.predispostion = predispotions
        self.pos_backlog = []
        self.gathered_resource_backlog = []
        self.movement = "pathfinding"  # ["pathfinding", "random"]
        self.goal_position = (8,8)  # y, x
    