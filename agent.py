import random
import numpy as np
import math

from neural_agent import NeuralAgent

resources = ["wood", "food"]
t_ra_de_t_hr_es_ho_ld = 1.5
u_pk_ee_p_c_os_t = 0.035  # was 0.02

d_ar_k_b_ro_wn = (60, 40, 0)
d_ar_k_g_re_en = (0, 102, 34)
p_in_k = (255, 192, 203)
o_ra_ng_e = (240, 70, 0)


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
            "wood": u_pk_ee_p_c_os_t,
            "food": u_pk_ee_p_c_os_t,
        }
        self.behaviour = ("",)  # 'trade_wood', 'trade_food'
        self.agent_type = agent_type
        if not self.agent_type == 'neural':
            self.movement = "random"  # initialize as random for all agent types, since their movement changes only when wanting to trade
        else:
            self.movement = 'neural_movement'
        self.goal_position = (None, None)  # x, y
        self.nearest_neighbors = []  # list of (x,y) of the nearest neighbors
        self.closest_market_pos = (None, None)
        self.in_market = False
        self.market = market
        self.treshold_new_neighbours = 0
        self.utility = 0

    def print_info(self):
        if self.alive:
            print(f"{self.current_stock=} \n{self.behaviour=} \n{self.get_pos()=}")

    def set_in_market(self, in_market):
        self.in_market = in_market

    def update_time_alive(self):
        self.time_alive += 1

    def get_time_alive(self):
        return self.time_alive

    def set_movement(self, movement):
        self.movement = movement

    def update_behaviour(self, positions_tree, agent_positions, agents, k, view_radius):
        # update trade behaviour
        
        ratio = self.calculate_resource_ratio("wood", "food")
        if (
            ratio > t_ra_de_t_hr_es_ho_ld
        ):
            self.color = d_ar_k_b_ro_wn
            self.behaviour = "trade_wood"  # means selling wood
            # adapt movement behaviour
            self.movement = self.agent_type
            if len(self.nearest_neighbors)==0 and self.treshold_new_neighbours==0:
                self.get_set_closest_neighbor(positions_tree, agents, min(k, len(agent_positions)), view_radius)
                self.treshold_new_neighbours=50
        elif (
            1 / ratio > t_ra_de_t_hr_es_ho_ld
        ):
            self.color = d_ar_k_g_re_en
            self.behaviour = "trade_food"  # means selling food
            self.movement = self.agent_type
            if len(self.nearest_neighbors)==0 and self.treshold_new_neighbours==0:
                self.get_set_closest_neighbor(positions_tree, agents, min(k, len(agent_positions)), view_radius)
                self.treshold_new_neighbours=50
        else:
            self.color = o_ra_ng_e
            self.behaviour = ""
            if not self.agent_type == 'neural':
                self.movement = "random"
            
    def get_set_closest_neighbor(self, positions_tree, agents, k, view_radius):
        # update agent position for the k_d-tree
        x, y = self.get_pos()
        
        # distance and indices of 5 nearest neighbors within view radius
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
            self.set_nearest_neighbors(neighboring_agents)
    
    def choose_step(self, agent_positions):
        """pick the next direction to walk in for the agent
        input:
            self: agent
        output:
            dx
            dy
        """
        dx, dy = 0, 0
        # compute where to
        if self.movement == "pathfind_neighbor" and len(self.nearest_neighbors)>0: 
            # if it could not find any suitable neighbors, move randomly for 100 timesteps
            # -> force it to move randomly
            x_nn, y_nn = self.nearest_neighbors[0].get_pos()
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
                
        if self.movement == "random":
            dx = random.randint(-1, 1)
            dy = random.randint(-1, 1)

        if self.agent_type == 'neural':
            util = self.utility
            


        return dx, dy

    def compatible(self, agent_b):
        """compatible if both agents are in market when this is the simulation situation."""
        if self.agent_type == "pathfind_market":
            if self.in_market and agent_b.in_market:
                if (
                    self.behaviour == "trade_wood"
                    and agent_b.get_behaviour() == "trade_food"
                ) or (
                    self.behaviour == "trade_food"
                    and agent_b.get_behaviour() == "trade_wood"
                ):
                    return True
        elif self.agent_type != "pathfind_market" and (
            (self.behaviour == "trade_wood" and agent_b.get_behaviour() == "trade_food")
            or (
                self.behaviour == "trade_food"
                and agent_b.get_behaviour() == "trade_wood"
            )
        ):
            return True
        else:
            return False

    def trade(self, agent_b):
        traded_quantity = 0.0
        minimum_difference = min(abs(self.current_stock["wood"] - self.current_stock["food"]), abs(agent_b.current_stock["wood"] - agent_b.current_stock["food"]))
        traded_quantity = minimum_difference/2.0
        # sell wood for food
        if self.behaviour == "trade_wood":
            self.color = p_in_k
            agent_b.set_color = p_in_k
            self.current_stock["wood"] -= traded_quantity
            agent_b.current_stock["wood"] += traded_quantity
            agent_b.current_stock["food"] -= traded_quantity
            self.current_stock["food"] += traded_quantity
        # sell food for wood  
        elif self.behaviour == "trade_food":
            self.color = p_in_k
            agent_b.set_color = p_in_k
            self.current_stock["food"] -= traded_quantity
            agent_b.current_stock["food"] += traded_quantity
            agent_b.current_stock["wood"] -= traded_quantity
            self.current_stock["wood"] += traded_quantity

        return traded_quantity

    def remove_closest_neighbor(self):
        """removes closest neighbor from list"""
        self.nearest_neighbors.pop(0)
        
    def find_non_market_square(self):
        idx_market_False = np.argwhere(np.invert(self.market))
        smallest_distance = np.inf
        x_nmp, y_nmp = 0, 0
        for x_market, y_market in idx_market_False:
            distance = math.dist([x_market, y_market], [self.x, self.y])
            if distance < smallest_distance:
                smallest_distance = distance
                x_nmp, y_nmp = x_market, y_market
        return x_nmp, y_nmp

    def add_wood_location(self, pos):
        if pos not in self.wood_locations:
            self.wood_locations.append(pos)

    def remove_wood_location(self, pos):
        if pos in self.wood_locations:
            self.wood_locations.remove(pos)

    def add_food_location(self, pos):
        if pos not in self.food_locations:
            self.food_locations.append(pos)

    def remove_food_location(self, pos):
        if pos in self.food_locations:
            self.food_locations.remove(pos)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def upkeep(self):
        ''' Update upkeep and update utility value'''
        for resource in self.current_stock.keys():
            self.current_stock[resource] -= self.upkeep_cost[resource]
            if self.current_stock[resource] < 0:
                self.alive = False

            # Update utility
            self.utility += self.current_stock[resource]

    def gather_resource(self, chosen_resource, gather_amount):
        self.current_stock[chosen_resource] += gather_amount

    def is_at(self, x, y):
        return self.x == x and self.y == y

    def get_pos(self):
        return self.x, self.y

    def get_behaviour(self):
        return self.behaviour

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color

    def is_alive(self):
        return self.alive

    def preferred_resource(self):
        if self.current_stock["food"] < self.current_stock["wood"]:
            return "food", self.calculate_resource_ratio("food", "wood")
        else:
            return "wood", self.calculate_resource_ratio("wood", "food")

    def calculate_resource_ratio(self, resource_1: str, resource_2: str):
        return self.current_stock[resource_1] / self.current_stock[resource_2]

    def get_capacity(self, chosen_resource):
        if chosen_resource == "wood":
            return self.wood_capacity
        elif chosen_resource == "food":
            return self.food_capacity

    def get_current_stock(self, chosen_resource):
        return self.current_stock[chosen_resource]

    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def get_i_d(self):
        return self.id

    def set_nearest_neighbors(self, nearest_neighbors):
        self.nearest_neighbors = nearest_neighbors

    def set_closest_market_pos(self, closest_market_pos):
        self.closest_market_pos = closest_market_pos
        
    def get_nearest_neigbors(self):
        return self.nearest_neighbors
    
    def get_treshold_new_neighbours(self):
        return self.treshold_new_neighbours
    
    def update_treshold_new_neighbours(self):
        self.treshold_new_neighbours -= 1
