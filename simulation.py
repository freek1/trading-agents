from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import random
import numpy as np
import seaborn as sns
import copy
import pandas as pd
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process

# Functions file
from funcs import *

# Agent class
from agent import Agent

ENABLE_RENDERING = False

def runSimulation(arg):
    # Unpacking input arguments
    print(arg)
    NUM_AGENTS, SCENARIO, AGENT_TYPE, MOVE_PROB, DISTRIBUTION, TRADING, SAVE_TO_FILE, RUN_NR = arg

    # Set the dimensions of the screen
    GRID_WIDTH, GRID_HEIGHT, CELL_SIZE = get_grid_params()
    SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
    SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

    if ENABLE_RENDERING:
        # Initialize Pygame
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    fps = 144
    clock = pygame.time.Clock()
    time = 1

    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BROWN = (153, 102, 0)
    YELLOW = (255, 255, 0)
    DARK_GREEN = (0, 200, 0)
    BLUE = (30, 70, 250)
    ORANGE = (240, 70, 0)

    # TODO: check if we never want to chance these variables, otherwise we need to take them out of the function (I dont think we ever want no regen or chance the maximum resources over runs though)
    REGEN_ACTIVE = True
    MAX_WOOD = 2
    MAX_FOOD = 2
    # Grid distribution parameters
    BLOB_SIZE = 3

    # place and size of market
    MARKET_PLACE = "Middle"
    MARKET_SIZE = 6

    market = np.full((GRID_HEIGHT, GRID_WIDTH), False, dtype=bool)
    if SCENARIO == "Market":
        if MARKET_PLACE == "Middle":
            for x in range(
                int((GRID_WIDTH / 2) - MARKET_SIZE), int((GRID_WIDTH / 2) + MARKET_SIZE)
            ):
                for y in range(
                    int((GRID_HEIGHT / 2) - MARKET_SIZE),
                    int((GRID_HEIGHT / 2) + MARKET_SIZE),
                ):
                    market[x][y] = True

    wood = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    food = np.zeros((GRID_HEIGHT, GRID_WIDTH))

    # number of resource cells
    wood_cell_count = 0
    food_cell_count = 0

    if DISTRIBUTION == "Sides":
        # Resources in non-random positions
        for x in range(0, GRID_WIDTH):
            for y in range(0, 8):
                if not market[x][y]:
                    wood_cell_count += 1
                    wood[x][y] = MAX_WOOD  # random.uniform(MIN_WOOD, MAX_WOOD)
        for x in range(0, GRID_WIDTH):
            for y in range(32, GRID_HEIGHT):
                if not market[x][y]:
                    food_cell_count += 1
                    food[x][y] = MAX_FOOD  # random.uniform(MIN_FOOD, MAX_FOOD)

    elif DISTRIBUTION == "Uniform":
        # TODO: uniform distibution, but low resources such that it supports a certain number of agents
        for x in range(0, GRID_WIDTH):
            for y in range(0, GRID_HEIGHT):
                if not market[x][y]:
                    wood_cell_count += 1
                    food_cell_count += 1
                    wood[x][y] = MAX_WOOD  # random.uniform(MIN_WOOD, MAX_WOOD)
                    food[x][y] = MAX_FOOD  # random.uniform(MIN_FOOD, MAX_FOOD)

    elif DISTRIBUTION == "RandomGrid":
        for x in range(0, GRID_WIDTH):
            for y in range(0, GRID_HEIGHT):
                if (
                    not market[x][y]
                    and int(x / BLOB_SIZE) % 2 == 0
                    and int(y / BLOB_SIZE) % 2 == 0
                ):
                    if random.random() > 0.5:
                        wood_cell_count += 1
                        wood[x][y] = MAX_WOOD  # random.uniform(MIN_WOOD, MAX_WOOD)
                    else:
                        food_cell_count += 1
                        food[x][y] = MAX_FOOD  # random.uniform(MIN_FOOD, MAX_FOOD)

    total_food_regen = 4.2 # originally 2.1
    total_wood_regen = 4.2
    initial_food_qty = 420
    initial_wood_qty = 420
    initial_food_qty_cell = initial_food_qty / food_cell_count
    initial_wood_qty_cell = initial_wood_qty / wood_cell_count
    # normalize resource regeneration such that the total regen. is same regardless of number of resource cells)
    food_regen_rate = total_food_regen / food_cell_count
    wood_regen_rate = total_wood_regen / wood_cell_count
    # normalize initial quantities
    for x in range(0, GRID_WIDTH):
        for y in range(0, GRID_HEIGHT):
            if food[x][y] > 0:
                food[x][y] = initial_food_qty_cell
            if wood[x][y] > 0:
                wood[x][y] = initial_wood_qty_cell

    resources = {
        "wood": wood,
        "food": food,
    }
    max_resources = copy.deepcopy(resources)

    # Set up the agents
    agents = []
    agent_colours = sns.color_palette("bright", n_colors=NUM_AGENTS)

    gather_amount = 1.0

    agent_positions = np.zeros([NUM_AGENTS, 2])

    # Creating agents
    for i in range(NUM_AGENTS):
        x = random.randint(0, GRID_WIDTH - 2)
        y = random.randint(0, GRID_HEIGHT - 2)
        #color = (255.0, 0.0, 0.0) if y < GRID_HEIGHT / 2 else (0.0, 255.0, 0.0)
        color = (255,110,0)
        agent = Agent(
            i, x, y, AGENT_TYPE, color, market
        )  # color = np.array(agent_colours[i])*255
        agents.append(agent)

        # Save agent position for the KD-tree
        agent_positions[i] = [x, y]
        # Initialize KDTree
        positions_tree = KDTree(agent_positions)

    # Run the simulation
    running = True

    while running:
        # Handle events
        if ENABLE_RENDERING:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # Counting the nr of alive agents for automatic stopping
        nr_agents = 0

        # COUNT AVG_RESOURCES of agents for DEBUGGING
        total_wood = 0.0

        # Update the agents
        for agent in agents:
            if agent.isAlive():
                agent.updateBehaviour()
                total_wood += agent.getCurrentStock('wood')
                nr_agents += 1
                agent.update_time_alive()
                x, y = agent.getPos()
                #rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                #pygame.draw.rect(screen, ORANGE, rect)

                # Do agent behaviour
                if TRADING:
                    if (
                        agent.getBehaviour() == "trade_wood"
                        or agent.getBehaviour() == "trade_food"
                    ) and SCENARIO == "Baseline":
                        traded = False
                        neighboring_cells = [
                            (dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                        ]
                        neighboring_cells.remove((0, 0))
                        while not traded and neighboring_cells:
                            dx, dy = random.choice(neighboring_cells)
                            neighboring_cells.remove((dx, dy))
                            if 0 <= x + dx < GRID_WIDTH and 0 <= y + dy < GRID_HEIGHT:
                                x_check = agent.getPos()[0] + dx
                                y_check = agent.getPos()[1] + dy
                                occupied, agent_B = cellAvailable(
                                    x_check, y_check, agents
                                )
                                if agent_B is None:
                                    continue
                                if agent.compatible(agent_B):
                                    traded_qty = agent.trade(agent_B)
                                    traded = True
                                    agent.clearBlacklistedAgents()
                                else:
                                    # If not compatible, find next nearest neighbor
                                    if AGENT_TYPE == "pathfind_neighbor":
                                        agent.removeClosestNeighbor()
                    elif (
                        agent.getBehaviour() == "trade_wood"
                        or agent.getBehaviour() == "trade_food"
                    ) and SCENARIO == "Market":
                        market_idx = np.argwhere(market)
                        if [x, y] in market_idx:
                            traded = False
                            neighboring_cells = [
                                (dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                            ]
                            neighboring_cells.remove((0, 0))

                            while not traded and bool(neighboring_cells):
                                dx, dy = random.choice(neighboring_cells)
                                neighboring_cells.remove((dx, dy))
                                if (
                                    0 <= x + dx < GRID_WIDTH
                                    and 0 <= y + dy < GRID_HEIGHT
                                    and [x + dx, y + dy] in market_idx
                                ):
                                    x_check = agent.getPos()[0] + dx
                                    y_check = agent.getPos()[1] + dy
                                    occupied, agent_B = cellAvailable(
                                        x_check, y_check, agents
                                    )
                                    if agent_B is None:
                                        continue
                                    if agent.compatible(agent_B):
                                        # print(f"TRADE at {agent.getPos()} at pos={agent_B.getPos()}")
                                        # print(f"  Agent {agent.getID()} = {agent.current_stock}, {agent.behaviour}")
                                        # print(f"  Agent {agent_B.getID()} = {agent_B.current_stock}, {agent_B.behaviour}")
                                        traded_qty = agent.trade(agent_B)
                                        agent.updateBehaviour()
                                        agent_B.updateBehaviour()
                                        # print(f"  Qty traded: {traded_qty}")
                                        # print(f"  Agent {agent.getID()} = {agent.current_stock}, {agent.behaviour}")
                                        # print(f"  Agent {agent_B.getID()} = {agent_B.current_stock}, {agent_B.behaviour}")
                                        traded = True

                            if traded:
                                agent.set_movement = "random"

                # Update the resource gathering
                chosen_resource = choose_resource(
                    agent, resources, gather_amount
                )  # make agent choose which resource to gather
                # if able_to_take_resource(agent, chosen_resource, resources):
                take_resource(agent, chosen_resource, resources, gather_amount)

                # Upkeep of agents and check if agent can survive
                agent.upkeep()

                # closest distance to market
                agent.setClosestMarketPos(findClosestMarketPos(agent, market))

                # # Choose behaviour
                # agent.updateBehaviour() # Agent brain

                # closest distance to market
                agent.setClosestMarketPos(findClosestMarketPos(agent, market))

                # Probabalistic movement
                if random.uniform(0, 1) < MOVE_PROB:
                    # Choose step
                    preferred_direction = agent.chooseStep(market)
                    moveAgent(preferred_direction, agent, agents)

                # Update market bool
                agent.setInMarket(in_market(agent, market))

                # Distance and indices of 5 nearest neighbors within view radius
                view_radius = 20
                dist, idx = positions_tree.query([[x, y]], k=5)
                for i, d in enumerate(dist[0]):
                    if d > view_radius:
                        # neighbors_too_far += 1
                        np.delete(dist, i)
                        np.delete(idx, i)

                if len(idx) > 0:
                    agent.setNearestNeighbors(agent_positions[idx][0])

                # closest distance to market
                agent.setClosestMarketPos(findClosestMarketPos(agent, market))

                # Update agent position for the KD-tree
                agent_positions[i] = [x, y]

            # Updating KD-tree
            positions_tree = KDTree(agent_positions)
            
        avg_wood = total_wood / nr_agents
        # print(f" {nr_agents=} {avg_wood=} ")

        if REGEN_ACTIVE:
            for resource in resources:
                regen_rate = (
                    food_regen_rate if resource == "food" else wood_regen_rate
                )  # get regen_rate for specific resource
                for y in range(GRID_HEIGHT):
                    for x in range(GRID_WIDTH):
                        if (
                            resources[resource][x][y]
                            < max_resources[resource][x][y] - regen_rate
                        ):
                            resources[resource][x][y] += regen_rate
                        else:
                            resources[resource][x][y] = max_resources[resource][x][
                                y
                            ]  # Set to max            

        if ENABLE_RENDERING:
            # Clear the screen
            screen.fill(WHITE)

            # Draw resources
            for row in range(GRID_HEIGHT):
                for col in range(GRID_WIDTH):
                    wood_value = wood[row][col]
                    food_value = food[row][col]
                    # Map the resource value to a shade of brown or green
                    if market[row][col]:
                        blended_color = YELLOW
                    else:
                        # Food: GREEN
                        inv_food_color = tuple(map(lambda i, j: i - j, WHITE, DARK_GREEN))
                        food_percentage = food_value / initial_food_qty_cell
                        inv_food_color = tuple(
                            map(lambda i: i * food_percentage, inv_food_color)
                        )
                        food_color = tuple(map(lambda i, j: i - j, WHITE, inv_food_color))
                        Wood: BLUE
                        inv_wood_color = tuple(map(lambda i, j: i - j, WHITE, BLUE))
                        wood_percentage = wood_value / initial_wood_qty_cell
                        inv_wood_color = tuple(
                            map(lambda i: i * wood_percentage, inv_wood_color)
                        )
                        wood_color = tuple(map(lambda i, j: i - j, WHITE, inv_wood_color))

                        # Weighted blended color
                        if food_percentage > 0.0 and food_percentage > 0.0:
                            food_ratio = food_percentage / (food_percentage + wood_percentage)
                            wood_ratio = wood_percentage / (food_percentage + wood_percentage)
                        elif food_percentage == 0.0 and wood_percentage == 0.0:
                            food_ratio = wood_ratio = 0.5
                        elif food_percentage == 0.0:
                            wood_ratio = 1.0
                            food_ratio = 0.0
                        else:
                            wood_ratio = 0.0
                            food_ratio = 1.0
                        blended_color = tuple(map(lambda f, w: f*food_ratio + w*wood_ratio, food_color, wood_color))

                    rect = pygame.Rect(row * CELL_SIZE, col * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    draw_rect_alpha(screen, blended_color, rect)

            # Draw agents
            mini_rect_size = 14
            for id, agent in enumerate(agents):
                #if alive_count < 100:
                #if id == 0:
                #    agent.print_info()
                if agent.isAlive():
                    x, y = agent.getPos()
                    if ENABLE_RENDERING:
                        rect = pygame.Rect(x * CELL_SIZE + (CELL_SIZE-mini_rect_size)/2, y * CELL_SIZE + (CELL_SIZE-mini_rect_size)/2, mini_rect_size, mini_rect_size)
                        pygame.draw.rect(screen, agent.getColor(), rect) # agent.getColor()  of  ORANGE

                    # special_condition = False
                    # # TODO: implement for debugging
                    # if special_condition:
                    #     mini_rect = pygame.Rect(x * CELL_SIZE + (CELL_SIZE-mini_rect_size)/2, y * CELL_SIZE + (CELL_SIZE-mini_rect_size)/2, mini_rect_size, mini_rect_size)
                    #     pygame.draw.rect(screen, BLACK, mini_rect)

            # Draw the grid
            for x in range(0, SCREEN_WIDTH, CELL_SIZE):
                pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
            for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
                pygame.draw.line(screen, BLACK, (0, y), (SCREEN_WIDTH, y))

            # Update the display
            pygame.display.flip()

        clock.tick(fps)
        dt = clock.tick(fps) / 100
        time += 1

        if nr_agents == 0:
            print("No agents left, ending simulation")
            running = False

        if time > 1000:
            print('Time up, ending sim')
            running = False

    # Clean up
    pygame.quit()

    if SAVE_TO_FILE:
        # Time alive of agents distribution
        alive_times = np.zeros(NUM_AGENTS)
        for agent in agents:
            alive_times[agent.id] = agent.time_alive

        # only record deaths, kaplan meier and cox model both can deal with this
        events = np.ones(len(alive_times))

        # Saving data to file
        file_path = f"outputs/{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}-{NUM_AGENTS}-{TRADING}-{MOVE_PROB}-{RUN_NR}.csv"

        if not os.path.exists(file_path):
            empty = pd.DataFrame({"ignore": [0] * time})
            empty.to_csv(file_path, index=False)

        data = pd.read_csv(file_path)

        data = pd.DataFrame(
            {
                "T": alive_times,
                "E": events,
                "Scenario": SCENARIO,
                "Agent_type": AGENT_TYPE,
                "Distribution": DISTRIBUTION,
                "Num_agents": NUM_AGENTS,
                "Trading": TRADING,
                "Move_prob": MOVE_PROB,
                "Run_number": RUN_NR,
            }
        )

        # Assign the adjusted events and alive_times to DataFrame columns
        data.to_csv(file_path, index=False)


if __name__ == "__main__":
    print("Number of cpu : ", multiprocessing.cpu_count())

    SAVE_TO_FILE = True

    distributions = ["Uniform", "Sides", "RandomGrid"]
    num_agents_list = [50, 100, 200, 300]
    move_probabilities = [0.5, 0.8, 1]
    trading = [True, False]
    scenarios = ["Baseline", "Market"]
    agent_types = ["random", "pathfind_neighbor", "pathfind_market"]


    scenarios_without_trading = "Baseline"
    agents_without_trading = "random"
    agent_types_with_trading_with_market = "pathfind_market"
    agent_types_with_trading_without_market = ["random", "pathfind_neighbor"]

    test_run = False

    processes = []

    pool = multiprocessing.Pool()

    if test_run:
        tasks = []
        for i in range(3):
            # Create the processes
            tasks.append((200,"Market",'random',0.8,"Uniform",True,False,i,))
 
        pool.map_async(runSimulation, tasks)
        pool.close()
        pool.join()

    else:
        tasks = []
        for RUN_NR in [0,2,3,4]:
            for DISTRIBUTION in distributions:
                for NUM_AGENTS in num_agents_list:
                    for MOVE_PROB in move_probabilities:
                        for TRADING in trading:
                            if not TRADING:
                                SCENARIO = scenarios_without_trading
                                AGENT_TYPE = agents_without_trading
                                tasks.append((NUM_AGENTS, SCENARIO, AGENT_TYPE, MOVE_PROB, DISTRIBUTION, TRADING, SAVE_TO_FILE, RUN_NR,))
                            else:
                                for SCENARIO in scenarios:
                                    if SCENARIO == "Market":
                                        AGENT_TYPE = agent_types_with_trading_with_market
                                        tasks.append((NUM_AGENTS, SCENARIO, AGENT_TYPE, MOVE_PROB, DISTRIBUTION, TRADING, SAVE_TO_FILE, RUN_NR,))
                                    else:
                                        for AGENT_TYPE in agent_types_with_trading_without_market:
                                            tasks.append((NUM_AGENTS, SCENARIO, AGENT_TYPE, MOVE_PROB, DISTRIBUTION, TRADING, SAVE_TO_FILE, RUN_NR,))
        # Run parallel
        pool.map_async(runSimulation, tasks)
        # Close 
        pool.close()
        pool.join()
