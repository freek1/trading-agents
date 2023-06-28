import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import random
import numpy as np
import seaborn as sns
import copy
import pandas as pd
import os
from datetime import datetime
import multiprocessing
import traceback

# functions file
from funcs import *

# agent class
from agent import Agent

def run_simulation(arg):
    try:
        # unpacking input arguments
        print(arg)
        NUM_AGENTS, SCENARIO, AGENT_TYPE, MOVE_PROB, DISTRIBUTION, TRADING, SAVE_TO_FILE, RUN_NR, run_time, enable_rendering = arg

        # set the dimensions of the screen
        g_ri_d_w_id_th, g_ri_d_h_ei_gh_t, c_el_l_s_iz_e = get_grid_params()
        s_cr_ee_n_w_id_th = g_ri_d_w_id_th * c_el_l_s_iz_e
        s_cr_ee_n_h_ei_gh_t = g_ri_d_h_ei_gh_t * c_el_l_s_iz_e

        if enable_rendering:
            # initialize pygame
            pygame.init()
            screen = pygame.display.set_mode((s_cr_ee_n_w_id_th, s_cr_ee_n_h_ei_gh_t))

        fps = 20
        clock = pygame.time.Clock()
        time = 1
        d_ur_at_io_n = 1000

        # define some colors
        b_la_ck = (0, 0, 0)
        w_hi_te = (255, 255, 255)
        b_ro_wn = (153, 102, 0)
        y_el_lo_w = (255, 255, 0)
        d_ar_k_g_re_en = (0, 200, 0)
        b_lu_e = (30, 70, 250)
        o_ra_ng_e = (240, 70, 0)

        # t_od_o: check if we never want to chance these variables, otherwise we need to take them out of the function (i dont think we ever want no regen or chance the maximum resources over runs though)
        r_eg_en_a_ct_iv_e = True
        m_ax_w_oo_d = 2
        m_ax_f_oo_d = 2
        # grid distribution parameters
        b_lo_b_s_iz_e = 3

        # place and size of market
        m_ar_ke_t_p_la_ce = "middle"
        m_ar_ke_t_s_iz_e = 6

        market = np.full((g_ri_d_h_ei_gh_t, g_ri_d_w_id_th), False, dtype=bool)
        if SCENARIO == "market":
            if m_ar_ke_t_p_la_ce == "middle":
                for x in range(
                    int((g_ri_d_w_id_th / 2) - m_ar_ke_t_s_iz_e), int((g_ri_d_w_id_th / 2) + m_ar_ke_t_s_iz_e)
                ):
                    for y in range(
                        int((g_ri_d_h_ei_gh_t / 2) - m_ar_ke_t_s_iz_e),
                        int((g_ri_d_h_ei_gh_t / 2) + m_ar_ke_t_s_iz_e),
                    ):
                        market[x][y] = True

        wood = np.zeros((g_ri_d_h_ei_gh_t, g_ri_d_w_id_th))
        food = np.zeros((g_ri_d_h_ei_gh_t, g_ri_d_w_id_th))

        # number of resource cells
        wood_cell_count = 0
        food_cell_count = 0

        if DISTRIBUTION == "sides":
            # resources in non-random positions
            for x in range(0, g_ri_d_w_id_th):
                for y in range(0, 8):
                    if not market[x][y]:
                        wood_cell_count += 1
                        wood[x][y] = m_ax_w_oo_d  # random.uniform(m_in_w_oo_d, m_ax_w_oo_d)
            for x in range(0, g_ri_d_w_id_th):
                for y in range(32, g_ri_d_h_ei_gh_t):
                    if not market[x][y]:
                        food_cell_count += 1
                        food[x][y] = m_ax_f_oo_d  # random.uniform(m_in_f_oo_d, m_ax_f_oo_d)

        elif DISTRIBUTION == "uniform":
            # t_od_o: uniform distibution, but low resources such that it supports a certain number of agents
            for x in range(0, g_ri_d_w_id_th):
                for y in range(0, g_ri_d_h_ei_gh_t):
                    if not market[x][y]:
                        wood_cell_count += 1
                        food_cell_count += 1
                        wood[x][y] = m_ax_w_oo_d  # random.uniform(m_in_w_oo_d, m_ax_w_oo_d)
                        food[x][y] = m_ax_f_oo_d  # random.uniform(m_in_f_oo_d, m_ax_f_oo_d)

        elif DISTRIBUTION == "random_grid":
            random_array = np.random.rand(20, 20)
            blob_types = np.where(random_array < 0.5, 0, 1)
            for x in range(0, g_ri_d_w_id_th):
                for y in range(0, g_ri_d_h_ei_gh_t):
                    if (
                        not market[x][y]
                        and int(x / b_lo_b_s_iz_e) % 2 == 0
                        and int(y / b_lo_b_s_iz_e) % 2 == 0
                    ):
                        x_blob_index = int(x / b_lo_b_s_iz_e)
                        y_blob_index = int(y / b_lo_b_s_iz_e)
                        #if random.random() > 0.5:
                        if blob_types[x_blob_index, y_blob_index] == 0:
                            wood_cell_count += 1
                            wood[x][y] = m_ax_w_oo_d  # random.uniform(m_in_w_oo_d, m_ax_w_oo_d)
                        else:
                            food_cell_count += 1
                            food[x][y] = m_ax_f_oo_d  # random.uniform(m_in_f_oo_d, m_ax_f_oo_d)

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
        for x in range(0, g_ri_d_w_id_th):
            for y in range(0, g_ri_d_h_ei_gh_t):
                if food[x][y] > 0:
                    food[x][y] = initial_food_qty_cell
                if wood[x][y] > 0:
                    wood[x][y] = initial_wood_qty_cell

        resources = {
            "wood": wood,
            "food": food,
        }
        max_resources = copy.deepcopy(resources)

        # set up the agents
        agents = []
        agent_colours = sns.color_palette("bright", n_colors=NUM_AGENTS)

        gather_amount = 1.0

        agent_positions = []

        # creating agents
        for i in range(NUM_AGENTS):
            x = random.randint(0, g_ri_d_w_id_th - 2)
            y = random.randint(0, g_ri_d_h_ei_gh_t - 2)
            #color = (255.0, 0.0, 0.0) if y < g_ri_d_h_ei_gh_t / 2 else (0.0, 255.0, 0.0)
            color = (255,110,0)
            agent = Agent(
                i, x, y, AGENT_TYPE, color, market
            )  # color = np.array(agent_colours[i])*255
            
            agents.append(agent)

            # save agent position for the k_d-tree
            agent_positions.append([x, y])
        # initialize KDTree
        positions_tree = KDTree(agent_positions)
                
        alive_times = np.zeros([NUM_AGENTS])
        alive_times.fill(d_ur_at_io_n)

        # run the simulation
        running = True

        while running:
            # handle events
            if enable_rendering:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

            # counting the nr of alive agents for automatic stopping
            nr_agents = 0

            # c_ou_nt a_vg_r_es_ou_rc_es of agents for d_eb_ug_gi_ng
            total_wood = 0.0
            
            # update the agents
            for i, agent in enumerate(agents):
                if agent.is_alive():   
                    agent.update_behaviour(positions_tree, agent_positions, agents, 6, 20)
                    total_wood += agent.get_current_stock('wood')
                    nr_agents += 1
                    agent.update_time_alive()
                    x, y = agent.get_pos()
                    #rect = pygame.rect(x * c_el_l_s_iz_e, y * c_el_l_s_iz_e, c_el_l_s_iz_e, c_el_l_s_iz_e)
                    #pygame.draw.rect(screen, o_ra_ng_e, rect)

                    # do agent behaviour
                    if TRADING:
                        if (
                            agent.get_behaviour() == "trade_wood"
                            or agent.get_behaviour() == "trade_food"
                        ) and SCENARIO == "baseline":
                            traded = False
                            neighboring_cells = [
                                (dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                            ]
                            neighboring_cells.remove((0, 0))
                            while not traded and neighboring_cells:
                                dx, dy = random.choice(neighboring_cells)
                                neighboring_cells.remove((dx, dy))
                                if 0 <= x + dx < g_ri_d_w_id_th and 0 <= y + dy < g_ri_d_h_ei_gh_t:
                                    x_check = agent.get_pos()[0] + dx
                                    y_check = agent.get_pos()[1] + dy
                                    occupied, agent_b = cell_available(
                                        x_check, y_check, agents
                                    )
                                    if agent_b is None:
                                        continue
                                    if agent.compatible(agent_b):
                                        trade_qty = agent.trade(agent_b)
                                        traded = True
                                    # if not compatible, find next nearest neighbor
                                    elif AGENT_TYPE == "pathfind_neighbor" and len(agent.nearest_neighbors)>0:
                                        if agent.get_nearest_neigbors()[0].get_pos() == agent_b.get_pos():
                                            agent.remove_closest_neighbor()

                        elif (
                            agent.get_behaviour() == "trade_wood"
                            or agent.get_behaviour() == "trade_food"
                        ) and SCENARIO == "market":
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
                                    x_check = agent.get_pos()[0] + dx
                                    y_check = agent.get_pos()[1] + dy
                                    occupied, agent_b = cell_available(
                                        x_check, y_check, agents
                                    )
                                    if agent_b is None:
                                        continue
                                    if agent.compatible(agent_b):
                                        trade_qty = agent.trade(agent_b)
                                        traded = True
                                    
                    if len(agent.nearest_neighbors)==0 and agent.treshold_new_neighbours>0:
                        agent.update_treshold_new_neighbours()

                    # update the resource gathering
                    chosen_resource = choose_resource(
                        agent, resources, gather_amount
                    )  # make agent choose which resource to gather
                    # if able_to_take_resource(agent, chosen_resource, resources):
                    take_resource(agent, chosen_resource, resources, gather_amount)

                    # closest distance to market
                    agent.set_closest_market_pos(find_closest_market_pos(agent, market))

                    # choose behaviour
                    agent.update_behaviour(positions_tree, agent_positions, agents, 6, 20) # to prevent two trades in same timestep (initialized by other agent)

                    # closest distance to market
                    agent.set_closest_market_pos(find_closest_market_pos(agent, market))

                    # probabalistic movement
                    if random.uniform(0, 1) < MOVE_PROB:
                        # choose step
                        preferred_direction = agent.choose_step(agent_positions)
                        move_agent(preferred_direction, agent, agents)

                    # update market bool
                    agent.set_in_market(in_market(agent, market))

                    # closest distance to market
                    agent.set_closest_market_pos(find_closest_market_pos(agent, market))
                    
                    agent_positions[i] = agent.get_pos()
            
            death_agents = []
            # upkeep of agents and check if agent can survive
            for agent in agents:
                agent.upkeep()
                
                # if agent died, then remove from list and save death time
                if not agent.is_alive():
                    death_agents.append(agent)
                    alive_times[agent.id] = time

            for death_agent in death_agents:
                agents.remove(death_agent)
                agent_positions.remove(death_agent.get_pos())
            
            if len(agent_positions)>0:
                # updating k_d-tree
                positions_tree = KDTree(agent_positions)  

            # if nr_agents:
            #     avg_wood = total_wood / nr_agents
            # else: avg_wood = .0
            # print(f" {nr_agents=} {avg_wood=} ")

            if r_eg_en_a_ct_iv_e:
                for resource in resources:
                    regen_rate = (
                        food_regen_rate if resource == "food" else wood_regen_rate
                    )  # get regen_rate for specific resource
                    for y in range(g_ri_d_h_ei_gh_t):
                        for x in range(g_ri_d_w_id_th):
                            if (
                                resources[resource][x][y]
                                < max_resources[resource][x][y] - regen_rate
                            ):
                                resources[resource][x][y] += regen_rate
                            else:
                                resources[resource][x][y] = max_resources[resource][x][y]  # set to max            
            
            if enable_rendering:
                # clear the screen
                screen.fill(w_hi_te)

                # draw resources
                for row in range(g_ri_d_h_ei_gh_t):
                    for col in range(g_ri_d_w_id_th):
                        wood_value = wood[row][col]
                        food_value = food[row][col]
                        # map the resource value to a shade of brown or green
                        if market[row][col]:
                            blended_color = y_el_lo_w
                        else:
                            # food: g_re_en
                            inv_food_color = tuple(map(lambda i, j: i - j, w_hi_te, d_ar_k_g_re_en))
                            food_percentage = food_value / initial_food_qty_cell
                            inv_food_color = tuple(
                                map(lambda i: i * food_percentage, inv_food_color)
                            )
                            food_color = tuple(map(lambda i, j: i - j, w_hi_te, inv_food_color))
                            wood: b_lu_e
                            inv_wood_color = tuple(map(lambda i, j: i - j, w_hi_te, b_lu_e))
                            wood_percentage = wood_value / initial_wood_qty_cell
                            inv_wood_color = tuple(
                                map(lambda i: i * wood_percentage, inv_wood_color)
                            )
                            wood_color = tuple(map(lambda i, j: i - j, w_hi_te, inv_wood_color))

                            # weighted blended color
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

                        rect = pygame.Rect(row * c_el_l_s_iz_e, col * c_el_l_s_iz_e, c_el_l_s_iz_e, c_el_l_s_iz_e)
                        draw_rect_alpha(screen, blended_color, rect)

                # draw agents
                mini_rect_size = 14
                for id, agent in enumerate(agents):
                    #if alive_count < 100:
                    #if id == 0:
                    #    agent.print_info()
                    if agent.is_alive():
                        x, y = agent.get_pos()
                        if enable_rendering:
                            rect = pygame.Rect(x * c_el_l_s_iz_e + (c_el_l_s_iz_e-mini_rect_size)/2, y * c_el_l_s_iz_e + (c_el_l_s_iz_e-mini_rect_size)/2, mini_rect_size, mini_rect_size)
                            pygame.draw.rect(screen, agent.get_color(), rect) # agent.get_color()  of  o_ra_ng_e

                        # special_condition = False
                        # # t_od_o: implement for debugging
                        # if special_condition:
                        #     mini_rect = pygame.rect(x * c_el_l_s_iz_e + (c_el_l_s_iz_e-mini_rect_size)/2, y * c_el_l_s_iz_e + (c_el_l_s_iz_e-mini_rect_size)/2, mini_rect_size, mini_rect_size)
                        #     pygame.draw.rect(screen, b_la_ck, mini_rect)

                # draw the grid
                for x in range(0, s_cr_ee_n_w_id_th, c_el_l_s_iz_e):
                    pygame.draw.line(screen, b_la_ck, (x, 0), (x, s_cr_ee_n_h_ei_gh_t))
                for y in range(0, s_cr_ee_n_h_ei_gh_t, c_el_l_s_iz_e):
                    pygame.draw.line(screen, b_la_ck, (0, y), (s_cr_ee_n_w_id_th, y))

                # update the display
                pygame.display.flip()

            clock.tick(fps)
            dt = clock.tick(fps) / 100
            time += 1

            if nr_agents == 0:
                print("no agents left, ending simulation")
                running = False

            if time > d_ur_at_io_n:
                print('time up, ending sim')
                running = False
        #print("l_in_e 422")
        
        if SAVE_TO_FILE:
            print(" save results...")
            # list of when agents died
            events = np.zeros([NUM_AGENTS])
            for i, ev in enumerate(alive_times):
                # if agent died before the final timestep (otherwise it was still alive at the end)
                ev = int(ev)
                if ev < d_ur_at_io_n and ev > 0:
                    events[i] = 1
                else:
                    events[i] = 0

            if AGENT_TYPE == 'random' and TRADING == False:
                # saving data to file
                file_path = f"outputs/{run_time}/{SCENARIO}-{'no_trade'}-{DISTRIBUTION}-{NUM_AGENTS}-{MOVE_PROB}-{RUN_NR}.csv"
                if not os.path.exists(file_path):
                    empty = pd.data_frame({"ignore": [0] * time})
                    empty.to_csv(file_path, index=False)

                data = pd.read_csv(file_path)

                data = pd.data_frame(
                    {
                        "t": alive_times,
                        "e": events,
                        "scenario": SCENARIO,
                        "agent_type": 'no_trade',
                        "distribution": DISTRIBUTION,
                        "num_agents": NUM_AGENTS,
                        "trading": TRADING,
                        "move_prob": MOVE_PROB,
                        "run_number": RUN_NR,
                    }
                )
                # assign the adjusted events and alive_times to data_frame columns
                try:
                    data.to_csv(file_path, index=False)
                except Exception:
                    traceback.print_exc()
            else:
                file_path = f"outputs/{run_time}/{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}-{NUM_AGENTS}-{MOVE_PROB}-{RUN_NR}.csv"
                if not os.path.exists(file_path):
                    empty = pd.data_frame({"ignore": [0] * time})
                    empty.to_csv(file_path, index=False)

                data = pd.read_csv(file_path)

                data = pd.data_frame(
                    {
                        "t": alive_times,   
                        "e": events,
                        "scenario": SCENARIO,
                        "agent_type": AGENT_TYPE,
                        "distribution": DISTRIBUTION,
                        "num_agents": NUM_AGENTS,
                        "trading": TRADING,
                        "move_prob": MOVE_PROB,
                        "run_number": RUN_NR,
                    }
                )
                # assign the adjusted events and alive_times to data_frame columns
                try:
                    data.to_csv(file_path, index=False)
                except Exception:
                    traceback.print_exc()
            


        pygame.quit()
        
    except Exception as e:
        traceback.print_exc()
        
if __name__ == "__main__":
    run_time_str = datetime.now().strftime("%Y%m%d%H%M%S") # current date and time

    print("CPUs available: ", multiprocessing.cpu_count())

    distributions = ['uniform', 'sides', 'random_grid'] #["uniform", "sides", "random_grid"]
    num_agents_list = [50, 100, 200, 300]
    move_probabilities = [0.5, 0.8, 1]
    trading = [True, False]
    scenarios = ["baseline", "market"]
    agent_types = ["random", "pathfind_neighbor", "pathfind_market"]


    scenarios_without_trading = "baseline"
    agents_without_trading = "random"
    agent_types_with_trading_with_market = "pathfind_market"
    agent_types_with_trading_without_market = ["random", "pathfind_neighbor"]

    processes = []

    pool = multiprocessing.Pool()

    # ! Test run
    test_run = 1

    if test_run:
        enable_rendering = 1
        SAVE_TO_FILE = 0
        if not os.path.exists(f"outputs/{run_time_str}") and SAVE_TO_FILE:
            os.makedirs(f"outputs/{run_time_str}")
        
        tasks = []
        for i in range(1):
            tasks.append((1, 'baseline', 'neural' , 1, 'random_grid', True, SAVE_TO_FILE, i, run_time_str, enable_rendering))
        pool.map_async(run_simulation, tasks)
        pool.close()
        pool.join()

    else:
        SAVE_TO_FILE = 1
        if not os.path.exists(f"outputs/{run_time_str}") and SAVE_TO_FILE:
            os.makedirs(f"outputs/{run_time_str}")
        tasks = []
        for RUN_NR in [1, 2, 3, 4, 5]:
            for DISTRIBUTION in distributions:
                for NUM_AGENTS in num_agents_list:
                    for MOVE_PROB in move_probabilities:
                        for TRADING in trading:
                            if not TRADING:
                                SCENARIO = scenarios_without_trading
                                AGENT_TYPE = agents_without_trading
                                tasks.append((NUM_AGENTS, SCENARIO, AGENT_TYPE, MOVE_PROB, DISTRIBUTION, TRADING, SAVE_TO_FILE, RUN_NR, run_time_str, False))
                            else:
                                for SCENARIO in scenarios:
                                    if SCENARIO == "market":
                                        AGENT_TYPE = agent_types_with_trading_with_market
                                        tasks.append((NUM_AGENTS, SCENARIO, AGENT_TYPE, MOVE_PROB, DISTRIBUTION, TRADING, SAVE_TO_FILE, RUN_NR, run_time_str, False))
                                    else:
                                        for AGENT_TYPE in agent_types_with_trading_without_market:
                                            tasks.append((NUM_AGENTS, SCENARIO, AGENT_TYPE, MOVE_PROB, DISTRIBUTION, TRADING, SAVE_TO_FILE, RUN_NR, run_time_str, False))
        # run parallel
        pool.map_async(run_simulation, tasks)
        # close 
        pool.close()
        pool.join()
