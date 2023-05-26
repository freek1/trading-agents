import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import pandas as pd

SCENARIO = 'Market'
AGENT_TYPE = 'pathfind_market'
DISTRIBUTION = 'RandomGrid'
NUM_AGENTS = 200

RUN_NR = 1

data = pd.read_csv(f'outputs/{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}-{NUM_AGENTS}.csv')

time = len(data['ignore'])
duration = np.arange(time)
events = data[f'events-{RUN_NR}']
alive_times = data[f'alive_times-{RUN_NR}'][:NUM_AGENTS]

# Result figures
kmf = KaplanMeierFitter()
kmf.fit(duration, events)
km_graph = plt.figure()
kmf.plot()
plt.title('Kaplan-Meier curve of agent deaths')
plt.ylabel('Survival probability')

time_alive_fig = plt.figure()
plt.bar(np.arange(NUM_AGENTS), np.sort(alive_times))
plt.plot(np.arange(NUM_AGENTS), np.sort(alive_times), 'k')
plt.xlabel('Agents')
plt.ylabel('Time alive [timesteps]')
plt.title('Time alive distribution of the agents')

# Keep images open
plt.show()