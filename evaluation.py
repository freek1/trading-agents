import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import pandas as pd

SCENARIO = 'Market'
AGENT_TYPE = 'pathfind_market'
DISTRIBUTION = 'RandomGrid'
NUM_AGENTS = 200

data = pd.read_csv(f'outputs/{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}-{NUM_AGENTS}.csv')

amt_runs = int((len(data.columns) - 1) / 2)

# Create figures
km_graph = plt.figure()

mean_survival_probs = []

for run_nr in range(1, amt_runs + 1):
    time = len(data[data.columns[0]])
    duration = np.arange(time)
    events = data[f'events-{run_nr}']
    alive_times = data[f'alive_times-{run_nr}'][:NUM_AGENTS]

    # Compute survival probabilities for each run
    kmf = KaplanMeierFitter()
    kmf.fit(duration, event_observed=events)

    # Store survival probabilities for each timepoint
    survival_probs = kmf.survival_function_['KM_estimate'].to_numpy()
    mean_survival_probs.append(survival_probs)

    # Plot Kaplan-Meier curve for each run
    kmf.plot_survival_function(label=f'Run {run_nr}')

# Compute mean survival probability across runs
mean_survival_probs = np.mean(mean_survival_probs, axis=0)

plt.plot(mean_survival_probs, 'k', linewidth=2, label='Mean')
plt.title('Kaplan-Meier curve of mean agent deaths')
plt.ylabel('Survival probability')
plt.legend()

# Display the graph
plt.savefig(f'imgs/{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}-{NUM_AGENTS}.png')
plt.show()

####
# Old figure:

# plt.bar(np.arange(NUM_AGENTS), np.sort(alive_times))
# plt.plot(np.arange(NUM_AGENTS), np.sort(alive_times), 'k', label=f'Run {run_nr}')
# plt.xlabel('Agents')
# plt.ylabel('Time alive [timesteps]')
# plt.title('Time alive distribution of the agents')
