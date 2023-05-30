import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

SCENARIO = 'Baseline'
AGENT_TYPE = 'pathfind_neighbor'
DISTRIBUTION = 'RandomGrid'
NUM_AGENTS = 200
TRADING = True

data = pd.read_csv(f'outputs/{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}-{NUM_AGENTS}-{TRADING}.csv')

amt_runs = int((len(data.columns) - 1) / 2)

# Create Kaplan-Meier figure
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
msp = pd.DataFrame(data={'mean_survival_probs': mean_survival_probs})
msp.to_csv(f'eval/msp-{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}-{NUM_AGENTS}-{TRADING}.csv')

plt.plot(mean_survival_probs, '--k', linewidth=2, label='Mean')
plt.ylabel('Survival probability')
plt.xlabel('Timesteps')
plt.title(f'{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}, trading={TRADING}', fontsize=10)
plt.suptitle(f'Kaplan-Meier curve for {NUM_AGENTS} agents', fontsize=18)
plt.legend()

# Display the graph
plt.savefig(f'imgs/{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}-{NUM_AGENTS}-{TRADING}.png')
plt.show()


# Comparing multiple situations
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "eval/*.csv"))
  
# loop over the list of csv files
msp_situation = []
names = []
for f in csv_files:
    msp_df = pd.read_csv(f)
    msp_situation.append(msp_df['mean_survival_probs'].to_numpy())

    # Extract the filename from the path
    filename = os.path.basename(f)
    # Remove the file extension
    filename = os.path.splitext(filename)[0]
    # Split the filename by hyphens
    parts = filename.split('-')
    # Join the desired parts with hyphens
    desired_part = '-'.join(parts[:3])
    names.append(desired_part)

names = ['Pathfind-neighbor', 'Trading = False', 'Trading = True']
for i in range(len(msp_situation)):
    plt.plot(np.arange(len(msp_situation[i])), msp_situation[i], label=f'{names[i]}')

plt.title(f'{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}', fontsize=10)
plt.suptitle(f'Comparing Kaplan-Meier curves', fontsize=18)
plt.xlabel('Timesteps')
plt.ylabel('Survival probability')
plt.legend()
plt.show()

####
# Old figure:

# plt.bar(np.arange(NUM_AGENTS), np.sort(alive_times))
# plt.plot(np.arange(NUM_AGENTS), np.sort(alive_times), 'k', label=f'Run {run_nr}')
# plt.xlabel('Agents')
# plt.ylabel('Time alive [timesteps]')
# plt.title('Time alive distribution of the agents')
