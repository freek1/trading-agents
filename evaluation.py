import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import glob
from pathlib import Path


kmf = KaplanMeierFitter()

data_path = Path(os.getcwd())
csv_files = glob.glob(os.path.join(data_path, "outputs/*.csv"))

# Group runs by experiment
grouped_files = {}
for file in csv_files:
    file_name = os.path.basename(file)
    name_without_suffix = file_name.rsplit('-', 1)[0]
    suffix = file_name.rsplit('-', 1)[1]
    group_key = name_without_suffix
    
    if group_key not in grouped_files:
        grouped_files[group_key] = []
    
    grouped_files[group_key].append(file)

# Print the grouped file paths
for group_key, files in grouped_files.items():
    fig = plt.figure()

    print(f"Group: {group_key}")
    for file_path in files:
        data = pd.read_csv(file_path)
        kmf.fit(data['T'], data['E'])
        kmf.plot_survival_function(label=f'Run {data["RUN_NUMBER"][0]}')
    plt.suptitle('Kaplan-Meier survival graph', fontsize=18)
    plt.title(group_key, fontsize=10)
    plt.xlabel('Time steps')
    plt.ylabel('Survival probability')

    fig.show()
    plt.savefig(f'imgs/km-{group_key}.png')

def concatAllRuns(data_path: Path):
    csv_files = glob.glob(os.path.join(data_path, "outputs/*.csv"))
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files])
    return combined_df

combined_df = concatAllRuns(data_path)
le = LabelEncoder()
print(combined_df.keys())
combined_df['Agent_type'] = le.fit_transform(combined_df['Agent_type'])
combined_df['Scenario'] = le.fit_transform(combined_df['Scenario'])
combined_df = combined_df.drop(['RUN_NUMBER'], axis=1)

cph = CoxPHFitter()
cph.fit(combined_df, 'T', 'E', show_progress=True)
cph.print_summary()



# data = pd.read_csv(f'outputs/{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}-{NUM_AGENTS}-{TRADING}.csv')

# amt_runs = int((len(data.columns) - 1) / 2)

# # Create Kaplan-Meier figure
# km_graph = plt.figure()

# mean_survival_probs = []

# time = len(data[data.columns[0]])
# duration = np.arange(time)
# coxphdata = pd.DataFrame({'duration': duration})

# for run_nr in range(1, amt_runs + 1):   
#     events = data[f'events-{run_nr}']
#     alive_times = data[f'alive_times-{run_nr}'][:NUM_AGENTS]

#     # Save data for cox ph analysis
#     coxphdata[f'events-{run_nr}'] = events

#     # Compute survival probabilities for each run
#     kmf = KaplanMeierFitter()
#     kmf.fit(duration, event_observed=events)

#     # Store survival probabilities for each timepoint
#     survival_probs = kmf.survival_function_['KM_estimate'].to_numpy()
#     mean_survival_probs.append(survival_probs)

#     # Plot Kaplan-Meier curve for each run
#     kmf.plot_survival_function(label=f'Run {run_nr}')

# # Compute mean survival probability across runs
# mean_survival_probs = np.mean(mean_survival_probs, axis=0)
# msp = pd.DataFrame(data={'mean_survival_probs': mean_survival_probs})
# msp.to_csv(f'eval/msp-{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}-{NUM_AGENTS}-{TRADING}.csv')

# plt.plot(mean_survival_probs, '--k', linewidth=2, label='Mean')
# plt.ylabel('Survival probability')
# plt.xlabel('Timesteps')
# plt.title(f'{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}, trading={TRADING}', fontsize=10)
# plt.suptitle(f'Kaplan-Meier curve for {NUM_AGENTS} agents', fontsize=18)
# plt.legend()

# # Display the graph
# plt.savefig(f'imgs/{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}-{NUM_AGENTS}-{TRADING}.png')
# plt.show()




# cph = CoxPHFitter(penalizer=0.1)
# cph.fit(coxphdata, 'duration', 'events-2', show_progress=True)
# cph.check_assumptions(coxphdata, show_plots=True)
# cph.print_summary()
# # TODO: Do this but for comparison between mean survival probs in different situations.

# # Comparing multiple situations
# path = os.getcwd()
# csv_files = glob.glob(os.path.join(path, "eval/*.csv"))
  
# # loop over the list of csv files
# msp_situation = []
# names = []
# for f in csv_files:
#     msp_df = pd.read_csv(f)
#     msp_situation.append(msp_df['mean_survival_probs'].to_numpy())

#     # Extract the filename from the path
#     filename = os.path.basename(f)
#     # Remove the file extension
#     filename = os.path.splitext(filename)[0]
#     # Split the filename by hyphens
#     parts = filename.split('-')
#     # Join the desired parts with hyphens
#     desired_part = '-'.join(parts[:3])
#     names.append(desired_part)

# names = ['Pathfind-neighbor', 'Trading = False', 'Trading = True']
# for i in range(len(msp_situation)):
#     plt.plot(np.arange(len(msp_situation[i])), msp_situation[i], label=f'{names[i]}')

# plt.title(f'{SCENARIO}-{AGENT_TYPE}-{DISTRIBUTION}', fontsize=10)
# plt.suptitle(f'Comparing Kaplan-Meier curves', fontsize=18)
# plt.xlabel('Timesteps')
# plt.ylabel('Survival probability')
# plt.legend()
# plt.show()

####
# Old figure:

# plt.bar(np.arange(NUM_AGENTS), np.sort(alive_times))
# plt.plot(np.arange(NUM_AGENTS), np.sort(alive_times), 'k', label=f'Run {run_nr}')
# plt.xlabel('Agents')
# plt.ylabel('Time alive [timesteps]')
# plt.title('Time alive distribution of the agents')
