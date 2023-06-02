import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import glob
from pathlib import Path
import seaborn as sns


kmf = KaplanMeierFitter()

data_path = Path(os.getcwd())
csv_files = glob.glob(os.path.join(data_path, "outputs/*.csv"))

# Group runs by experiment
grouped_files = {}
for file in csv_files:
    file_name = os.path.basename(file)
    name_without_suffix = file_name.rsplit("-", 1)[0]
    suffix = file_name.rsplit("-", 1)[1]
    group_key = name_without_suffix

    if group_key not in grouped_files:
        grouped_files[group_key] = []

    grouped_files[group_key].append(file)

mean_survival_plots = pd.DataFrame()

# Print the grouped file paths
for group_key, files in grouped_files.items():
    # fig = plt.figure()
    # For computing the mean
    surv_func_ci = pd.DataFrame()
    amt_of_runs = len(files)
    
    plt.figure()
    for i, file_path in enumerate(files):
        data = pd.read_csv(file_path)
        kmf.fit(data["T"], data["E"])
        kmf.plot_survival_function(label=f'Run {data["Run_number"][0]}')
        surv_func_ci[f"surv_func-{str(i)}"] = kmf.survival_function_
        surv_func_ci[f"ci_lower-{str(i)}"] = kmf.confidence_interval_[
            "KM_estimate_lower_0.95"
        ]
        surv_func_ci[f"ci_upper-{str(i)}"] = kmf.confidence_interval_[
            "KM_estimate_upper_0.95"
        ]

        time = max(data["T"])

    surv_func_ci = surv_func_ci.fillna(method="ffill")

    columns = ["surv_func", "ci_lower", "ci_upper"]
    means = {}
    for column in columns:
        pair_columns = [column + "-" + str(i) for i in range(amt_of_runs)]
        means[column] = surv_func_ci[pair_columns].mean(axis=1)

    # Save each situations mean survival function
    mean_survival_plots[f'{group_key}'] = means['surv_func']

    num = len(means["surv_func"])
    
    # PLOT
    sns.lineplot(means["surv_func"], errorbar=("ci", 95), label="Mean", color="black")

    plt.suptitle("Kaplan-Meier survival graph", fontsize=18)
    plt.title(group_key, fontsize=10)
    plt.xlabel("Time steps")
    plt.ylabel("Survival probability")
    plt.legend()

    plt.savefig(f"imgs/km-{group_key}.png")
    plt.close()

# Effect of movement probab. 
fig = plt.figure()
plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
plt.title('Effect of movement probability', fontsize=10)
plt.xlabel("Time steps")
plt.ylabel("Survival probability")
sns.lineplot(data=[
                    mean_survival_plots['Market-pathfind_market-Sides-200-True-0.5'],
                    mean_survival_plots['Market-pathfind_market-Sides-200-True-0.8'],
                    mean_survival_plots['Market-pathfind_market-Sides-200-True-1']
                ],
             errorbar=("ci", 95))
plt.show()


# Effect of distribution
fig = plt.figure()
plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
plt.title('Effect of resource distribution', fontsize=10)
plt.xlabel("Time steps")
plt.ylabel("Survival probability")
sns.lineplot(data=[
                    mean_survival_plots['Baseline-random-Sides-200-True-1'],
                    mean_survival_plots['Baseline-random-Uniform-200-True-1'],
                    mean_survival_plots['Baseline-random-RandomGrid-200-True-1']
                ],
             errorbar=("ci", 95))
plt.show()


# Effect of market
fig = plt.figure()
plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
plt.title('Effect of market', fontsize=10)
plt.xlabel("Time steps")
plt.ylabel("Survival probability")
sns.lineplot(data=[
                    mean_survival_plots['Baseline-random-Sides-200-True-1'],
                    mean_survival_plots['Baseline-pathfind_neighbor-Sides-200-True-1'],
                    mean_survival_plots['Market-pathfind_market-Sides-200-True-1']
                ],
             errorbar=("ci", 95))
plt.show()

# Effect of market
fig = plt.figure()
plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
plt.title('Effect of trading', fontsize=10)
plt.xlabel("Time steps")
plt.ylabel("Survival probability")
sns.lineplot(data=[
                    mean_survival_plots['Baseline-random-RandomGrid-200-True-1'],
                    mean_survival_plots['Baseline-random-RandomGrid-200-False-1'],
                ],
             errorbar=("ci", 95))
plt.show()


# Analysis
def concatAllRuns(data_path: Path):
    csv_files = glob.glob(os.path.join(data_path, "outputs/*.csv"))
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files])
    return combined_df


combined_df = concatAllRuns(data_path)
le = LabelEncoder()
print(combined_df.keys())
combined_df["Agent_type"] = le.fit_transform(combined_df["Agent_type"])
combined_df["Scenario"] = le.fit_transform(combined_df["Scenario"])
combined_df["Trading"] = le.fit_transform(combined_df["Trading"])
combined_df["Distribution"] = le.fit_transform(combined_df["Distribution"])
combined_df = combined_df.drop(["Run_number"], axis=1)

cph = CoxPHFitter(penalizer=0.1)
cph.fit(combined_df, "T", "E", show_progress=False)
cph.print_summary()
