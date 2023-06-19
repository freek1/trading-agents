import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import glob
from pathlib import Path
import seaborn as sns

kaplan_plots = False
cox_analysis = True

date_time_str = '20230619_192815'
data_path = Path(os.getcwd())

if kaplan_plots:
    kmf = KaplanMeierFitter()

    csv_files = glob.glob(os.path.join(data_path, f"outputs/{date_time_str}/*.csv"))

    if not os.path.exists(f"imgs/{date_time_str}"):
        os.makedirs(f"imgs/{date_time_str}")

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


    kmfs = {}

    # Print the grouped file paths
    for group_key, files in grouped_files.items():
        fig = plt.figure()
        # For computing the mean
        surv_func_ci = pd.DataFrame()
        amt_of_runs = len(files)
        
        mean_survival_plots = pd.DataFrame(columns=list("TE"))

        for i, file_path in enumerate(files):
            data = pd.read_csv(file_path)
            datakf = data[list('TE')]
            mean_survival_plots = pd.concat([mean_survival_plots, datakf])

        kmf = KaplanMeierFitter(label=group_key)

        kmfs[group_key] = kmf.fit(mean_survival_plots["T"], mean_survival_plots['E'])
        kmf.plot(label='Mean')

        plt.suptitle("Kaplan-Meier survival graph", fontsize=18)
        plt.title(group_key, fontsize=10)
        plt.xlabel("Time steps")
        plt.ylabel("Survival probability")
        plt.legend()

        plt.savefig(f"imgs/{date_time_str}/km-{group_key}.png")
        plt.close()


    # Effect of movement probab. 
    fig = plt.figure()
    plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
    plt.title('Effect of movement probability', fontsize=10)
    plt.xlabel("Time steps")
    plt.ylabel("Survival probability")
    kmfs['Baseline-pathfind_neighbor-RandomGrid-50-True-0.5'].plot(label='prob. = 0.5')
    kmfs['Baseline-pathfind_neighbor-RandomGrid-50-True-0.8'].plot(label='prob. = 0.8')
    kmfs['Baseline-pathfind_neighbor-RandomGrid-50-True-1'].plot(label='prob. = 1')
    plt.savefig(f"imgs/{date_time_str}/kms-comparison-mvmnt-prob.png")
    plt.close()


    # Effect of distribution
    fig = plt.figure()
    plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
    plt.title('Effect of resource distribution', fontsize=10)
    plt.xlabel("Time steps")
    plt.ylabel("Survival probability")
    kmfs['Baseline-random-Sides-200-True-1'].plot(label='Sides')
    kmfs['Baseline-random-Uniform-200-True-1'].plot(label='Uniform')
    kmfs['Baseline-random-RandomGrid-200-True-1'].plot(label='RandomGrid')
    plt.savefig(f"imgs/{date_time_str}/kms-comparison-distributions.png")
    plt.close()


    # Effect of market
    fig = plt.figure()
    plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
    plt.title('Effect of market', fontsize=10)
    plt.xlabel("Time steps")
    plt.ylabel("Survival probability")
    kmfs['Baseline-random-Sides-200-True-1'].plot(label='No market, random')
    kmfs['Baseline-pathfind_neighbor-Sides-200-True-1'].plot(label='No market, neighbor')
    kmfs['Market-pathfind_market-Sides-200-True-1'].plot(label='Market')
    plt.savefig(f"imgs/{date_time_str}/kms-comparison-market.png")
    plt.close()

    # Effect of market
    fig = plt.figure()
    plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
    plt.title('Effect of trading', fontsize=10)
    plt.xlabel("Time steps")
    plt.ylabel("Survival probability")
    kmfs['Baseline-random-Sides-50-True-1'].plot(label='Trading 50')
    kmfs['Baseline-random-Sides-50-False-1'].plot(label='No trading 50')
    kmfs['Baseline-random-Sides-100-True-1'].plot(label='Trading 100')
    kmfs['Baseline-random-Sides-100-False-1'].plot(label='No trading 100')
    kmfs['Baseline-random-Sides-200-True-1'].plot(label='Trading 200')
    kmfs['Baseline-random-Sides-200-False-1'].plot(label='No trading 200')
    kmfs['Baseline-random-Sides-300-True-1'].plot(label='Trading 300')
    kmfs['Baseline-random-Sides-300-False-1'].plot(label='No trading 300')
    plt.savefig(f"imgs/{date_time_str}/kms-comparison-trading.png")
    plt.close()

    # Test for mvmt prob weird plot
    # fig = plt.figure()
    # plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
    # plt.title('Effect of movement probability', fontsize=10)
    # plt.xlabel("Time steps")
    # plt.ylabel("Survival probability")
    # kmfs['Baseline-random-RandomGrid-50-True-0.5'].plot(label='Trading, 0.5')
    # kmfs['Baseline-random-RandomGrid-50-True-0.6'].plot(label='Trading, 0.6')
    # kmfs['Baseline-random-RandomGrid-50-True-0.7'].plot(label='Trading, 0.7')
    # kmfs['Baseline-random-RandomGrid-50-True-0.8'].plot(label='Trading, 0.8')
    # kmfs['Baseline-random-RandomGrid-50-True-0.9'].plot(label='Trading, 0.9')
    # kmfs['Baseline-random-RandomGrid-50-True-1'].plot(label='Trading 1')
    # plt.savefig(f"imgs/{date_time_str}/kms-comparison-trading-probs.png")
    # plt.show()
    # plt.close()


if cox_analysis:
    # Analysis
    def concatAllRuns(data_path: Path):
        csv_files = glob.glob(os.path.join(data_path, f"outputs/{date_time_str}/*.csv"))
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
    print("Summary dataframe:")
    cph_df = cph.summary
    print(cph_df)
    cph_df.to_csv(f"outputs/{date_time_str}_CPH-results.csv")