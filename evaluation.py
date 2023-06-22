import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import glob
from pathlib import Path

# If you want to run the cox analyses, you need to have kaplan_plots = True, as it generates the data.
kaplan_plots = True
cox_analysis_blobs = True
cox_analysis_sides = True
cox_analysis_uniform = True
cox_analysis_nr_agents_sides = True
cox_analysis_alldata = True

# Set this to the newest generated data datetime.
date_time_str = '20230622_002127'
data_path = Path(os.getcwd())

if kaplan_plots:
    kmf = KaplanMeierFitter()

    csv_files = glob.glob(os.path.join(data_path, f"outputs/{date_time_str}/*.csv"))

    if not os.path.exists(f"imgs/{date_time_str}"):
        os.makedirs(f"imgs/{date_time_str}")

    if not os.path.exists(f"outputs/{date_time_str}/results"):
        os.makedirs(f"outputs/{date_time_str}/results")

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
    cphs = {}

    # Print the grouped file paths
    for group_key, files in grouped_files.items():
        fig = plt.figure()
        # For computing the mean
        surv_func_ci = pd.DataFrame()
        amt_of_runs = len(files)
        
        mean_survival_plots = pd.DataFrame(columns=list("TE"))
        CPH_data = pd.DataFrame()

        for i, file_path in enumerate(files):
            data_for_CPH = pd.read_csv(file_path)
            
            data_for_CPH_copy = data_for_CPH.copy()
            data_for_CPH_copy.loc[data_for_CPH_copy['T'] == 1000, 'E'] = 0 # post hoc fix if the last timestep is used as time of death
            
            datakf = data_for_CPH_copy[list('TE')]
            
            mean_survival_plots = pd.concat([mean_survival_plots, datakf])
            CPH_data = pd.concat([CPH_data, data_for_CPH_copy])

        kmf = KaplanMeierFitter(label=group_key)
        cphs[group_key] = CPH_data

        kmfs[group_key] = kmf.fit(mean_survival_plots["T"], mean_survival_plots['E']) # Deze line geeft die warnings, maar kon het niet oplossen nog

    # Effect of trading RANDOM single figure
    fig = plt.figure()
    kmfs[f'Baseline-no_trade-RandomGrid-50-0.8'].plot(label='Non-trading')
    kmfs[f'Baseline-random-RandomGrid-50-0.8'].plot(label='Random-trading')
    kmfs[f'Baseline-pathfind_neighbor-RandomGrid-50-0.8'].plot(label='Search-trading')
    kmfs[f'Market-pathfind_market-RandomGrid-50-0.8'].plot(label='Market-trading')
    plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
    plt.title('Effect of trading. Random Blobs, Nr agents = 50, prob. = 0.8', fontsize=14)
    plt.xlabel("Time steps", fontsize=14)
    plt.ylabel("Survival probability", fontsize=14)
    plt.savefig(f"imgs/{date_time_str}/kms-comparison-trading-randomblobs.pdf")
    plt.close()

    # Effect of trading SIDES single figure
    fig = plt.figure()
    kmfs[f'Baseline-no_trade-Sides-50-0.8'].plot(label='Non-trading')
    kmfs[f'Baseline-random-Sides-50-0.8'].plot(label='Random-trading')
    kmfs[f'Baseline-pathfind_neighbor-Sides-50-0.8'].plot(label='Search-trading')
    kmfs[f'Market-pathfind_market-Sides-50-0.8'].plot(label='Market-trading')
    plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
    plt.title('Effect of trading. Sides, Nr agents = 50, prob. = 0.8', fontsize=14)
    plt.xlabel("Time steps", fontsize=14)
    plt.ylabel("Survival probability", fontsize=14)
    plt.savefig(f"imgs/{date_time_str}/kms-comparison-trading-sides.pdf")
    plt.close()

    # Effect of trading UNIFORM single figure
    fig = plt.figure()
    kmfs[f'Baseline-no_trade-Uniform-300-0.8'].plot(label='Non-trading')
    kmfs[f'Baseline-random-Uniform-300-0.8'].plot(label='Random-trading')
    kmfs[f'Baseline-pathfind_neighbor-Uniform-300-0.8'].plot(label='Search-trading')
    kmfs[f'Market-pathfind_market-Uniform-300-0.8'].plot(label='Market-trading')
    plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
    plt.title('Effect of trading. Uniform, Nr agents = 300, prob. = 0.8', fontsize=14)
    plt.xlabel("Time steps", fontsize=14)
    plt.ylabel("Survival probability", fontsize=14)
    plt.savefig(f"imgs/{date_time_str}/kms-comparison-trading-uniform.pdf")
    plt.close()

    # Uber figs A and B:

    # All combinations image A (for Appendix)
    fig = plt.figure(figsize=(10, 10))

    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.axis('off')

    nr_agents = [50, 100] # [50, 100, 200, 300]
    dists = ['Sides', 'Uniform', 'RandomGrid']
    probs = [0.5, 0.8, 1]
    i=0

    for nr_agent in nr_agents:
        for dist in dists:
            for prob in probs:
                i+=1

                if i == 1:
                    ax = plt.subplot(6, 3, i)
                else:
                    ax = plt.subplot(6, 3, i, sharex=ax, sharey=ax)

                if dist == 'RandomGrid':
                    dist_name = 'Random Blobs'
                else:
                    dist_name = dist
                plt.title(f'{dist_name}, nr_agents = {nr_agent}, prob. = {prob}', fontsize=10)
                ax = kmfs[f'Baseline-no_trade-{dist}-{nr_agent}-{prob}'].plot(label='Non-trading', legend=None, linewidth=1)
                ax.xaxis.set_label_text('')
                ax = kmfs[f'Baseline-random-{dist}-{nr_agent}-{prob}'].plot(label='Random-trading', legend=None, linewidth=1)
                ax.xaxis.set_label_text('')
                ax = kmfs[f'Baseline-pathfind_neighbor-{dist}-{nr_agent}-{prob}'].plot(label='Search-trading', legend=None, linewidth=1)
                ax.xaxis.set_label_text('')
                ax = kmfs[f'Market-pathfind_market-{dist}-{nr_agent}-{prob}'].plot(label='Market-trading', legend=None, linewidth=1)
                ax.xaxis.set_label_text('')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 0.94), fontsize=12)
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.9))
    plt.subplots_adjust(wspace=0.3)
    fig.text(0.5, 0.03, 'Time steps', ha='center', va='center', fontsize=14)
    fig.text(0.03,  0.5, 'Survival probability', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.suptitle("Kaplan-Meier survival graphs (A)", fontsize=20, y=0.98)
    plt.savefig(f"imgs/{date_time_str}/kms-comparison-market-uber-A.pdf")
    plt.close()


    # All combinations image B (for Appendix)
    fig = plt.figure(figsize=(10, 10))

    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.axis('off')

    nr_agents = [200, 300] # [50, 100, 200, 300]
    dists = ['Sides', 'Uniform', 'RandomGrid']
    probs = [0.5, 0.8, 1]
    i=0

    for nr_agent in nr_agents:
        for dist in dists:
            for prob in probs:
                i+=1

                if i == 1:
                    ax = plt.subplot(6, 3, i)
                else:
                    ax = plt.subplot(6, 3, i, sharex=ax, sharey=ax)
                
                if dist == 'RandomGrid':
                    dist_name = 'Random Blobs'
                else:
                    dist_name = dist
                plt.title(f'{dist_name}, nr_agents = {nr_agent}, prob. = {prob}', fontsize=10)
                ax = kmfs[f'Baseline-no_trade-{dist}-{nr_agent}-{prob}'].plot(label='Non-trading', legend=None, linewidth=1)
                ax.xaxis.set_label_text('')
                ax = kmfs[f'Baseline-random-{dist}-{nr_agent}-{prob}'].plot(label='Random-trading', legend=None, linewidth=1)
                ax.xaxis.set_label_text('')
                ax = kmfs[f'Baseline-pathfind_neighbor-{dist}-{nr_agent}-{prob}'].plot(label='Search-trading', legend=None, linewidth=1)
                ax.xaxis.set_label_text('')
                ax = kmfs[f'Market-pathfind_market-{dist}-{nr_agent}-{prob}'].plot(label='Market-trading', legend=None, linewidth=1)
                ax.xaxis.set_label_text('')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 0.94), fontsize=12)
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.9))
    plt.subplots_adjust(wspace=0.3)
    fig.text(0.5, 0.03, 'Time steps', ha='center', va='center', fontsize=14)
    fig.text(0.03,  0.5, 'Survival probability', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.suptitle("Kaplan-Meier survival graphs (B)", fontsize=20, y=0.98)
    plt.savefig(f"imgs/{date_time_str}/kms-comparison-market-uber-B.pdf")
    plt.close()


# CPH blobs
if cox_analysis_blobs:
    combined_df = pd.concat([cphs['Baseline-no_trade-RandomGrid-50-0.8'],
                             cphs['Baseline-random-RandomGrid-50-0.8'],
                             cphs['Baseline-pathfind_neighbor-RandomGrid-50-0.8'],
                             cphs['Market-pathfind_market-RandomGrid-50-0.8'],
    ])

    
    le = LabelEncoder()
    print(combined_df.keys())

    def update_trades(row):
        if row["Agent_type"] == 'no_trade':
            row["Trade_random"] = 0
            row["Trade_search"] = 0
            row["Trade_market"] = 0
        elif row["Agent_type"] == 'random':
            row["Trade_random"] = 1
            row["Trade_search"] = 0
            row["Trade_market"] = 0
        elif row["Agent_type"] == 'pathfind_neighbor':
            row["Trade_random"] = 0
            row["Trade_search"] = 1
            row["Trade_market"] = 0
        elif row["Agent_type"] == 'pathfind_market':
            row["Trade_random"] = 0
            row["Trade_search"] = 0
            row["Trade_market"] = 1
        return row

    # Assuming you have a DataFrame named combined_df
    combined_df = combined_df.apply(update_trades, axis=1)
    
    combined_df["Scenario"] = le.fit_transform(combined_df["Scenario"])
    combined_df["Trading"] = le.fit_transform(combined_df["Trading"])
    combined_df["Distribution"] = le.fit_transform(combined_df["Distribution"])
    combined_df = combined_df.drop(["Run_number", 'Num_agents', 'Distribution', 'Move_prob', 'Trading', 'Agent_type', 'Scenario'], axis=1)

    print( combined_df)
    combined_df.to_csv(f"outputs/{date_time_str}/results/CPH-trading-randomblobs-data.csv")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(combined_df, "T", "E", show_progress=False)
    cph_df = cph.summary
    cph_df.to_csv(f"outputs/{date_time_str}/results/CPH-trading-randomblobs-results.csv")

    plt.figure(figsize=(4, 3))
    cph.plot()
    plt.vlines(x=0, ymin=-1, ymax=6, color='#1f77b4', linestyles='--')
    plt.tight_layout()
    plt.savefig(f"imgs/{date_time_str}/results-trading-randomblobs.pdf")

# CPH sides
if cox_analysis_sides:
    combined_df = pd.concat([cphs[f'Baseline-no_trade-Sides-50-1'],
                             cphs[f'Baseline-random-Sides-50-1'],
                             cphs[f'Baseline-pathfind_neighbor-Sides-50-1'],
                             cphs[f'Market-pathfind_market-Sides-50-1'],
    ])

    
    le = LabelEncoder()
    print(combined_df.keys())

    def update_trades(row):
        if row["Agent_type"] == 'no_trade':
            row["Trade_random"] = 0
            row["Trade_search"] = 0
            row["Trade_market"] = 0
        elif row["Agent_type"] == 'random':
            row["Trade_random"] = 1
            row["Trade_search"] = 0
            row["Trade_market"] = 0
        elif row["Agent_type"] == 'pathfind_neighbor':
            row["Trade_random"] = 0
            row["Trade_search"] = 1
            row["Trade_market"] = 0
        elif row["Agent_type"] == 'pathfind_market':
            row["Trade_random"] = 0
            row["Trade_search"] = 0
            row["Trade_market"] = 1
        return row

    # Assuming you have a DataFrame named combined_df
    combined_df = combined_df.apply(update_trades, axis=1)
    
    combined_df["Scenario"] = le.fit_transform(combined_df["Scenario"])
    combined_df["Trading"] = le.fit_transform(combined_df["Trading"])
    combined_df["Distribution"] = le.fit_transform(combined_df["Distribution"])
    combined_df = combined_df.drop(["Run_number", 'Num_agents', 'Distribution', 'Move_prob', 'Trading', 'Agent_type', 'Scenario'], axis=1)

    print( combined_df)
    combined_df.to_csv(f"outputs/{date_time_str}/results/CPH-trading-Sides-data.csv")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(combined_df, "T", "E", show_progress=False)
    cph.print_summary()
    print("Summary dataframe:")
    cph_df = cph.summary
    print(cph_df)
    cph_df.to_csv(f"outputs/{date_time_str}/results/CPH-trading-Sides-results.csv")

    plt.figure(figsize=(4, 3))
    cph.plot()
    plt.vlines(x=0, ymin=-1, ymax=6, color='#1f77b4', linestyles='--')
    plt.tight_layout()
    plt.savefig(f"imgs/{date_time_str}/results-trading-sides.pdf")
    
# Cox analysis sides nr agents
if cox_analysis_nr_agents_sides:
    nagentsss = [50, 100, 200, 300]

    plt.figure(figsize=(16,3))

    for i, n_agents in enumerate(nagentsss):
        if i > 0:
            ax = plt.subplot(1, 4, i+1, sharey=ax)
        else:
            ax = plt.subplot(1, 4, i+1)
        combined_df = pd.concat([cphs[f'Baseline-no_trade-Sides-{n_agents}-0.8'],
                                cphs[f'Baseline-random-Sides-{n_agents}-0.8'],
                                cphs[f'Baseline-pathfind_neighbor-Sides-{n_agents}-0.8'],
                                cphs[f'Market-pathfind_market-Sides-{n_agents}-0.8'],
        ])
        
        le = LabelEncoder()
        print(combined_df.keys())

        def update_trades(row):
            if row["Agent_type"] == 'no_trade':
                row["Trade_random"] = 0
                row["Trade_search"] = 0
                row["Trade_market"] = 0
            elif row["Agent_type"] == 'random':
                row["Trade_random"] = 1
                row["Trade_search"] = 0
                row["Trade_market"] = 0
            elif row["Agent_type"] == 'pathfind_neighbor':
                row["Trade_random"] = 0
                row["Trade_search"] = 1
                row["Trade_market"] = 0
            elif row["Agent_type"] == 'pathfind_market':
                row["Trade_random"] = 0
                row["Trade_search"] = 0
                row["Trade_market"] = 1
            return row

        # Assuming you have a DataFrame named combined_df
        combined_df = combined_df.apply(update_trades, axis=1)
        
        combined_df["Scenario"] = le.fit_transform(combined_df["Scenario"])
        combined_df["Trading"] = le.fit_transform(combined_df["Trading"])
        combined_df["Distribution"] = le.fit_transform(combined_df["Distribution"])
        combined_df = combined_df.drop(["Run_number", 'Num_agents', 'Distribution', 'Move_prob', 'Trading', 'Agent_type', 'Scenario'], axis=1)

        combined_df.to_csv(f"outputs/{date_time_str}/results/CPH-trading-Sides-{n_agents}-data.csv")

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(combined_df, "T", "E", show_progress=False)
        cph_df.to_csv(f"outputs/{date_time_str}/results/CPH-trading-Sides-{n_agents}-results.csv")

        cph.plot()
        plt.title(f'Nr agents = {n_agents}')
        plt.vlines(x=0, ymin=-1, ymax=6, color='#1f77b4', linestyles='--')
        plt.tight_layout()
        plt.suptitle('Comparison of significance on Sides. prob. = 0.8')

    plt.savefig(f"imgs/{date_time_str}/results-trading-nagents-sides.pdf")


# Cox analysis UNIFORM
if cox_analysis_uniform:
    
    combined_df = pd.concat([cphs[f'Baseline-no_trade-Uniform-300-0.8'],
                             cphs[f'Baseline-random-Uniform-300-0.8'],
                             cphs[f'Baseline-pathfind_neighbor-Uniform-300-0.8'],
                             cphs[f'Market-pathfind_market-Uniform-300-0.8'],
    ])

    
    le = LabelEncoder()
    print(combined_df.keys())

    def update_trades(row):
        if row["Agent_type"] == 'no_trade':
            row["Trade_random"] = 0
            row["Trade_search"] = 0
            row["Trade_market"] = 0
        elif row["Agent_type"] == 'random':
            row["Trade_random"] = 1
            row["Trade_search"] = 0
            row["Trade_market"] = 0
        elif row["Agent_type"] == 'pathfind_neighbor':
            row["Trade_random"] = 0
            row["Trade_search"] = 1
            row["Trade_market"] = 0
        elif row["Agent_type"] == 'pathfind_market':
            row["Trade_random"] = 0
            row["Trade_search"] = 0
            row["Trade_market"] = 1
        return row

    # Assuming you have a DataFrame named combined_df
    combined_df = combined_df.apply(update_trades, axis=1)
    
    combined_df["Scenario"] = le.fit_transform(combined_df["Scenario"])
    combined_df["Trading"] = le.fit_transform(combined_df["Trading"])
    combined_df["Distribution"] = le.fit_transform(combined_df["Distribution"])
    combined_df = combined_df.drop(["Run_number", 'Num_agents', 'Distribution', 'Move_prob', 'Trading', 'Agent_type', 'Scenario'], axis=1)

    combined_df.to_csv(f"outputs/{date_time_str}/results/CPH-trading-uniform-data.csv")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(combined_df, "T", "E", show_progress=False)

    cph_df = cph.summary
    cph_df.to_csv(f"outputs/{date_time_str}/results/CPH-trading-uniform-results.csv")

    plt.figure(figsize=(4, 3))
    cph.plot()
    plt.vlines(x=0, ymin=-1, ymax=6, color='#1f77b4', linestyles='--')
    plt.tight_layout()
    plt.savefig(f"imgs/{date_time_str}/results-trading-uniform.pdf")


# Cox analysis for ALL DATA
if cox_analysis_alldata:
    # Analysis
    def concatAllRuns(data_path: Path):
        csv_files = glob.glob(os.path.join(data_path, f"outputs/{date_time_str}/*.csv"))
        combined_df = pd.concat([pd.read_csv(f) for f in csv_files])
        return combined_df
    
    combined_df = concatAllRuns(data_path)
    
    le = LabelEncoder()

    def update_trades(row):
        if row["Agent_type"] == 'no_trade':
            row["Trade_random"] = 0
            row["Trade_search"] = 0
            row["Trade_market"] = 0
        elif row["Agent_type"] == 'random':
            row["Trade_random"] = 1
            row["Trade_search"] = 0
            row["Trade_market"] = 0
        elif row["Agent_type"] == 'pathfind_neighbor':
            row["Trade_random"] = 0
            row["Trade_search"] = 1
            row["Trade_market"] = 0
        elif row["Agent_type"] == 'pathfind_market':
            row["Trade_random"] = 0
            row["Trade_search"] = 0
            row["Trade_market"] = 1
        return row

    # Assuming you have a DataFrame named combined_df
    combined_df = combined_df.apply(update_trades, axis=1)
    
    combined_df["Scenario"] = le.fit_transform(combined_df["Scenario"])
    combined_df["Trading"] = le.fit_transform(combined_df["Trading"])
    combined_df["Distribution"] = le.fit_transform(combined_df["Distribution"])
    combined_df = combined_df.drop(["Run_number", 'Trading', 'Agent_type', 'Scenario'], axis=1)

    combined_df.to_csv(f"outputs/{date_time_str}/results/CPH-trading-alldata-data.csv")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(combined_df, "T", "E", show_progress=False)
    cph_df = cph.summary

    cph_df.to_csv(f"outputs/{date_time_str}/results/CPH-trading-results-alldata.csv")

    plt.figure(figsize=(4, 3))
    cph.plot()
    plt.vlines(x=0, ymin=-1, ymax=6, color='#1f77b4', linestyles='--')
    plt.tight_layout()
    plt.savefig(f"imgs/{date_time_str}/results-trading-alldata.pdf")