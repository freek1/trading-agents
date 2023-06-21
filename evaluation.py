import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import glob
from pathlib import Path
import seaborn as sns

kaplan_plots = True
cox_analysis = False

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
            
            datakf_copy = datakf.copy()
            datakf_copy.loc[datakf_copy['T'] == 1000, 'E'] = 0 # post hoc fix if accidentally the last timestep is used as time of death
            
            mean_survival_plots = pd.concat([mean_survival_plots, datakf_copy])
        
        kmf = KaplanMeierFitter(label=group_key)

        kmfs[group_key] = kmf.fit(mean_survival_plots["T"], mean_survival_plots['E']) # Deze line geeft die warnings, maar kon het niet oplossen nog
        kmf.plot(label='Mean')

        plt.suptitle("Kaplan-Meier survival graph", fontsize=18)
        plt.title(group_key, fontsize=10)
        plt.xlabel("Time steps")
        plt.ylabel("Survival probability")
        plt.legend()

        plt.savefig(f"imgs/{date_time_str}/km-{group_key}.pdf")
        plt.close()


    # Effect of movement probab. 
    fig = plt.figure()
    kmfs['Baseline-pathfind_neighbor-RandomGrid-50-0.5'].plot(label='prob. = 0.5')
    kmfs['Baseline-pathfind_neighbor-RandomGrid-50-0.8'].plot(label='prob. = 0.8')
    kmfs['Baseline-pathfind_neighbor-RandomGrid-50-1'].plot(label='prob. = 1')
    plt.suptitle("Mean Kaplan-Meier survival graphs", fontsize=18)
    plt.title('Effect of movement probability', fontsize=10)
    plt.xlabel("Time steps")
    plt.ylabel("Survival probability")
    plt.savefig(f"imgs/{date_time_str}/kms-comparison-mvmnt-prob.png")
    plt.close()


    # Effect of distribution
    fig = plt.figure()
    kmfs['Baseline-random-Sides-200-1'].plot(label='Sides')
    kmfs['Baseline-random-Uniform-200-1'].plot(label='Uniform')
    kmfs['Baseline-random-RandomGrid-200-1'].plot(label='RandomGrid')
    plt.savefig(f"imgs/{date_time_str}/kms-comparison-distributions.png")
    plt.close()


    # All combinations image (for Appendix)
    fig = plt.figure(figsize=(10, 20))

    legend_ax = fig.add_subplot(111, frameon=False)
    legend_ax.axis('off')

    nr_agents = [50, 100, 200, 300]
    dists = ['Sides', 'Uniform', 'RandomGrid']
    probs = [0.5, 0.8, 1]
    i=0

    for nr_agent in nr_agents:
        for dist in dists:
            for prob in probs:
                i+=1

                if i == 1:
                    ax = plt.subplot(12, 3, i)
                else:
                    ax = plt.subplot(12, 3, i, sharex=ax, sharey=ax)

                plt.title(f'{dist}, nr_agents = {nr_agent}, prob. = {prob}', fontsize=10)
                ax = kmfs[f'Baseline-no_trade-{dist}-{nr_agent}-{prob}'].plot(label='No market, no trading', legend=None, linewidth=1)
                ax.xaxis.set_label_text('')
                ax = kmfs[f'Baseline-random-{dist}-{nr_agent}-{prob}'].plot(label='No market, random', legend=None, linewidth=1)
                ax.xaxis.set_label_text('')
                ax = kmfs[f'Baseline-pathfind_neighbor-{dist}-{nr_agent}-{prob}'].plot(label='No market, neighbor', legend=None, linewidth=1)
                ax.xaxis.set_label_text('')
                ax = kmfs[f'Market-pathfind_market-{dist}-{nr_agent}-{prob}'].plot(label='Market', legend=None, linewidth=1)
                ax.xaxis.set_label_text('')
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 0.965), fontsize=12)
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.95))
    plt.subplots_adjust(wspace=0.3)
    fig.text(0.5, 0.03, 'Time steps', ha='center', va='center', fontsize=14)
    fig.text(0.03,  0.5, 'Survival probability', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.suptitle("Kaplan-Meier survival graphs (B)", fontsize=20, y=0.98)
    plt.savefig(f"imgs/{date_time_str}/kms-comparison-market-uber.pdf")
    plt.close()

    
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

    plt.figure()
    cph.plot()
    plt.tight_layout()
    plt.show()
