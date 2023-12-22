from ablation_study.ablation_study_analysis import extract_logs
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from ablation_study.ablation_study_analysis_wandb import load_runs_from_wandb_project

if __name__ == '__main__':
    # Enable Latex for plot and choose font family
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    no_memory_color = 'xkcd:cornflower blue'
    attention_memory_color = 'xkcd:coral'

    cutoff = 16  # Compute metrics from time step 15 * 25000 onwards

    # ---------------------------
    # Data loading
    # ---------------------------

    # Load some old base run results using S = 10, A = 9 and nu = 0.05
    study_path = pathlib.Path('baseline')
    rewards_base = []
    for directory in study_path.iterdir():
        if directory.is_dir():
            logs, config = extract_logs(directory)
            rewards = logs['eval/reward']  # [:20]  # should only be using the first 20
            # metric = np.abs(np.mean(rewards[cutoff:]))
            metric = np.abs(np.min(rewards[cutoff:]))
            # metric = np.linalg.norm(rewards[15:], axis=0) / len(logs['eval/reward'][15:])
            rewards_base.append(metric)

    # Load some results for base TQC from WandB
    df = load_runs_from_wandb_project("why_are_all_the_good_names_taken_aaa/KS_Ablation_Base")
    df = df[[c for c in df.columns if c[0] == 10 and c[2] == 'reward']]
    # df_metric = np.linalg.norm(df.loc[15:], axis=0) / len(df.loc[15:])
    # df_metric = np.abs(np.mean(df.loc[cutoff:], axis=0))
    df_metric = np.abs(np.min(df.loc[cutoff:], axis=0))

    # Combine the metrics for base old and new runs
    rewards_base += list(df_metric)

    # Load metrics for attention from WandB
    df = load_runs_from_wandb_project(
        "why_are_all_the_good_names_taken_aaa/KS_ablation_attention-memory_idreordered")
    df = df[[c for c in df.columns if c[0] == 10 and c[2] == 'reward']]
    # df_metric = np.abs(np.mean(df.loc[cutoff:], axis=0))
    df_metric = np.abs(np.min(df.loc[cutoff:], axis=0))
    # df_metric = np.linalg.norm(df.loc[15:], axis=0) / len(df.loc[15:])
    rewards_attention = list(df_metric)

    # ---------------------------
    # Plotting
    # ---------------------------

    # to add more plots, just pass a list of the arrays/lists to use in the boxplots
    all_data = [rewards_base, rewards_attention]
    labels = ['No memory', 'AttentionMemory']

    fig = plt.figure(figsize=(4.5, 4.5))
    plt.boxplot(
        all_data,
        widths=(.5, .5),
        medianprops=dict(color=attention_memory_color)
    )
    plt.xticks([x+1 for x in range(len(all_data))], labels=labels)
    # plt.yscale('log')
    """
    plot_title = "Performance comparison for "
    plot_title += f"nu={config['env']['nu']} | "
    plot_title += f"a={config['env']['num_actuators']} | "
    plot_title += f"s={config['env']['num_sensors']}"
    """
    plot_title = f"Maximum episodic $\displaystyle L^2$ norm after convergence"
    plt.title(plot_title)
    plt.savefig(f"boxplot.png", dpi=300)
    plt.show()
