import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from ablation_study.ablation_study_analysis import extract_logs


SENSOR_EVALUATIONS = [5, 7, 10, 15, 20]


def load_runs_from_wandb_project(path):
    api = wandb.Api()
    df = pd.DataFrame()
    for run in api.runs(path=path):
        rewards = []
        last_rewards = []
        if not run.state == "finished":
            print(f"Run with ID {run.id} is not finished. Skipping this run.")
            continue
        num_sensors = eval(run.config['env'])['num_sensors']
        for i, row in run.history(keys=["eval/reward"]).iterrows():
            rewards.append(row["eval/reward"])
        for i, row in run.history(keys=["eval/last_reward_solution"]).iterrows():
            last_rewards.append(row["eval/last_reward_solution"])
        df[num_sensors, run.id, 'reward'] = rewards
        df[num_sensors, run.id, 'last_reward'] = last_rewards

    return df


def extract_summary_metrics(df, feature):
    metrics = []
    std_errors = []
    for num_sensors in SENSOR_EVALUATIONS:
        df_sens = df[[c for c in df.columns if c[0] == num_sensors and c[2] == feature]]
        # Discard the first 100k timesteps (due to random initialisation)
        means = df_sens.loc[4:].mean()
        maxs = df_sens.max()
        l2norms = np.linalg.norm(df_sens.loc[4:], axis=0) / len(df_sens.loc[4:])
        # Compute different summary metrics
        mean_reward = means.mean()
        median_reward = means.median()
        mean_max_reward = maxs.mean()
        mean_l2_reward = l2norms.mean()
        # Compute corresponding standard errors
        std_error_means = means.std() / np.sqrt(len(means))
        std_error_l2 = l2norms.std() / np.sqrt(len(means))

        # Choose which metrics to return here
        metrics.append(mean_l2_reward)
        std_errors.append(std_error_l2)

    return np.abs(np.asarray(metrics)), np.asarray(std_errors)


if __name__=='__main__':
    # Enable Latex use and update font
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    no_memory_color = 'xkcd:cornflower blue'
    attention_memory_color = 'xkcd:coral'


    df_base = load_runs_from_wandb_project("why_are_all_the_good_names_taken_aaa/KS_Ablation_Base")
    df_attention = load_runs_from_wandb_project("why_are_all_the_good_names_taken_aaa/KS_ablation_attention-memory_idreordered")

    feature = 'last_reward'  # reward or last_reward

    metrics_base, stds_base = extract_summary_metrics(df_base, feature=feature)
    metrics_attention, stds_attention = extract_summary_metrics(df_attention, feature=feature)

    print('Plotting mean/median/summary metrics...')

    fig, ax = plt.subplots()
    ax.plot(SENSOR_EVALUATIONS, metrics_base, label='No memory')
    ax.fill_between(
        SENSOR_EVALUATIONS,
        (metrics_base - 1.96*stds_base),
        (metrics_base + 1.96*stds_base),
        color='b', alpha=.1)
    ax.plot(SENSOR_EVALUATIONS, metrics_attention, label='AttentionMemory')
    ax.fill_between(
        SENSOR_EVALUATIONS,
        (metrics_attention - 1.96 * stds_attention),
        (metrics_attention + 1.96 * stds_attention),
        color='r', alpha=.1)
    ax.set_xlabel('Number of sensors')
    ax.set_ylabel('Mean episodic $\displaystyle L^2$ norm')
    ax.set_yscale('log')
    ax.set_xticks(SENSOR_EVALUATIONS, [str(x) for x in SENSOR_EVALUATIONS])
    plt.legend()
    plt.show()

    # Print time evolution of rewards
    print('Plotting min/max of evaluation as a function of timesteps')
    num_sensors = 10
    feature = 'last_reward'
    df_base_sens = df_base[[c for c in df_base.columns if c[0] == num_sensors and c[2] == feature]]
    df_attention_sens = df_attention[[c for c in df_attention.columns if c[0] == num_sensors and c[2] == feature]]

    # Add in extra columns into df_base_sens: Include old runs from HPC - only temporarily
    study_path = pathlib.Path('../comparison/baseline')
    rewards_base = []
    for i, directory in enumerate(study_path.iterdir()):
        if directory.is_dir():
            logs, config = extract_logs(directory)
            df_base_sens = df_base_sens.assign(**{f'{i}': logs['eval/reward'][:20]})

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_base_sens.abs().mean(axis=1), label='No memory', color=no_memory_color)
    ax.fill_between(
        range(len(df_base_sens.mean(axis=1))),
        df_base_sens.abs().mean(axis=1) - 1.96*df_base_sens.abs().sem(axis=1),  # df_base_sens.abs().min(axis=1),
        df_base_sens.abs().mean(axis=1) + 1.96*df_base_sens.abs().sem(axis=1),  # df_base_sens.abs().max(axis=1),
        color=no_memory_color, alpha=.2)
    ax.plot(df_attention_sens.abs().mean(axis=1), label='AttentionMemory', color=attention_memory_color)
    ax.fill_between(
        range(len(df_base_sens.mean(axis=1))),
        df_attention_sens.abs().mean(axis=1) - 1.96*df_attention_sens.abs().sem(axis=1),  # df_attention_sens.abs().min(axis=1),
        df_attention_sens.abs().mean(axis=1) + 1.96*df_attention_sens.abs().sem(axis=1),  # df_attention_sens.abs().max(axis=1),
        color=attention_memory_color, alpha=.2)
    ax.set_xticks(
        range(0, len(df_base_sens.min(axis=1)), 2),
        [f'{(x+1)*25}k' for x in range(0,len(df_base_sens.min(axis=1)),2)]
    )
    ax.set_xlabel('Solver steps')
    # ax.set_ylabel('$\displaystyle L^2$ norm')
    ax.set_yscale('log')
    plt.title(f"$\displaystyle L^2$ norm of KS solution per solver timestep")
    plt.legend()
    plt.savefig('l2norm.png', dpi=300)
    plt.show()


