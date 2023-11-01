from ablation_study.ablation_study_analysis import extract_logs
import pathlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    study_path = pathlib.Path('baseline')
    rewards = []
    for directory in study_path.iterdir():
        if directory.is_dir():
            logs, config = extract_logs(directory)
            rewards.append(np.mean(logs['eval/reward'][-5:]))

    # to add more plots, just pass a list of the arrays/lists to use in the boxplots
    all_data = [rewards, rewards]
    labels = ['baseline', 'lstm']

    plt.boxplot(all_data)
    plt.xticks([x+1 for x in range(len(all_data))], labels=labels)
    plot_title = "Performance comparison for "
    plot_title += f"nu={config['env']['nu']} | "
    plot_title += f"a={config['env']['num_actuators']} | "
    plot_title += f"s={config['env']['num_sensors']}"
    plt.title(plot_title)
    plt.show()
