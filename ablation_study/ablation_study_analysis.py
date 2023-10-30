import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml
import pathlib


def extract_logs(run_path):
    """
    Assuming that one run is contained in a directory, which contains the log and config files,
    this function returns the log and config file. Depends on the structure of the directory.
    """
    path = pathlib.Path(run_path)
    # Extract the logs
    for item in path.rglob("*.pkl"):
        with open(item, "rb") as f:
            logs = pickle.load(f)
    # Extract the config
    for item in path.rglob("*config.yaml"):
        with open(item, "rb") as f:
            config = yaml.safe_load(f)
    # Convert logs into numpy arrays
    for key, vals in logs.items():
        logs[key] = np.array(vals)

    return logs, config


def study_rewards(study_path):
    rewards = []
    num_sensors = []
    config = None
    for directory in study_path.iterdir():
        if directory.is_dir():
            logs, config = extract_logs(directory)
            num_sensors.append(config['env']['num_sensors'])
            rewards.append(np.mean(logs['eval/reward'][-5:]))

    print('Study generated in ' + str(study_path))

    arr = np.array([num_sensors, rewards])
    arr_sorted = np.sort(arr.T, axis=0)

    return arr_sorted, config


if __name__ == '__main__':
    sup_study_path = pathlib.Path('studies/comparing_nu_actuators')
    fig = plt.figure()
    ax = plt.subplot(111)
    for directory in sup_study_path.iterdir():
        if directory.is_dir():
            arr_sorted, config = study_rewards(directory)

            label = f"nu={config['env']['nu']} | a={config['env']['num_actuators']}"

            ax.plot(arr_sorted[:,0], arr_sorted[:,1], '--p', label=label)

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5, fontsize=5)

    ax.set_title('Comparison of different ablation studies')
    plt.xlabel('Number of sensors')
    plt.ylabel('Final evaluation reward')
    #plt.legend()
    #plt.yscale('symlog')

    # plt.show()
    plt.savefig(str(sup_study_path) + '/plot.png', dpi=300, format='png')

