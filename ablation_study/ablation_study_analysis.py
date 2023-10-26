import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml
import pathlib


def extract_logs(directory):
    """
    Assuming that one run is contained in a directory, which contains the log and config files,
    this function returns the log and config file. Depends on the structure of the directory.
    """
    path = pathlib.Path(directory)
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


def final_reward(logs):
    """
    Given a logs file, extract a mean over the last 5 evaluations.
    """
    # Compute the mean of the last 5 evaluation rewards
    return np.mean(logs['eval/reward'][-5:])


if __name__ == '__main__':
    rewards = []
    num_sensors = []

    study_path = pathlib.Path('studies/study_nu005_with_frameskip')
    for directory in study_path.iterdir():
        if directory.is_dir():
            logs, config = extract_logs(directory)
            num_sensors.append(config['env']['num_sensors'])
            rewards.append(final_reward(logs))

    # Plot the ablation curve
    arr = np.array([num_sensors, rewards])
    arr_sorted = np.sort(arr.T, axis=0)

    plt.plot(arr_sorted[:,0], arr_sorted[:,1], '--gp')
    plt.xlabel('Number of sensors')
    plt.ylabel('Final evaluation reward')
    #plt.yscale('symlog')
    plt.show()

