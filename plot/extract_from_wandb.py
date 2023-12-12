import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_data_from_wandb_run(path):
    df = pd.read_csv(path)

    # Throw away the "MIN" and "MAX" columns
    cols = [col for col in df.columns if (col[-3:] != 'MIN' and col[-3:] != 'MAX')]
    df = df[cols]
    step_col = cols.pop(0)
    steps = df[step_col].values
    x = df[cols].values

    return steps, x


if __name__ == '__main__':
    path = '../run_logs_from_hpc/identity_reordering/simple_update_identity_reordered.csv'
    steps, data_reordered = extract_data_from_wandb_run(path)

    path = '../run_logs_from_hpc/identity_reordering/simple_update_no_reordering.csv'
    steps, data_not_reordered = extract_data_from_wandb_run(path)

    reo = data_reordered.mean(axis=1)
    reostd = data_reordered.std(axis=1)
    notreo = data_not_reordered.mean(axis=1)
    notreostd = data_not_reordered.std(axis=1)

    plt.plot(steps, reo, label='Identity reordered')
    plt.fill_between(steps, reo-1.96*reostd, reo+1.96*reostd, alpha=.2)
    plt.plot(steps, notreo, label='Not reordered')
    plt.fill_between(steps, notreo - 1.96 * notreostd, notreo + 1.96 * notreostd, alpha=.2)
    plt.legend()
    plt.savefig('../run_logs_from_hpc/identity_reordering/comparison.png')
