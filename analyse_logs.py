import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml
import pathlib


#filepath = "run_logs_from_hpc/26_10_1/"
filepath = "ablation_study/studies/study_nu008_a10/13-17-45/"

path = pathlib.Path(filepath)
for item in path.rglob("*.pkl"):
    with open(item, "rb") as f:
        logs = pickle.load(f)
print('Loaded logs.')

for item in path.rglob("*config.yaml"):
    with open(item, "rb") as f:
        config = yaml.safe_load(f)
print('Loaded config.')

print('here')

for key, vals in logs.items():
    logs[key] = np.array(vals)

for key, vals in logs.items():
    print(f'{key} : {vals.shape}')

fig, axs = plt.subplots(2, 3)

x_tick_step_size = 200

# Plot evaluation reward
axs[0, 0].plot(len(logs['train/reward']) * np.linspace(0, 1, len(logs['eval/reward'])), logs['eval/reward'])
axs[0, 0].set_title('Eval reward (sum) - ' + f"mean (last 10 evals): {np.mean(logs['eval/reward'][-10:]):.5f}")
axs[0, 0].set_xticks(range(0, len(logs['train/reward']) + 1, x_tick_step_size),
                     [f'{x}k' for x in range(0, len(logs['train/reward']) + 1, 200)])
# axs[0,0].set(xlabel='Steps')

# Plot last reward (eval) of solution
axs[1, 0].plot(len(logs['train/reward']) * np.linspace(0, 1, len(logs['eval/last_reward'])),
               logs['eval/last_reward_solution'])
axs[1, 0].set_title(
    'Last eval reward (solution) - ' + f"mean (last 10 evals): {np.mean(logs['eval/last_reward'][-10:]):.5f}")
axs[1, 0].set_xticks(range(0, len(logs['train/reward']) + 1, x_tick_step_size),
                     [f'{x}k' for x in range(0, len(logs['train/reward']) + 1, 200)])
axs[1, 0].set(xlabel='Steps')

# Plot total loss
axs[0, 1].plot(logs['train/q_loss'] + logs['train/actor_loss'] + logs['train/alpha_loss'], label='Total loss (sum)')
axs[0, 1].plot(logs['train/q_loss'], label='Critic loss')
axs[0, 1].plot(logs['train/actor_loss'], label='Actor loss')
axs[0, 1].plot(logs['train/alpha_loss'], label='Alpha loss')
axs[0, 1].set_xticks(range(0, len(logs['train/reward']) + 1, x_tick_step_size),
                     [f'{x}k' for x in range(0, len(logs['train/reward']) + 1, 200)])
axs[0, 1].legend()
axs[0, 1].set_title('Training loss')
# axs[0,1].set(xlabel='Steps')

# Plot training reward
axs[1, 1].plot(logs['train/episode_length'])
axs[1, 1].set_title('Episode length (train time)')
axs[1, 1].set_xticks(range(0, len(logs['train/reward']) + 1, x_tick_step_size),
                     [f'{x}k' for x in range(0, len(logs['train/reward']) + 1, 200)])
axs[1, 1].set(xlabel='Steps')

# Plot alpha
axs[0, 2].semilogy(logs['train/alpha'])
axs[0, 2].set_title('Alpha (log scale on y axis)')
axs[0, 2].set_xticks(range(0, len(logs['train/reward']) + 1, x_tick_step_size),
                     [f'{x}k' for x in range(0, len(logs['train/reward']) + 1, 200)])
# axs[0,2].set(xlabel='Steps')

# Plot entropy
# axs[1,2].plot(logs['train/entropy'])
# axs[1,2].set_title('Entropy')
# axs[1,2].set_xticks(range(0,len(logs['train/reward'])+1,x_tick_step_size), [f'{x}k' for x in range(0,len(logs['train/reward'])+1,200)])
# axs[1,2].set(xlabel='Steps')

# Plot mean actuation
axs[1, 2].plot(len(logs['train/reward']) * np.linspace(0, 1, len(logs['eval/mean_actuation'])),
               logs['eval/mean_actuation'])
axs[1, 2].set_title('Mean absolute actuation (eval)')
axs[1, 2].set_xticks(range(0, len(logs['train/reward']) + 1, x_tick_step_size),
                     [f'{x}k' for x in range(0, len(logs['train/reward']) + 1, 200)])
axs[1, 2].set(xlabel='Steps')

title = 'HPC run: '
title += f"nu={config['env']['nu']} | "
title += f"actuators={config['env']['num_actuators']} | "
title += f"sensors={config['env']['num_sensors']} | "
title += f"burnin={config['env']['burnin']} | "
title += f"frameskip={config['env'].get('frame_skip', 1)} | "
title += f"softaction={config['env'].get('soft_action', False)} | "
title += f"action_loss={config['optim'].get('actuator_loss_weight', 0.0)} | "
title += f"autoreg_action={config['env'].get('autoreg_action', 0.0)}"

fig.suptitle(title)
fig.set_size_inches(15, 10)

# plt.show()
plt.savefig(filepath + 'plots.png', dpi=300, format='png')

