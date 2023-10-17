import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("test run logs/16th october/logs.pkl", "rb") as f:
    logs = pickle.load(f)
print('Loaded logs.')


print('here')

for key,vals in logs.items():
    logs[key] = np.array(vals)

for key,vals in logs.items():
    print(f'{key} : {vals.shape}')

# Plot total loss
plt.plot(logs['train/q_loss'] + logs['train/actor_loss'] + logs['train/alpha_loss'], label='Total loss (sum)')
plt.plot(logs['train/q_loss'], label='Critic loss')
plt.plot(logs['train/actor_loss'], label='Actor loss')
plt.plot(logs['train/alpha_loss'], label='Alpha loss')
plt.legend()
plt.title('Training loss')
plt.xlabel('Steps')
plt.show()

# Plot evaluation reward
plt.plot(len(logs['train/q_loss'])*np.linspace(0,1,len(logs['eval/reward'])), logs['eval/reward'])
plt.title('Evaluation reward (env: HalfCheetah-v4)')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.show()

# Plot training reward
plt.plot(logs['train/reward'])
plt.title('Training reward (env: HalfCheetah-v4)')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.show()


# Plot alpha
plt.semilogy(logs['train/alpha'])
plt.title('Alpha value over training time (log scale on y axis)')
plt.show()
