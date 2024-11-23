import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(filename='rewards_history.npy'):
    rewards = np.load(filename)
    plt.figure(figsize=(12,6))
    plt.plot(rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_rewards()
