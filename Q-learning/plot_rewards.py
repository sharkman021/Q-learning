# plot_rewards.py

import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(filename='rewards_history.npy'):
    try:
        rewards = np.load(filename)
    except FileNotFoundError:
        print(f"File {filename} not found. Please ensure that training has been completed and the file exists.")
        return
    except Exception as e:
        print(f"An error occurred while loading {filename}: {e}")
        return

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
