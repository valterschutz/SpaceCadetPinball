import numpy as np
import pickle
import matplotlib.pyplot as plt
from gg import DQN

def plot_q_values(episodes, q_values, name):
    q_values = np.array(q_values)
    plt.figure()
    for i in range(7):
        plt.plot(q_values[:,i], label="RrLl!.p"[i])

    #plt.xlabel('Episode')
    plt.ylabel('Q-Values')
    plt.legend()
    plt.savefig(f'figs/{name}_q-values.png')
    plt.close()

def plot_mean_loss(episodes, mean_loss_values, name):
    plt.figure()
    #plt.plot(episodes[:len(mean_loss_values)], mean_loss_values)
    plt.plot(mean_loss_values)
    #plt.xlabel('Episode')
    plt.ylabel('Mean Loss')
    plt.savefig(f'figs/{name}_mean-loss.png')
    plt.close()

def plot_reward(episodes, reward, name):
    plt.figure()
    #plt.plot(episodes[:len(reward)], reward)
    plt.plot(reward)
    #plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f'figs/{name}_reward.png')
    plt.close()

def plot_stuff(agent):
    plot_q_values(agent.episodes, agent.q, agent.name)
    plot_mean_loss(agent.episodes, agent.loss, agent.name)
    plot_reward(agent.episodes, agent.reward, agent.name)


if __name__ == '__main__':
    name = input("DQN agent to load: ")
    pickle_filename = f"pickles/model_{name}.pkl"
    with open(pickle_filename, "rb") as file:
        agent = pickle.load(file)

    plot_stuff(agent)

