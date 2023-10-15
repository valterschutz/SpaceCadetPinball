import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dqn import DQN

sns.set_style("whitegrid")

def plot_q_values(episodes, q_values, name):
    # q_values = np.array(q_values)
    plt.figure()
    for i in range(4):
        sns.lineplot(x=episodes, y=q_values[:,i], label=["Left flipper", "Right plunger", "Plunger", "No action"][i])

    plt.xlabel('Episode')
    plt.ylabel('Q-Values')
    plt.legend()
    plt.savefig(f'figs/{name}_q-values.png')
    plt.close()

def plot_mean_loss(episodes, mean_loss_values, name):
    plt.figure()
    sns.lineplot(x=episodes, y=mean_loss_values)
    plt.xlabel('Episode')
    plt.ylabel('Training loss')
    plt.savefig(f'figs/{name}_training-loss.png')
    plt.close()

def plot_reward(episodes, reward, name):
    plt.figure()
    sns.lineplot(x=episodes, y=reward)
    plt.xlabel('Episode')
    plt.ylabel('Validation reward')
    plt.savefig(f'figs/{name}_reward.png')
    plt.close()

def plot_eps(episodes, eps):
    plt.figure()
    sns.lineplot(x=episodes, y=eps)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.savefig(f'figs/{name}_eps.png')
    plt.close()


def plot_stuff(agent):
    plot_q_values(agent.saved_episodes, agent.saved_Qs, agent.name)
    plot_mean_loss(agent.saved_episodes, agent.saved_losses, agent.name)
    plot_reward(agent.saved_episodes, agent.saved_rewards, agent.name)
    plot_eps(agent.saved_episodes, agent.saved_eps, agent.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot data gathered from a RL agent playing pinball")
    parser.add_argument("name", help="Name of model")
    args = parser.parse_args()
    name = args.name

    agent = DQN(
        name=name
    )
    agent.load()

    if not os.path.exists("figs"):
        os.makedirs("figs")

    plot_stuff(agent)
