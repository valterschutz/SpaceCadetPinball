import argparse
from dqn import DQN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def eval_loop(agent, eps, delay):
    """Evaluate a RL agent with epsilon-greedy policy."""

    scores = []
    episodes = 0
    try:
        while True:
            r = agent.play_one_episode(mode="eval", eps=eps, delay=delay)
            score = r[1]
            scores.append(score)
            episodes += 1
    except:
        print(f"\nmean score over past {episodes} episodes: {sum(scores)/len(scores)}")
        plt.figure()
        sns.lineplot(x=range(1,episodes+1), y=scores)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title(f"Mean score: {sum(scores)/len(scores)}")
        plt.show()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a RL agent to play pinball")
    parser.add_argument("name", help="Name of model")
    parser.add_argument("--eps", type=float, default=0.1, help="Which epsilon to use for evaluation")
    parser.add_argument("--delay", type=float, default=0.02, help="How many seconds to wait between each step in the simulation")
    args = parser.parse_args()
    name = args.name
    eps = args.eps
    delay = args.delay

    agent = DQN(name=name)
    agent.load()

    eval_loop(agent, eps, delay)
