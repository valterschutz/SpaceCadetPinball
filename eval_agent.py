import argparse
from dqn import DQN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set_style("whitegrid")

def eval_loop(agent, episodes, eps, delay):
    """Evaluate a RL agent with epsilon-greedy policy."""

    scores = []
    episode = 0

    def loop():
        nonlocal episode, scores
        r = agent.play_one_episode(mode="eval", eps=eps, delay=delay)
        score = r[1]
        scores.append(score)
        episode += 1

    try:
        if episodes is None:
            while True:
                loop()
        else:
            with tqdm(initial=0, total=episodes) as pbar:
                while episode < episodes:
                    loop()
                    pbar.update(1)
            raise Exception("go to except")
    except:
        print(f"\nmean score over past {episodes} episodes: {sum(scores)/len(scores)}")
        fig, axes = plt.subplots(1,2)
        sns.lineplot(x=range(1,episode+1), y=scores, ax=axes[0])
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Score")
        axes[0].set_title(f"Mean score: {sum(scores)/len(scores)}")
        sns.histplot(np.log10(scores), kde=True, ax=axes[1])
        axes[1].set_xlabel("log10(score)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Score histogram")
        plt.show()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a RL agent to play pinball")
    parser.add_argument("name", help="Name of model")
    parser.add_argument("--episodes", type=int, default=None, help="How many episodes to play")
    parser.add_argument("--eps", type=float, default=0.1, help="Which epsilon to use for evaluation")
    parser.add_argument("--delay", type=float, default=0.02, help="How many seconds to wait between each step in the simulation")
    parser.add_argument("--n_frames", type=int, default=1, help="How many frames to wait between each action")
    args = parser.parse_args()
    name = args.name
    episodes = args.episodes
    eps = args.eps
    delay = args.delay
    n_frames = args.n_frames

    agent = DQN(name=name, n_frames=n_frames)
    agent.load()

    eval_loop(agent, episodes, eps, delay)
