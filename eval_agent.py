import argparse
from dqn import DQN

def eval_loop(agent, eps):
    """Evaluate a RL agent with epsilon-greedy policy."""

    while True:
        agent.play_one_episode(mode="eval", eps=eps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a RL agent to play pinball")
    parser.add_argument("name", help="Name of model")
    parser.add_argument("eps", help="Which epsilon to use for evaluation")
    args = parser.parse_args()
    name = args.name
    eps = args.eps

    agent = DQN(name=name)
    agent.load()

    eval_loop(agent, eps)
