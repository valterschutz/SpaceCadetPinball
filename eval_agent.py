import argparse
from dqn import DQN

def eval_loop(agent, eps, delay):
    """Evaluate a RL agent with epsilon-greedy policy."""

    while True:
        agent.play_one_episode(mode="eval", eps=eps, delay=delay)

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
