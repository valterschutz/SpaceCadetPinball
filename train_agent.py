import argparse
import itertools
from dqn import DQN
from tqdm import tqdm

def train_loop(agent, test_every_n_episodes):
    """Train the agent ad infinitum with decreasing epsilon. Test it every now and then."""

    # If replay buffer is not filled yet, fill it first
    is_trainable = agent.is_trainable()
    if not is_trainable:
        print(f"Filling up replay buffer...")
        with tqdm(initial=agent.buffer.real_size, total=agent.buffer.size) as pbar:
            while agent.buffer.real_size < agent.buffer.size:
                agent.play_one_episode(mode="train")
                pbar.update(agent.buffer.real_size - pbar.n)

    # If we are resuming a previously trained model, remember where we ended
    if len(agent.saved_episodes) > 0:
        episodes_so_far = agent.episodes[-1]
    else:
        episodes_so_far = 0

    # Keep track of accumulated loss over the past training episodes
    # and reset it when evaluating the model
    print("Replay buffer full. Starting training...")
    acc_loss = 0
    for episode in itertools.count(start=episodes_so_far):
        if episode != 0 and episode % test_every_n_episodes == 0:
            mean_loss = acc_loss/test_every_n_episodes
            print(f"Summary for episode {episode-test_every_n_episodes+1}-{episode-1}:")
            print(f"  Average loss: {mean_loss}")
            print(f"  Epsilon: {agent.eps}")
            episode_reward, episode_loss, normal_end, initial_Q = agent.play_one_episode(mode="eval")
            print(f"Validation, episode {episode}:")
            print(f"  Q: {initial_Q}")
            print(f"  Reward: {episode_reward}")
            print(f"  Epsilon: {agent.eps_eval}")
            agent.append_data(episode, episode_reward, mean_loss, initial_Q, agent.eps)
            print("Saving model and data...", end="")
            agent.save()
            print("done")
        else:
            episode_reward, episode_loss, normal_end, initial_Q = agent.play_one_episode(mode="train")
            acc_loss += episode_loss
        agent.eps_decay()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RL agent to play pinball")
    parser.add_argument("mode", help="Whether to 'load' an old model or to create a 'new' model")
    parser.add_argument("name", help="Name of model")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Target model update rate")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--eps_min", type=float, default=0.2, help="Minimum allowed epsilon")
    parser.add_argument("--eps_max", type=float, default=1, help="Maximum allowed epsilon")
    parser.add_argument("--eps_eval", type=float, default=0.1, help="Epsilon to use during evaluation of policy")
    parser.add_argument("--eps_decay_per_episode", type=float, default=1e-4, help="How much to decay epsilon by each episode")
    parser.add_argument("--buffer_size", type=int, default=10_000, help="Size of replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use during training on replay buffer")
    parser.add_argument("--test_every_n_episodes", type=int, default=50, help="How many episodes to wait before evaluating the model again")
    args = parser.parse_args()
    mode = args.mode
    name = args.name
    gamma = args.gamma
    tau = args.tau
    lr = args.lr
    eps_min = args.eps_min
    eps_max = args.eps_max
    eps_eval = args.eps_eval
    eps_decay_per_episode = args.eps_decay_per_episode
    buffer_size = args.buffer_size
    batch_size = args.batch_size
    test_every_n_episodes = args.test_every_n_episodes

    agent = DQN(
        gamma=gamma,
        tau=tau,
        lr=lr,
        eps_min=eps_min,
        eps_max=eps_max,
        eps_eval=eps_eval,
        eps_decay_per_episode=eps_decay_per_episode,
        buffer_size=buffer_size,
        batch_size=batch_size,
        name=name
    )

    if mode == "load":
        agent.load()
    elif mode == "new":
        pass
    else:
        raise Exception("Did not receive a valid mode")

    train_loop(agent, test_every_n_episodes)
