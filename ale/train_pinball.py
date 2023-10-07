import gymnasium as gym
from pinball import load_model, modified_reward, PinballAgent, FrameBuffer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import nn, optim

# Constants
PRINT = True

env = gym.make("ALE/VideoPinball-v5")

# hyperparameters
learning_rate = 1e-4
n_episodes = 201
initial_epsilon = 1.0
epsilon_decay = initial_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
batch_size = 32
discount_factor = 0.99
buffer_size = 3000 
criterion = nn.MSELoss
optimizer = optim.Adam

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

fb = FrameBuffer(4,(210,160))

model = load_model()
agent = PinballAgent(
    action_space=env.action_space,
    learning_rate=learning_rate,
    initial_epsilon=initial_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor,
    buffer_size=buffer_size,
    pinball_network=model,
    criterion=criterion,
    optimizer=optimizer
)

episode_mean_losses = []
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    fb.reset()
    fb.append(obs)
    done = False

    # Print estimated Q-values for starting state
    if PRINT:
        print(agent.DQN(fb.values().unsqueeze(0)))

    # play one episode
    episode_reward = 0
    episode_losses = []
    while not done:
        action = agent.get_action(fb.values().unsqueeze(0))

        next_obs, reward, terminated, truncated, info = env.step(action)
        reward = modified_reward(reward,action)
        episode_reward += reward
        fb.append(next_obs)
        agent.save_experience(fb,action,reward,terminated)

        # update the agent using replay experience
        loss = agent.replay_update(batch_size)
        episode_losses.append(loss)

        # update if the environment is done and the current obs
        done = terminated or truncated

    episode_mean_loss = sum(episode_losses)/len(episode_losses)
    episode_mean_losses.append(episode_mean_loss)
    # Print stats about episode
    print(f"Episode {episode} (eps = {agent.epsilon}) got reward {episode_reward:,} and mean loss {episode_mean_loss:,}")

    # Save model every 10th episode
    if (not episode == 0) and episode % 10 == 0:
        agent.save_model()

    agent.decay_epsilon()

fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
reward = np.array(env.return_queue).flatten()
axs[0].plot(range(len(reward)), reward)
axs[1].set_title("Episode lengths")
length = np.array(env.length_queue).flatten()
axs[1].plot(range(len(length)), length)
axs[2].set_title("Mean loss")
mean_losses = np.array(episode_mean_losses)
axs[2].plot(range(len(mean_losses)), mean_losses)
plt.tight_layout()
if not os.path.exists("figures"):
    os.makedirs("figures")
plt.savefig(f"figures/{agent.save_time}")
