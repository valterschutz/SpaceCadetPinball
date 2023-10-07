import gymnasium as gym
from pinball import load_model, PinballAgent, FrameBuffer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Constants
PRINT = True

env = gym.make("ALE/VideoPinball-v5")

# hyperparameters
learning_rate = 1e-1
n_episodes = 1000
initial_epsilon = 1.0
epsilon_decay = initial_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1
batch_size = 32
discount_factor = 0.99
buffer_size = 1000 


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
    buffer_size = buffer_size,
    pinball_network = model
)

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
    episode_loss = []
    while not done:
        action = agent.get_action(fb.values().unsqueeze(0))

        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        # Punish actions
        if action != 0:
            reward = float(reward) - 1
        fb.append(next_obs)
        agent.save_experience(fb,action,reward,terminated)

        # update the agent using replay experience
        loss = agent.replay_update(batch_size)
        episode_loss.append(loss)

        # update if the environment is done and the current obs
        done = terminated or truncated

    # Print stats about episode
    print(f"Episode {episode} (eps = {agent.epsilon}) got reward {episode_reward:,} and mean loss {sum(episode_loss)/len(episode_loss):,}")

    # Save model every 10th episode
    if (not episode == 0) and episode % 10 == 0:
        agent.save_model()

    agent.decay_epsilon()

# rolling_length = 500
# fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward = np.array(env.return_queue).flatten()
# reward_moving_average = (
#     np.convolve(
#         np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
#     )
#     / rolling_length
# )
axs[0].plot(range(len(reward)), reward)
axs[1].set_title("Episode lengths")
length = np.array(env.length_queue).flatten()
# length_moving_average = (
#     np.convolve(
#         np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
#     )
#     / rolling_length
# )
axs[1].plot(range(len(length)), length)
axs[2].set_title("Training Error")
training_error = np.array(agent.training_error)
# training_error_moving_average = (
#     np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
#     / rolling_length
# )
axs[2].plot(range(len(training_error)), training_error)
plt.tight_layout()
plt.show()
