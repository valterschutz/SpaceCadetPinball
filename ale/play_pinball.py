import gymnasium as gym
from pinball import obs_to_tensor, load_model, PinballAgent

env = gym.make("ALE/VideoPinball-v5", render_mode="human")

model = load_model()
agent = PinballAgent(action_space=env.action_space, pinball_network=model)
agent.DQN.eval()

# Keep playing until interupted
while True:
    obs, info = env.reset()
    obs = obs_to_tensor(obs)
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = obs_to_tensor(next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs
