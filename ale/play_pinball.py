import gymnasium as gym
from pinball import FrameBuffer, load_model, PinballAgent

env = gym.make("ALE/VideoPinball-v5", render_mode="human")

model = load_model()
agent = PinballAgent(
    action_space=env.action_space,
    pinball_network=model,
    initial_epsilon=0.1,
    epsilon_decay=0.0,
    final_epsilon=0.1
)
agent.DQN.eval()

fb = FrameBuffer(4,(210,160))

# Keep playing until interupted
while True:
    obs, info = env.reset()
    fb.reset()
    fb.append(obs)
    done = False

    # play one episode
    while not done:
        action = agent.get_action(fb.values().unsqueeze(0), explore=True)

        next_obs, reward, terminated, truncated, info = env.step(action)
        fb.append(next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs
