import time
import itertools
import glob
import os
import torch
import random
from copy import deepcopy
from cnn import get_device as device
from gamehandler import GameEnvironment
from gg import DQN

BUFFER_SIZE = 40000

def evaluate_policy(agent, episodes=None):
    # Loop forever if no episodes given
    if episodes == None:
        iter = itertools.count(start=0, step=1)
    else:
        iter = range(episodes)
    
    returns = []
    eps = 0.1

    for ep in iter:
        env = GameEnvironment(600, 416)
        done, total_reward = False, 0
        state = env.get_state()
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = agent.act(state.unsqueeze(0))
            state, reward = env.step(action)
            done = env.is_done()
            total_reward += reward
            time.sleep(0.05)
        returns.append(total_reward)
        print("")
        del env
        time.sleep(0.1)


# Load the model from "./gg/"
agent = DQN()
model_directory = "good_models"
# Get a list of all model files in the directory
model_files = glob.glob(os.path.join(model_directory, "model_*.pkl"))
# Sort the model files by timestamp (assuming the timestamp format is consistent)
model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
# Check if there are any model files
if model_files:
    latest_model_file = model_files[0]
    agent.model = torch.load(latest_model_file).to(device())
    agent.target_model = deepcopy(agent.model).to(device())
    print(f"Loaded {latest_model_file}...")
    evaluate_policy(agent)



