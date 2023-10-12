import time
import sys
import numpy as np
import itertools
import glob
import os
import torch
import random
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from cnn import device
from gamehandler import GameEnvironment
from gg import DQN


def plot_q_values(q_matrix, name):
    plt.figure()
    for i in range(7):
        plt.plot(range(q_matrix.shape[0]), q_matrix[:,i], label="RrLl!.p"[i])

    plt.xlabel('Steps')
    plt.ylabel('Q-Values')
    plt.legend()
    plt.savefig(f'figs/{name}_q-values-over-time.png')
    plt.close()

    ## normalize q
    row_means = np.mean(q_matrix, axis=1)
    normalized_q_matrix = q_matrix / row_means[:, np.newaxis]
    plt.figure()
    for i in range(7):
        plt.plot(range(normalized_q_matrix.shape[0]), normalized_q_matrix[:,i], label="RrLl!.p"[i])

    plt.xlabel('Steps')
    plt.ylabel('Q-Values')
    plt.legend()
    plt.savefig(f'figs/{name}_q-values-over-time-normalized.png')
    plt.close()

def get_qs(agent, state):
    with torch.no_grad():
        state = torch.as_tensor(state, dtype=torch.float).to(device)
        return agent.model(state.unsqueeze(0)).cpu().numpy()[0]

def evaluate_policy(agent, episodes=None):
    print("Playing game...")
    # Loop forever if no episodes given
    if episodes == None:
        iter = itertools.count(start=0, step=1)
    else:
        iter = range(episodes)
    
    eps = 0.3
    Qs = []

    for ep in iter:
        env = GameEnvironment(600, 416)
        done, total_reward = False, 0
        state = env.get_state()
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = agent.act(state.unsqueeze(0))
            Qs.append(get_qs(agent, state))
            state, reward = env.step(action)
            done = env.is_done()
        del env
        time.sleep(0.1)
    return np.array(Qs)


# model_directory = "pickles"
# model_files = glob.glob(os.path.join(model_directory, "model_*.pkl"))
# model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
# if len(sys.argv) > 1:
#     latest_model_file = sys.argv[1]
# else:
#     latest_model_file = model_files[0]
# with open(latest_model_file, "rb") as file:
#     agent = pickle.load(file)
#     agent.model = agent.model.to(device)
# print(f"Loaded {latest_model_file}...")
#name = latest_model_file.split("/")[-1].split(".")[0].split("_")[-1]
if __name__ == "__main__":
    name = input("DQN agent to eval: ")
    pickle_filename = f"pickles/model_{name}.pkl"
    with open(pickle_filename, "rb") as file:
        agent = pickle.load(file)
    agent.model = agent.model.to(device)
    Qs = evaluate_policy(agent, episodes=1)
    plot_q_values(Qs, name)
    print("Saved nice pictures...")
