import time
import itertools
import random
from ballhandler import GameEnvironment
from ballgg import DQN
from cnn import device
import pickle

BUFFER_SIZE = 40000

def eval_agent(agent, episodes=None):
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
        state = agent.get_state(env)
        while not done:
            action = agent.act(env, state.unsqueeze(0), eps)
            # state, reward = env.step(action)
            state, reward = agent.step(env,action)
            done = env.is_done()
            total_reward += reward
            time.sleep(0.03)
        returns.append(total_reward)
        print("")
        del env
        time.sleep(0.1)

if __name__ == '__main__':
    name = input("DQN agent to eval: ")
    pickle_filename = f"pickles/model_{name}.pkl"
    with open(pickle_filename, "rb") as file:
        agent = pickle.load(file)
    agent.model = agent.model.to(device)

    eval_agent(agent)
