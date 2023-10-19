import time
import sys
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
    scores = []
    eps = 0.1

    for ep in iter:
        env = GameEnvironment(600, 416)
        done, total_reward = False, 0
        state = agent.get_state(env)
        is_stuck = False
        while not done:
            action = agent.act(env, state.unsqueeze(0), eps)
            # state, reward = env.step(action)
            state, reward = agent.step(env,action)
            done, msg = env.is_done()
            total_reward += reward
            # time.sleep(0.03)
            if msg == "bumper":
                is_stuck = True
        if not is_stuck:
            returns.append(total_reward)
            score = env.score[0]
            scores.append(score)
            print(ep, score)
        del env
        # time.sleep(0.1)
    return scores

if __name__ == '__main__':
    name = sys.argv[1]
    episodes = int(sys.argv[2])
    pickle_filename = f"pickles/model_{name}.pkl"
    with open(pickle_filename, "rb") as file:
        agent = pickle.load(file)
    agent.model = agent.model.to(device)

    scores = eval_agent(agent, episodes)
    with open(f"pickles/perf_{name}.pkl", "wb") as file:
        pickle.dump(scores, file)
    print(f"Scores pickled.")
