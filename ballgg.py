import time
import os
import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
from copy import deepcopy
from ballbuffer import PrioritizedReplayBuffer
from cnn import device
from ballhandler import GameEnvironment

BUFFER_SIZE = 4000000

class DQN:
    def __init__(self, action_size=7, gamma=0.99, tau=0.01, lr=0.00025,
                 eps_min=0.2, eps_max=1, eps_eval=0.1,
                 eps_decay_per_episode=1e-4, name=""):

        # The second part of the Q-network
        self.model = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
           nn.Linear(128, action_size)
        ).to(device)

        self.target_model = deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.tau = tau
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay_per_episode = eps_decay_per_episode
        self.eps = eps_max
        self.eps_eval = eps_eval

        self.name = name

        # Create lists of Q-values (at start), loss and score.
        # Append to them at each validation step.
        self.episodes = []
        self.q = []
        self.loss = []
        self.reward = []

        # Keep track of flipper and plunger states
        # -1 means down, 1 means up
        self.rflipper_state = -1
        self.lflipper_state = -1
        self.plunger_state = 1

    def soft_update(self):
        """Updates target model towards source model."""

        for tp, sp in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def act(self, env, state, eps=None):
        """Calculate optimal action from a given state. Custom epsilon can be supplied, or use one provided by the agent."""

        if not eps:
            eps = self.eps

        with torch.no_grad():
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                state = torch.as_tensor(state, dtype=torch.float).to(device) # TODO: this should ideally already be on the device as a tensor?
                qs = self.model(state)
                action = torch.argmax(qs).cpu().numpy().item()
        return action

    def update(self, batch):
        """Updates the Q-network and returns the loss (scalar) and TD error (vector) given a batch of information."""

        state, action, reward, next_state, done = batch
        n_batches = len(action)

        V_next = self.target_model(next_state).max(dim=1).values
        Q_target = reward + self.gamma * (1 - done) * V_next
        Q = self.model(state)[torch.arange(n_batches), action.to(torch.long).flatten()] # TODO: do we really need to convert action?

        assert Q.shape == Q_target.shape, f"{Q.shape}, {Q_target.shape}"

        td_error = torch.abs(Q - Q_target).detach().cpu()
        loss = self.criterion(Q,Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.soft_update()

        return loss.item(), td_error

    def save(self):
        """Pickle the agent."""

        model_name = f"model_{self.name}.pkl"
        if not os.path.exists("pickles"):
            os.makedirs("pickles")
        with open(f"pickles/{model_name}", 'wb') as file:
            pickle.dump(self, file)

    def update_action_state(self, action):
        """Updates one of the three instance variables describing the state of the flippers and plunger"""

        action = "RrLl!.p"[action]
        if action == 'l':
            self.lflipper_state = -1
        elif action == 'L':
            self.lflipper_state = 1
        elif action == 'r':
            self.rflipper_state = -1
        elif action == 'R':
            self.rflipper_state = 1
        elif action == '!':
            self.plunger_state = -1
        elif action == '.':
            self.plunger_state = 1

    def reset_action_state(self):
        self.lflipper_state = -1
        self.rflipper_state = -1
        self.plunger_state = 1

    def get_state(self, env):
        state = env.get_state()
        state = self.augment_state(state)
        return state
    
    def step(self, env, action):
        self.update_action_state(action)
        next_state, reward = env.step(action)
        next_state = self.augment_state(next_state)
        return next_state, reward
    
    def augment_state(self, state):
        action_state_tensor = torch.tensor([self.lflipper_state, self.rflipper_state, self.plunger_state], dtype=torch.float32).to(device)
        return torch.cat((state, action_state_tensor), dim=0)

    def epsilon_decay(self):
        """Decays epsilon corresponding to one episode."""
        self.eps = max(self.eps_min, self.eps-self.eps_decay_per_episode)
    
    def play_one_episode(self, mode):
        """Play one complete episode, either in training mode or evaluation mode. Return the total reward."""
        # mode is either "train" or "eval"
        env = GameEnvironment(600, 416)
        agent.reset_action_state()
        state = agent.get_state(env)
        # choose epsilon depending on mode
        eps = self.eps if mode == "train" else self.eps_eval
        done = False
        while not done:
            # Action
            action = agent.act(env, state.unsqueeze(0))
            
            # Step and save in buffer
            next_state, reward = agent.step(env, action)
            acc_reward += reward
            done = env.is_done()
            buffer.add((state, action, reward, next_state, int(done)))
            state = next_state

            # Backprop
            if step == BUFFER_SIZE//1000:
                print("Starting backprop")
                training_started = True
            if step > BUFFER_SIZE//1000:
                batch, weights, tree_idxs = buffer.sample(batch_size)
                loss, td_error = agent.update(batch)
                buffer.update_priorities(tree_idxs, td_error.numpy())
                acc_loss += loss
                counter += 1
            step += 1


def evaluate_policy(agent, eps):
    """Evaluate the agent without training it for one episode."""
    
    print(f"Evaluating policy for one episode with eps={eps}") # TODO: should not be here

    env = GameEnvironment(600, 416)
    agent.reset_action_state()
    state = agent.get_state(env)
    qs = agent.model(state)
    qs_np = qs.cpu().numpy()[0]
    print(f"Q-values: {qs_np}") # TODO: weird to have printing here
    done, episode_reward = False, 0
    print("Actions:")
    while not done:
        action = agent.act(env, state.unsqueeze(0), eps)
        print("RrLl!.p"[action], end="")
        state, reward = agent.step(env, action)
        episode_reward += reward
        done = env.is_done()
    print("")
    del env
    time.sleep(0.1)
        
    return episode_reward, qs_np


def train(agent, buffer, batch_size=128,
        eps_max=1, eps_min=0.0, decrease_eps_steps=1000000, test_every_episodes=50):
    """Train the agent ad infinitum with decreasing epsilon."""

    # TODO: check this whole function

    # If we are resuming a previously trained model, remember where we ended
    if agent.episodes:
        episodes = agent.episodes[-1]
    else:
        episodes = 0
    step = 0 # How many states we have seen in total across all episodes
    eps = max(eps_max - (eps_max - eps_min) * step / decrease_eps_steps, eps_min)
    acc_reward, acc_loss = 0, 0 # Accumulated reward and loss over several episodes
    counter = 0 # Keeps track of how many episodes the above variables correspond to

    done = False
    training_started = False # Only start training when buffer is sufficiently full
    while True:
        if counter and training_started:
            time.sleep(0.1)
            eval_episode_reward, eval_qs = evaluate_policy(agent, eps)
            agent.episode.append(episodes)
            agent.q.append(eval_qs)
            agent.reward.append(eval_episode_reward)
            mean_loss = acc_loss / counter
            agent.loss.append(mean_loss)
            print(f"Summary of last {test_every_episodes} episodes: Step: {step}, Mean Loss: {mean_loss:.6f}, Eps: {eps}\n")
            agent.save()
            counter, acc_loss = 0, 0

        # Run some episodes
        for _ in range(test_every_episodes):
            agent.play_one_episode()

            # Episode finished
            done = False
            del env
            time.sleep(0.1)
            print(f"   Episode {episodes} done... Total reward = {acc_reward:.3f}")
            episodes += 1

def append_to_file(data):
    """Write data to a file with filename equal to process ID."""

    pid = os.getpid()
    file_path = f"textdata/{pid}.txt"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        # Create the file if it doesn't exist
        with open(file_path, 'w'):
            pass  # This will create an empty file
    
    with open(file_path, "a") as file:
        file.write(data)

def print_model_layers(model):
    """Print the contents of all layers in a model and whether the gradients are computed."""

    for name, param in model.named_parameters():
        print(f"Layer: {name}, Requires Gradients: {param.requires_grad}")

def run_train_loop(agent):
    buffer = PrioritizedReplayBuffer(1, BUFFER_SIZE)
    train(agent, buffer, batch_size=128, eps_max=1, eps_min=0.5, decrease_eps_steps=1000000, test_every_episodes=20)

if __name__ == "__main__":
    lr = 1e-6
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        name = input("DQN agent to load: ")
        pickle_filename = f"pickles/model_{name}.pkl"
        with open(pickle_filename, "rb") as file:
            agent = pickle.load(file)
    else:
        # Create a new DQN model if not loading
        # Ask for a name
        name = input("Enter a name for new DQN agent: ")
        agent = DQN(lr=lr, name=name, gamma=0.995)
    agent.optimizer = optim.Adam(agent.model.parameters(), lr=lr)
    agent.tau = 0.05
        
    run_train_loop(agent)


