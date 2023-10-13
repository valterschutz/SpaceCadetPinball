import time
import glob
import os
import sys
import datetime
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
from copy import deepcopy
from ballbuffer import PrioritizedReplayBuffer
from cnn import get_device as device
from ballhandler import GameEnvironment

BUFFER_SIZE = 4000000

class DQN:
    def __init__(self, action_size=7, gamma=0.99, tau=0.01, lr=0.00025, name=""):

        # The second part of the Q-network
        self.model = nn.Sequential(
                nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
           nn.Linear(128, action_size)
        ).to(device())
        self.target_model = deepcopy(self.model).to(device())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.tau = tau

        self.name = name

        # Create lists of Q-values (at start), loss and score.
        # Append to them at each validation step.
        self.episodes = []
        self.q = []
        self.loss = []
        self.reward = []

    def soft_update(self):
        """Updates target model towards source model."""

        for tp, sp in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def act(self, state, save_Q=False):
        """Calculate optimal action from a given state."""

        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float).to(device()) # TODO: this should ideally already be on the device as a tensor?
            qs = self.model(state)
            qs_np = qs.cpu().numpy()[0]
            if save_Q:
                self.q.append(qs_np)
                print(f"Q-values: {qs_np}") # TODO: weird to have printing here
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

def evaluate_policy(agent, episodes=5, is_printing=False):
    """Evaluate the agent without training it for a certain amount of episodes."""
    
    returns = []
    eps = 0.1 # TODO: should be a parameter

    print(f"Evaluating policy for {episodes} episodes with eps={eps}") # TODO: should not be here

    for _ in range(episodes):
        env = GameEnvironment(600, 416)
        state = env.get_state()
        done, total_reward = False, 0
        print("Actions:")
        is_first_frame = True
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = agent.act(state.unsqueeze(0), is_printing and is_first_frame)
                print("RrLl!.p"[action], end="")
                is_first_frame = False
            state, reward = env.step(action)
            total_reward += reward
            done = env.is_done()
        returns.append(total_reward)
        if is_printing:
            print("")
        del env
        time.sleep(0.1)
        
    return np.mean(returns), np.std(returns)


def train(agent, buffer, batch_size=128,
        eps_max=1, eps_min=0.0, decrease_eps_steps=1000000, test_every_episodes=50):
    """Train the agent ad infinitum with decreasing epsilon."""

    # TODO: check this whole function

    # If we are resuming a previously trained model, remember where we ended
    if agent.episodes:
        episodes = agent.episodes[-1]
    else:
        episodes = 0
    step = 0
    total_reward = 0
    loss_count, total_loss = 0, 0

    done = False
    training_started = False
    while True:
        if loss_count and training_started:
            time.sleep(0.1)
            eval_reward, _ = evaluate_policy(agent, episodes=1, is_printing=True)
            agent.reward.append(eval_reward)
            mean_loss = total_loss / loss_count
            agent.loss.append(mean_loss)
            agent.episodes.append(episodes)
            print(f"Evaluation reward: {eval_reward}")
            print(f"Summary of last {test_every_episodes} episodes: Step: {step}, Mean Loss: {mean_loss:.6f}, Eps: {eps}\n")
            agent.save()
            loss_count, total_loss = 0, 0

        # Run some episodes
        for _ in range(test_every_episodes):
            total_reward = 0
            env = GameEnvironment(600, 416)
            state = env.get_state()
            while not done:

                # Action
                eps = max(eps_max - (eps_max - eps_min) * step / decrease_eps_steps, eps_min)
                if random.random() < eps:
                    action = env.action_space.sample()
                else:
                    action = agent.act(state.unsqueeze(0))
                
                # Step and save in buffer
                next_state, reward = env.step(action)
                total_reward += reward
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
                    total_loss += loss
                    loss_count += 1
                step += 1

            # Episode finished
            done = False
            del env
            time.sleep(0.1)
            print(f"   Episode {episodes} done... Total reward = {total_reward:.3f}")
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
    train(agent, buffer, batch_size=32, eps_max=1, eps_min=0.2, decrease_eps_steps=1000000, test_every_episodes=20)

if __name__ == "__main__":
    lr = 5e-7
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        name = input("DQN agent to load: ")
        pickle_filename = f"pickles/model_{name}.pkl"
        with open(pickle_filename, "rb") as file:
            agent = pickle.load(file)
    else:
        # Create a new DQN model if not loading
        # Ask for a name
        name = input("Enter a name for new DQN agent: ")
        agent = DQN(lr=lr, name=name)
    agent.optimizer = optim.Adam(agent.model.parameters(), lr=lr)
    agent.tau = 0.001
        
    run_train_loop(agent)


