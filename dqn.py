import pdb
import time
import os
import torch
import random
import torch.nn as nn
import torch.optim as optim
import pickle
from copy import deepcopy
from ballbuffer import PrioritizedReplayBuffer
from cnn import device
from ballhandler import GameEnvironment

class DQN:
    def __init__(self, action_size=4, gamma=0.99, tau=0.01, lr=0.00025,
                 eps_min=0.2, eps_max=1, eps_eval=0.1,
                 eps_decay_per_episode=1e-4, buffer_size=4000000, batch_size=32,
                 env_fun=lambda : GameEnvironment(600, 416), name=""):

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

        self.env_fun = env_fun # Called each time to start new episode

        self.target_model = deepcopy(self.model).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Prioritized replay buffer
        self.buffer = PrioritizedReplayBuffer(buffer_size)
        self.batch_size = batch_size # Batch size for replay training

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

    def soft_update(self):
        """Updates target model towards source model."""

        for tp, sp in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def act(self, env, state, eps=None):
        """Calculate optimal action from a given state. Custom epsilon can be supplied, or use one provided by the agent."""

        if eps is None:
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

        pdb.set_trace()
        state, action, reward, next_state, done = batch

        V_next = self.target_model(next_state).max(dim=1).values
        Q_target = reward + self.gamma * (1 - done) * V_next
        Q = torch.gather(self.model(state), 1, action.to(torch.int64).unsqueeze(1)).squeeze()

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

    # def get_state(self, env):
    #     state = env.get_state()
    #     state = self.augment_state(state)
    #     return state
    
    # def step(self, env, action):
    #     """Takes one step in environment and returns the next state, reward, and whether it terminated or stuck."""
    #     next_state, reward, is_done, is_stuck = env.step(action)
    #     return next_state, reward, is_done, is_stuck
    

    def epsilon_decay(self):
        """Decays epsilon corresponding to one episode."""
        self.eps = max(self.eps_min, self.eps-self.eps_decay_per_episode)

    def is_trainable(self):
        """Returns True if the model is ready to be trained, i.e the replay buffer is full."""
        return self.buffer.real_size == self.buffer.size

    
    def play_one_episode(self, mode, eps=None):
        """Play one complete episode, either in training mode or evaluation mode, optionally with a custom epsilon. Return the total episode reward, loss and whether the episode finished 'normally'."""
        # mode is either "train" or "eval"
        env = self.env_fun()
        state = env.get_state()
        # choose epsilon depending on mode
        if eps is None:
            eps = self.eps if mode == "train" else self.eps_eval
        is_done, is_stuck = False, False
        episode_reward, episode_loss = 0, 0
        while not is_done:
            # Action
            action = self.act(env, state.unsqueeze(0), eps=eps)
            
            # Step and optionally save in buffer if training
            next_state, reward, is_done, is_stuck = env.step(action)
            is_done = is_done or is_stuck # Also stop if we get stuck
            episode_reward += float(reward)
            if mode == "train":
                self.buffer.add((state, action, reward, next_state, int(is_done)))
            state = next_state

            # Backprop if in training mode and agent is ready to start training
            if mode == "train" and self.is_trainable():
                batch, _, tree_idxs = self.buffer.sample(self.batch_size)
                loss, td_error = self.update(batch)
                self.buffer.update_priorities(tree_idxs, td_error.numpy())
                episode_loss += loss

        # Remove environment and return
        del env
        return episode_reward, episode_loss, (not is_stuck)
