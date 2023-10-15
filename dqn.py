import pdb
import numpy as np
import time
import os
import torch
import random
import torch.nn as nn
import torch.optim as optim
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
        self.model.eval()

        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay_per_episode = eps_decay_per_episode
        self.eps = eps_max
        self.eps_eval = eps_eval

        self.env_fun = env_fun # Called each time to start new episode

        self.target_model = deepcopy(self.model).to(device)
        self.target_model.eval()
        self.optimizer = self.init_optimizer()
        self.criterion = nn.MSELoss()

        # Prioritized replay buffer
        self.buffer = PrioritizedReplayBuffer(buffer_size)
        self.batch_size = batch_size # Batch size for replay training

        self.name = name

        # Create lists of Q-values (at start), loss and score.
        # Append to them at each validation step.
        self.saved_episodes = []
        self.saved_Qs = []
        self.saved_losses = []
        self.saved_rewards = []
        self.saved_eps = []

    def init_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

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
        self.model.train()

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

        self.model.eval()

        return loss.item(), td_error

    def save(self):
        """Save the model and data in the agent."""

        model_filename = f"{self.name}.pth"
        data_filename = f"{self.name}.npz"
        if not os.path.exists("saves"):
            os.makedirs("saves")
        torch.save(self.model.state_dict(), f"saves/{model_filename}")
        np.savez(
            f"saves/{data_filename}",
            saved_episodes=self.saved_episodes,
            saved_rewards=self.saved_rewards,
            saved_losses=self.saved_losses,
            saved_Qs=self.saved_Qs,
            saved_eps=self.saved_eps 
        )

    def load(self):
        """Load the model and saved data."""

        model_filename = f"{self.name}.pth"
        data_filename = f"{self.name}.npz"
        self.model.load_state_dict(torch.load(f"saves/{model_filename}"))
        self.optimizer = self.init_optimizer()
        data = np.load(f"saves/{data_filename}")
        self.saved_episodes = data["saved_episodes"]
        self.saved_rewards = data["saved_rewards"]
        self.saved_losses = data["saved_losses"]
        self.saved_Qs = data["saved_Qs"]
        self.saved_eps = data["saved_eps"]


    def append_data(self, episode, episode_reward, mean_loss, initial_Q, eps):
        self.saved_episodes.append(episode)
        self.saved_rewards.append(episode_reward)
        self.saved_losses.append(mean_loss)
        self.saved_Qs.append(initial_Q)
        self.saved_eps.append(eps)

    def eps_decay(self):
        """Decays epsilon corresponding to one episode."""
        self.eps = max(self.eps_min, self.eps-self.eps_decay_per_episode)

    def is_trainable(self):
        """Returns True if the model is ready to be trained, i.e the replay buffer has at least the same size as batch size."""
        return self.buffer.real_size >= self.batch_size

    
    def play_one_episode(self, mode, eps=None, delay=None):
        """
        Play one complete episode, either in training mode or evaluation mode,
        optionally with a custom epsilon. Returns:
            the total episode reward
            loss
            whether the episode finished 'normally'
            the Q-values found at the start
            length of episode in steps
        """
        # mode is either "train" or "eval"
        env = self.env_fun()
        state = env.get_state()
        initial_Q = self.model(state).detach().cpu().numpy()
        # choose epsilon depending on mode
        if eps is None:
            eps = self.eps if mode == "train" else self.eps_eval
        is_done, is_stuck = False, False
        episode_reward, episode_loss = 0, 0
        episode_len = 0
        while not is_done:
            episode_len += 1

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

            # Delay optional
            if not delay is None:
                time.sleep(delay)

        # Remove environment and return
        del env
        return episode_reward, episode_loss, (not is_stuck), initial_Q, episode_len
