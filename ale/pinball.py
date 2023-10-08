from __future__ import annotations

from datetime import datetime
import glob
import os

import numpy as np
import torch
from torch import nn, optim

class FrameBuffer:
    def __init__(self,n_frames,frame_size):
        self.buffer_size = n_frames+1
        self.frame_size = frame_size
        self.reset()
    def append(self,obs):
        # Send to device
        t = torch.from_numpy(obs).to(torch.float32).to(get_device())
        # First convert to grayscale
        obs = 0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2]
        if self.fill < self.buffer_size:
            self.state[self.fill] = obs
            self.fill += 1
        else:
            self.state[0:self.buffer_size-1] = self.state[1:].clone()
            self.state[-1] = obs
    def values(self):
        if self.fill == 0:
            return torch.zeros(self.buffer_size, *self.frame_size)
        elif self.fill < self.buffer_size:
            # Repeat the last frame
            return torch.cat((self.state[0:self.fill], self.state[self.fill-1].unsqueeze(0).repeat(self.buffer_size-self.fill-1,1,1)))
        else:
            return self.state[1:]
    def prev_values(self):
        if self.fill == 0:
            return torch.zeros(self.buffer_size, *self.frame_size)
        elif self.fill < self.buffer_size:
            # Repeat the last frame
            return torch.cat((self.state[0:self.fill], self.state[self.fill-1].unsqueeze(0).repeat(self.buffer_size-self.fill-1,1,1)))
        else:
            return self.state[0:-1]
    def reset(self):
        self.state = torch.zeros(self.buffer_size,*self.frame_size).to(get_device())
        self.fill = 0

class PinballNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4,8,8,stride=4,padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8,16,4,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.lin_layers = nn.Sequential(
            nn.Linear(8320, 256),  # TODO: remove hard-coded value
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 9)
        )

    def forward(self, x):
        # print(f"shape before conv layers: {x.shape}")
        x = self.conv_layers(x)
        # print(f"shape after conv layers: {x.shape}")
        x = nn.Flatten()(x)
        # print(f"shape after flattening: {x.shape}")
        x = self.lin_layers(x)
        # print(f"shape after lin layers: {x.shape}")
        return x

class PinballAgent:
    def __init__(
        self,
        action_space,
        learning_rate=0.001,
        initial_epsilon=0.9,
        epsilon_decay=0.0,
        final_epsilon=0.1,
        discount_factor=0.95,
        buffer_size = 1000,
        pinball_network = None,
        criterion = nn.MSELoss,
        optimizer = optim.Adam
    ):
        device = get_device()

        self.action_space = action_space

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # The underlying DQN network
        self.DQN = pinball_network or PinballNetwork()
        self.DQN.train()
        self.DQN.to(device)

        # Optimization
        self.criterion = criterion()
        self.optimizer = optimizer(self.DQN.parameters(), lr=self.lr)

        # Experience replay buffer
        self.buffer_size = buffer_size
        self.current_state_buffer = torch.zeros(self.buffer_size,4,210,160).to(device)
        self.next_state_buffer = torch.zeros(self.buffer_size,4,210,160).to(device)
        self.action_buffer = torch.zeros(self.buffer_size,1,dtype=torch.int64).to(device)
        self.reward_buffer = torch.zeros(self.buffer_size,1).to(device)
        self.terminating_buffer = torch.zeros(self.buffer_size,1,dtype=torch.bool).to(device)
        self.buffer_index = 0 # Next position to fill in the buffer
        self.buffer_fill = -1 # Up to what index we have filled the buffer

        # In case we save the model, remember when
        self.save_time = None

    def get_action(self, state):
        # Assume only one observation
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            self.DQN.eval()
            with torch.no_grad():
                output = self.DQN(state)
                return output.squeeze().argmax() # Assume only one output

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        terminated
    ):
        """Updates the Q-value of an action. Should work with batches."""
        self.DQN.train()

        self.optimizer.zero_grad()

        # Forward pass
        current_outputs = self.DQN(state)
        current_q_value = torch.gather(current_outputs,1,action)
        future_outputs = self.DQN(next_state)

        future_v_value = (~terminated) * future_outputs.max(dim=1).values.unsqueeze(-1)
        td_target = reward + self.discount_factor * future_v_value

        # Loss, backward pass
        loss = self.criterion(current_q_value, td_target)
        loss.backward()
        self.optimizer.step()

        # Store loss
        loss_item = loss.item()
        return loss_item

    def save_experience(self, frame_buffer, action, reward, terminated):
        """Stores a transition in replay experince"""
        next_state = frame_buffer.values()
        state = frame_buffer.prev_values()
        self.current_state_buffer[self.buffer_index] = state
        self.next_state_buffer[self.buffer_index] = next_state
        self.action_buffer[self.buffer_index] = action
        self.reward_buffer[self.buffer_index] = reward
        self.terminating_buffer[self.buffer_index] = terminated
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        self.buffer_fill = max(self.buffer_fill, self.buffer_index)

    def replay_update(self, batch_size):
        """Do one update using batch_size samples from replay experience"""
        random_indices = torch.randint(0, self.buffer_fill, (batch_size,))
        s = self.current_state_buffer[random_indices,:]
        a = self.action_buffer[random_indices,:]
        r = self.reward_buffer[random_indices,:]
        sp = self.next_state_buffer[random_indices,:]
        term = self.terminating_buffer[random_indices,:]

        return self.update(s, a, r, sp, term)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_model(self):
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
        file_name = f"{formatted_datetime}.pth"
        path = f"models/{file_name}"
        torch.save(self.DQN.state_dict(), path)
        print(f"Model saved to '{path}'")
        self.save_time = formatted_datetime

    def get_Q(self, state):
        """Returns Q-values for a single state"""
        self.DQN.eval()
        with torch.no_grad():
            Q = self.DQN(state.unsqueeze(0))
        return Q

def get_device():
    # Get CUDA device
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    return device

def load_model():
    """Tries to load latest model from directory "models", otherwise initialize new model."""

    def path_to_datestr(path):
        filename = os.path.basename(path)
        return os.path.splitext(filename)[0]

    # Create directory models if it does not exist already
    if not os.path.exists("models"):
        os.makedirs("models")

    try:
        paths = glob.glob(os.path.join("models","*.pth"))
        latest_path = max(paths, key=lambda x: datetime.strptime(path_to_datestr(x), "%Y-%m-%d_%H:%M:%S"))
        model = PinballNetwork()
        model.load_state_dict(torch.load(latest_path))
        print(f"Resuming model '{latest_path}'")
    except:
        model = None
        print(f"Created new model")
    return model

def modified_reward(reward, action):
    """Given a true reward and an action take it, modify it so all positive awards have value 1 and actions are punished"""
    reward = float(reward)
    if reward > 0:
        reward = 1
    else:
        reward = 0
    # if action != 0:
    #     reward -= 0.1
    return reward
