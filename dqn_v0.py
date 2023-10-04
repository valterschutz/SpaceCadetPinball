import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from train_cnn import load_latest_model, get_device
from gamehandler import GameEnvironment

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, NUM_ACTIONS):
        super(DQN, self).__init__()
        
        # Load the BallDetectionCNN model
        self.ball_cnn = load_latest_model()
        for param in self.ball_cnn.parameters():
            param.requires_grad = False
        
        # Define the Q-network layers
        self.layers = nn.Sequential(
                self.ball_cnn,
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, NUM_ACTIONS),
                )
        
    def forward(self, x):
        return self.layers(x)

NUM_ACTIONS = 7 # 11 with table bump, 7 without
GAMMA = 0.995
EPSILON = 0.1
REPLAY_BUFFER_SIZE = 1000  # Size of the replay buffer
BATCH_SIZE = 32  # Batch size for training

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def main():
    dqn = DQN(NUM_ACTIONS)
    dqn.to(get_device())

    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    # Training loop
    num_episodes = 1000  # Number of episodes to train
    for episode in range(num_episodes):
        env = GameEnvironment(600, 416)
        state = env.get_external_state().to(get_device())
        print(f"   Training episode {episode}...")
        while True:
            if random.random() < EPSILON:
                action = random.randint(0, NUM_ACTIONS - 1)  # Explore
            else:
                q_values = dqn(state.unsqueeze(0))
                action = torch.argmax(q_values).item()  # Exploit
            
            # Simulate the game step and get the next state and reward
            next_state, reward = env.step(action)
            next_state = next_state.to(get_device())
            reward = reward.to(get_device())
            if reward.item():
                print(f"      Score updated to {env.score}")
                #last_layer_weights = list(dqn.parameters())[-2].data
                #print("Part of last layer in dqn:", last_layer_weights[:,0].detach().cpu().numpy())
                #print("Q values: [R r L l ! . p]", q_values.detach().cpu().numpy())

            # Store the transition in the replay buffer
            if env.frame_id > 500: # We don't need to save the first 300 frames. The ball hasn't dropped yet.
                replay_buffer.push((state, action, next_state, reward))

            # Sample a batch of experiences from the replay buffer
            if len(replay_buffer.buffer) >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)

                states, actions, next_states, rewards = zip(*batch)

                states = torch.stack(states)
                actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(get_device())
                next_states = torch.stack(next_states)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(get_device())

                optimizer.zero_grad()

                # Compute the Q-values and target values
                q_values = dqn(states)
                next_q_values = dqn(next_states)
                max_next_q_values = torch.max(next_q_values, dim=1, keepdim=True)[0]
                targets = rewards + GAMMA * max_next_q_values

                # Get the Q-values for the selected actions
                # what the f*** does this do?
                q_values = q_values.gather(1, actions)

                # Compute the loss and update the Q-network
                loss = criterion(q_values, targets)
                #print(f"Loss: {loss.item():.3f}")
                loss.backward()
                optimizer.step()

            state = next_state
            if env.is_done():
                del env
                break

if __name__ == "__main__":
    main()


"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from train_cnn import load_latest_model, get_device
from gamehandler import GameEnvironment


# Define the DQN model
class DQN(nn.Module):
    def __init__(self, NUM_ACTIONS):
        super(DQN, self).__init__()
        
        # Load the BallDetectionCNN model
        self.ball_cnn = load_latest_model()
        for param in self.ball_cnn.parameters():
            param.requires_grad = False
        
        # Define the Q-network layers
        self.layers = nn.Sequential(
                self.ball_cnn,
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, NUM_ACTIONS),
                )
        
    def forward(self, x):
        return self.layers(x)

NUM_ACTIONS = 7
GAMMA = 0.995
EPSILON = 0.1

def main():
    dqn = DQN(NUM_ACTIONS)
    dqn.to(get_device())

    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_episodes = 1000  # Number of episodes to train
    for episode in range(num_episodes):
        env = GameEnvironment(600, 416)
        state = env.get_external_state().to(get_device())
        print(f"   Training episode {episode}...")
        while True:
            if random.random() < EPSILON:
                action = random.randint(0, NUM_ACTIONS - 1)  # Explore
            else:
                q_values = dqn(state.unsqueeze(0))
                action = torch.argmax(q_values).item()  # Exploit
            
            # Simulate the game step and get the next state and reward
            next_state, reward = env.step(action)
            next_state = next_state.to(get_device())
            reward = reward.to(get_device())
            if reward.item():
                print(f"      Score updated to {env.score}")
            
            # Update the Q-network using the Q-learning formula
            target = reward + 0.99 * torch.max(dqn(next_state.unsqueeze(0)))
            current = dqn(state.unsqueeze(0))[0][action]
            loss = criterion(current, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            if env.is_done():
                del env
                break




main()
"""
