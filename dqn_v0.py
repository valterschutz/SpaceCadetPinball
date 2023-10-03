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

