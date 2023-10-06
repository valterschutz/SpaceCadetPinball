import torch
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import random
import time
from cnn import load_latest_model, get_device
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

def save_dqn_model(dqn):
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = f"./dqn_models/model{current_datetime}.pt"
    torch.save(dqn.state_dict(), model_path)

NUM_ACTIONS = 7 # 11 with table bump, 7 without
GAMMA = 0.999
EPSILON_START = 1
EPSILON_END = 0.1
REPLAY_BUFFER_SIZE = 1200  # Size of the replay buffer, 12000 works
BATCH_SIZE = 8  # Batch size for training
LR = 1e-2

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, transition):
        # print(len(self.buffer))
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def main():
    dqn = DQN(NUM_ACTIONS)
    dqn.to(get_device())

    optimizer = optim.Adam(dqn.parameters(), LR)
    criterion = nn.MSELoss()

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    # Training loop
    num_episodes = 10000  # Number of episodes to train
    for episode in range(1, num_episodes):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%d %b %H:%M:%S")
        if episode % 100 == 0:
            print(f"{formatted_time} Saving model")
            save_dqn_model(dqn)
        EPSILON = EPSILON_START * (1 - episode/num_episodes) + episode/num_episodes * EPSILON_END 
        env = GameEnvironment(600, 416)
        state = env.get_state()
        # print(f"{formatted_time} Training episode {episode}/{num_episodes} epsilon={EPSILON:.3f}")

        # Calculate and print the expected Q-value
        q_values = dqn(state.unsqueeze(0))
        # expected_q_value = torch.max(q_values).item()
        print(f"{formatted_time} Training episode {episode}/{num_episodes} epsilon={EPSILON:.3f} Q-values: {q_values}")

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

            #Store the transition in the replay buffer
            replay_buffer.push((state, action, next_state, reward))

            # Sample a batch of experiences from the replay buffer
            if len(replay_buffer.buffer) == REPLAY_BUFFER_SIZE:
                #print(len(replay_buffer.buffer))
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


