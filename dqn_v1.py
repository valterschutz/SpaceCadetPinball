import torch
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
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
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, NUM_ACTIONS),
        )
        
    def forward(self, x):
        return self.layers(x)

def save_dqn_model(dqn):
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = f"./dqn_models/model{current_datetime}.pt"
    torch.save(dqn.state_dict(), model_path)

NUM_ACTIONS = 7 # 11 with table bump, 7 without
GAMMA = 0.995
EPSILON_START = 0.8
EPSILON_END = 0.1
REPLAY_BUFFER_SIZE = 10000  # Size of the replay buffer, 12000 works
BATCH_SIZE = 8  # Batch size for training
LR = 1e-3
NUM_EPISODES = 500

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.alpha = alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, transition):
        if len(self.buffer) == self.capacity // 10:
            print("STARTING BACKPROP")

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        priorities = priorities ** self.alpha
        prob_weights = priorities / priorities.sum()

        top_indices = np.argsort(prob_weights)[::-1][:10]
        top_values = prob_weights[top_indices]

        indices = np.random.choice(len(self.buffer), batch_size, p=prob_weights)
        transitions = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * prob_weights[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        return [transitions, indices, weights]

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, transition):
        if len(self.buffer) == self.capacity//10:
            print("STARTING BACKPROP")
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

    prio_replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)

    # Training loop
    for episode in range(1, NUM_EPISODES):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%d %b %H:%M:%S")
        if episode % 100 == 0:
            print(f"{formatted_time} Saving model")
            save_dqn_model(dqn)
        EPSILON = EPSILON_START * (1 - episode/NUM_EPISODES) + episode/NUM_EPISODES * EPSILON_END 
        env = GameEnvironment(600, 416)
        state = env.get_state()

        # Calculate and print the expected Q-value
        q_values = dqn(state.unsqueeze(0))
        expected_q_value = torch.max(q_values).item()
        # Print the letters and corresponding values
        print(f"{formatted_time} Training episode {episode}/{NUM_EPISODES} epsilon={EPSILON:.3f}")
        for i, letter in enumerate(['R', 'r', 'L', 'l', '!', '.', 'p']):
            value = q_values[0,i].item()
            print(f"{letter} {value:.4f},   ", end="")
        print("")
        num_states = 0
        while True:
            num_states += 1
            if random.random() < EPSILON:
                action = random.randint(0, NUM_ACTIONS - 1)  # Explore
            else:
                q_values = dqn(state.unsqueeze(0))
                action = torch.argmax(q_values).item()  # Exploit
            
            # Simulate the game step and get the next state and reward
            next_state, reward = env.step(action)
            #if reward.item():
            #    print(reward.item())
            next_state = next_state.to(get_device())
            reward = reward.to(get_device())

            #Store the transition in the replay buffer
            prio_replay_buffer.push((state, action, next_state, reward))

            # Sample a batch of experiences from the replay buffer
            if env.is_done():
                print(f"Created {num_states} states")
                del env
                break
            if len(prio_replay_buffer.buffer) > REPLAY_BUFFER_SIZE/10:
                transitions, indices, weights = prio_replay_buffer.sample(BATCH_SIZE)
                states, actions, next_states, rewards = zip(*transitions)
            
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
                loss.backward()
                optimizer.step()

                td_errors = torch.abs(q_values - targets).squeeze().tolist()

                # Update priorities based on TD errors
                priorities = [(error + 1e-6) * w for error, w in zip(td_errors, weights)] # IS THIS CORRECT?
                prio_replay_buffer.update_priorities(indices, priorities)

            state = next_state

if __name__ == "__main__":
    main()


