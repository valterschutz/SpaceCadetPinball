import time
import sys
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from buffer import PrioritizedReplayBuffer
from cnn import get_device as device
from cnn import load_latest_model
from gamehandler import GameEnvironment


class DQN:
    def __init__(self, action_size=7, gamma=0.99, tau=0.01, lr=1e-3):
        self.ball_cnn = load_latest_model()
        for param in self.ball_cnn.parameters():
            param.requires_grad = False
        self.model = nn.Sequential(
            self.ball_cnn,
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        ).to(device())
        self.target_model = deepcopy(self.model).to(device())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def act(self, state, print_Q=False):
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float).to(device())
            if print_Q:
                print(f"Q values at game start: {self.model(state).cpu().numpy()[0]}")
            action = torch.argmax(self.model(state)).cpu().numpy().item()
        return action

    def update(self, batch, weights=1):
        state, action, reward, next_state, done = batch

        Q_next = self.target_model(next_state).max(dim=1).values
        Q_target = reward + self.gamma * (1 - done) * Q_next
        Q = self.model(state)[torch.arange(len(action)), action.to(torch.long).flatten()]

        assert Q.shape == Q_target.shape, f"{Q.shape}, {Q_target.shape}"

        td_error = torch.abs(Q - Q_target).detach().cpu()
        loss = torch.mean((Q - Q_target)**2 * weights.to(device()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.soft_update(self.target_model, self.model)

        return loss.item(), td_error

    def save(self):
        torch.save(self.model, "gg/agent.pkl")

def evaluate_policy(agent, episodes=5):

    returns = []
    eps = 0.1

    print(f"Evaluating policy for {episodes} episodes with eps={eps}")

    for ep in range(episodes):
        print_Q = True
        env = GameEnvironment(600, 416)
        done, total_reward = False, 0
        state = env.get_state()

        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = model.act(state.unsqueeze(0), print_Q=print_Q)
                print_Q = False
            state, reward = env.step(action)
            done = env.is_done()
            total_reward += reward
        returns.append(total_reward)
        del env
        time.sleep(0.01)
        
    return np.mean(returns), np.std(returns)

def train(model, buffer, batch_size=128,
        eps_max=1, eps_min=0.0, decrease_eps_steps=1000000, test_every_episodes=50):

    env = GameEnvironment(600, 416)

    rewards_total, stds_total = [], []
    loss_count, total_loss = 0, 0

    episodes = 0
    best_reward = -np.inf

    done = False
    state = env.get_state()
    step = 0
    total_reward = 0
    while True:
        if done:
            print(f"Episode {episodes} done... Total reward = {total_reward}")
            total_reward = 0
            del env
            time.sleep(0.01)
            ######## EVALUATE/SAVE START #########
            if episodes % test_every_episodes == 0:
                mean, std = evaluate_policy(model, episodes=1)

                print(f"Step: {step}, Reward mean: {mean:.2f}, Loss: {total_loss / loss_count:.4f}, Eps: {eps}\n")

                if mean > best_reward:
                    best_reward = mean
                    model.save()

                rewards_total.append(mean)
                stds_total.append(std)
            ######## EVALUATE/SAVE END #########
            done = False
            env = GameEnvironment(600, 416)
            state = env.get_state()
            episodes += 1

        eps = eps_max - (eps_max - eps_min) * step / decrease_eps_steps

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = model.act(state.unsqueeze(0))

        next_state, reward = env.step(action)
        total_reward += reward
        done = env.is_done()
        buffer.add((state, action, reward, next_state, int(done)))

        state = next_state

        if step > batch_size:
            batch, weights, tree_idxs = buffer.sample(batch_size)
            loss, td_error = model.update(batch, weights=weights)

            buffer.update_priorities(tree_idxs, td_error.numpy())

            total_loss += loss
            loss_count += 1
        step += 1


    return np.array(rewards_total), np.array(stds_total)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        # Load the model from "./gg/agent.pkl"
        model = torch.load("./gg/agent.pkl")
    else:
        # Create a new DQN model if not loading
        model = DQN(lr=1e-1)

    buffer_size = 30000
    buffer = PrioritizedReplayBuffer(1, buffer_size)
    train(model, buffer, batch_size=8, eps_max=1, eps_min=0.1, decrease_eps_steps=100000, test_every_episodes=10)
