import numpy as np
import torch
from ballhandler import GameEnvironment
from ballgg import DQN
import pygame
import sys
import glob
import os
import pickle
import time
from cnn import device
import itertools
import random

# Replace this with your actual matrix
"""
matrix = np.load("qs_tmp.npy")
matrix = matrix / (3*np.mean(matrix))
n = matrix.shape[0
]
"""
class RealtimePlotter():
    def __init__(self, agent):
        # Window
        pygame.init()
        self.screen = pygame.display.set_mode((840, 550))
        pygame.display.set_caption("SpaceCadet live Q-value barplot")

        # Colors & text
        self.colors = np.array([(255, 0, 50), (255, 50, 0), (50, 255, 0), (0, 255, 50), (0, 100, 255), (100, 0, 255), (150, 150, 150)])
        self.labels = ["Right flipper", "Left flipper", "Plunger"]
        self.sublabels = ["Up", "Down", "Up", "Down", "Pull", "Release", "Wait"]
        self.smallfont = pygame.font.Font(None, 25)
        self.mediumfont = pygame.font.Font(None, 35)
        self. largefont = pygame.font.Font(None, 55)

        # Control animation speed
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.agent = agent


    def run_animation(self):
        running = True
        for Qs in evaluate_policy(self.agent):
            # Qs = Qs / np.max(Qs)
            Qs = Qs / 3
            Qs_argmax = np.argmax(Qs)

            # Reshuffle Q and labels to show left flipper first
            Qs[0], Qs[1] = Qs[1], Qs[0]
            labels = self.labels.copy()
            labels[0], labels[1] = labels[1], labels[0]
            # Check if user trying to close window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Background and header
            self.screen.fill((25, 25, 25))
            header_text = self.largefont.render("Q-values", True, (255, 255, 255))
            self.screen.blit(header_text, (330, 50))
            
            # Right flipper, Left flipper, Plunger
            for j, label in enumerate(labels):
                x = 110 + j * 210
                label_text = self.mediumfont.render(label, True, (255, 255, 255))
                self.screen.blit(label_text, (x, 450))

            # Up, down and moving rectangles.
            for j, value in enumerate(Qs):
                x = 100 + j * 100
                y = 400 - value * 300
                w = 50
                h = value * 300
                pygame.draw.rect(self.screen, self.colors[j] / (1 if j == Qs_argmax else 2), (x, y, w, h))

                # Draw labels
                if self.sublabels[j] == "Up":
                    x+=15
                elif self.sublabels[j] in ("Pull", "Wait"):
                    x += 7
                sublabel_text = self.smallfont.render(self.sublabels[j], True, (255, 255, 255))
                self.screen.blit(sublabel_text, (x, 405))

            # Control speed
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()

def get_qs(agent, state):
    with torch.no_grad():
        state = torch.as_tensor(state, dtype=torch.float).to(device)
        return agent.model(state.unsqueeze(0)).cpu().numpy()[0]

def evaluate_policy(agent, episodes=None):
    print("Playing game...")
    # Loop forever if no episodes given
    if episodes == None:
        iter = itertools.count(start=0, step=1)
    else:
        iter = range(episodes)
    
    eps = 0.1
    Qs = []

    for ep in iter:
        env = GameEnvironment(600, 416)
        done, total_reward = False, 0
        state = agent.get_state(env)
        while not done:
            action = agent.act(env, state.unsqueeze(0), eps)
            yield get_qs(agent, state)
            state, reward = agent.step(env, action)
            done = env.is_done()
        del env
        time.sleep(0.1)
    return None


model_name = sys.argv[1]
# model_filename = f"pickles/model_{model_name}.pkl"
weights_path = f"weights/{model_name}.pth"
# with open(model_filename, "rb") as file:
#     agent = pickle.load(file)
#     agent.model = agent.model.to(device)
#     agent.target_model = agent.target_model.to(device)
agent = DQN()
agent.model.load_state_dict(torch.load(weights_path))
print(f"Loaded {model_name}...")

realtime = RealtimePlotter(agent)
realtime.run_animation()
