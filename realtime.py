import numpy as np
from gamehandler import GameEnvironment
from gg import DQN
import pygame

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
        self.colors = [(255, 0, 50), (255, 50, 0), (50, 255, 0), (0, 255, 50), (0, 100, 255), (100, 0, 255), (150, 150, 150)]
        self.labels = ["Right flipper", "Left flipper", "Plunger"]
        self.sublabels = ["Up", "Down", "Up", "Down", "Pull", "Release", "Wait"]
        self.smallfont = pygame.font.Font(None, 25)
        self.mediumfont = pygame.font.Font(None, 35)
        self. largefont = pygame.font.Font(None, 55)

        # Control animation speed
        self.clock = pygame.time.Clock()
        self.fps = 24


    def run_animation(self):
        running = True
        for Qs in evaluate_policy(self.agent) and running:
            # Check if user trying to close window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Background and header
            self.screen.fill((25, 25, 25))
            header_text = self.largefont.render("Q-values", True, (255, 255, 255))
            self.screen.blit(header_text, (330, 50))
            
            # Right flipper, Left flipper, Plunger
            for j, label in enumerate(self.labels):
                x = 110 + j * 210
                label_text = self.mediumfont.render(label, True, (255, 255, 255))
                self.screen.blit(label_text, (x, 450))

            # Up, down and moving rectangles.
            for j, value in enumerate(Qs):
                x = 100 + j * 100
                y = 400 - value * 300
                w = 50
                h = value * 300
                pygame.draw.rect(self.screen, self.colors[j], (x, y, w, h))

                # Draw labels
                if self.sublabels[j] == "Up":
                    x+=15
                elif self.sublabels[j] in ("Pull", "Wait"):
                    x += 7
                sublabel_text = self.smallfont.render(self.sublabels[j], True, (255, 255, 255))
                self.screen.blit(sublabel_text, (x, 405))

            # Control speed
            pygame.display.flip()
            clock.tick(fps)

        pygame.quit()

def get_qs(agent, state):
    with torch.no_grad():
        state = torch.as_tensor(state, dtype=torch.float).to(device())
        return agent.model(state.unsqueeze(0)).cpu().numpy()[0]

def evaluate_policy(agent, episodes=None):
    print("Playing game...")
    # Loop forever if no episodes given
    if episodes == None:
        iter = itertools.count(start=0, step=1)
    else:
        iter = range(episodes)
    
    eps = 0.3
    Qs = []

    for ep in iter:
        env = GameEnvironment(600, 416)
        done, total_reward = False, 0
        state = env.get_state()
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = agent.act(state.unsqueeze(0))
            yield get_qs(agent, state)
            state, reward = env.step(action)
            done = env.is_done()
        del env
        time.sleep(0.1)
    return None


model_directory = "pickles"
model_files = glob.glob(os.path.join(model_directory, "model_*.pkl"))
model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
if len(sys.argv) > 1:
    latest_model_file = sys.argv[1]
else:
    latest_model_file = model_files[0]
with open(latest_model_file, "rb") as file:
    agent = pickle.load(file)
    agent.model = agent.model.to(device())
print(f"Loaded {latest_model_file}...")

