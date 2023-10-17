import subprocess
import numpy as np
from multiprocessing import shared_memory
import torch
import random
from cnn import device

class ActionSpace:
    def __init__(self, num_actions):
        self.space = list(range(num_actions))

    def sample(self):
        return random.choice(self.space)

class GameEnvironment:
    def __init__(self, width, height, plotter=None):
        self.width = width
        self.height = height
        self.save_width = width//2
        self.save_height = height//2
        self.frame_id = 0
        self.same_reward_counter = 0
        self.prev_action = None
        self.plotter=plotter
        self.action_space = ActionSpace(4)
        self.left_flipper_up = False
        self.right_flipper_up = False
        self.plunger_down = False

        self.prev_score = np.array([0], dtype=np.int32)
        
        ### INIT SHM
        self.shm_objs = []
        # ball info is not used but needs to be opened atm since we modified the C++ code.
        init_ball_info = np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
        self.ball_info = self.init_shared_memory("ball_info", init_ball_info, np.float32)
        self.init_sem = np.array([0], dtype=np.int32)
        self.sem = self.init_shared_memory("sem", self.init_sem, np.int32)
        init_action = np.array([33], dtype=np.uint8)
        self.action = self.init_shared_memory("action", init_action, np.uint8)
        init_score = np.array([0], dtype=np.int32)
        self.score = self.init_shared_memory("score", init_score, np.int32)
        init_pixels = np.zeros([height//2*width//2*4], dtype=np.uint8)
        self.pixels = self.init_shared_memory("pixels", init_pixels, np.uint8)


        # START GAME AND FAST FORWARD 
        self.process = self.start_game()
        self.fast_forward_frames(550)

    def __del__(self):
        self.process.kill()
        """if self.plotter:
            self.plotter.process_data(f"Score: {self.score[0]}")
        else:
            print(f"Score: {self.score[0]}")"""
        for shm in self.shm_objs:
            shm.close()
            shm.unlink()
        self.process.terminate()

    def init_shared_memory(self, name, data, dtype):
        shm = shared_memory.SharedMemory(name, create=True, size=data.nbytes)
        self.shm_objs.append(shm)
        arr = np.ndarray(data.shape, dtype=dtype, buffer=shm.buf)
        arr[:] = data[:]
        return arr

    def start_game(self):
        c_program_path = './bin/SpaceCadetPinball'
        return subprocess.Popen([c_program_path])

    def fast_forward_frames(self, n):
        for _ in range(n):
            while self.sem[0] == 0:
                pass
            self.frame_id += 1
            self.sem[:] = self.init_sem[:] # Tell C to proceed
        
    def is_done(self):
        if self.sem[0] < 0 or self.ball_info[1]>14.0:
            return True
        return False

    def is_stuck(self):
        if self.same_reward_counter > 500:
            return True
        return False

    def int_to_c_action(self, int_action):
        """Given an integer action representing toggle actions, return a string represting the action we should send to the game."""
        if int_action == 0:
            action = "l" if self.left_flipper_up else "L"
        elif int_action == 1:
            action = "r" if self.right_flipper_up else "R"
        elif int_action == 2:
            action = "." if self.plunger_down else "!"
        elif int_action == 3:
            action = "p"
        else:
            raise Exception(f"Unknown int_action: {int_action}")

        # self.extra_additive_reward = 0 if action in "RrLl!." else 0
        # self.extra_multiplicative_reward = 1.2 if action in "p" else 1
        # self.prev_action = action
        return np.array([ord(action)], dtype=np.uint8)

    def get_reward(self):
        score_diff = self.score[0] - self.prev_score[0]
        if score_diff == 0:
            self.same_reward_counter += 1
            reward = 0
        else:
            self.same_reward_counter = 0
            reward = 1
        reward = torch.tensor(reward, dtype=torch.float32)
        reward.to(device)
        self.prev_score[:] = self.score[:]
        return reward, score_diff

    def get_ball_state(self):
        """Get the current ball state (position and velocity, 4 values)."""
        state = self.ball_info.astype(np.float32)
        state = state[[0,1,4,5]]
        state[0] = state[0] / 10
        state[1] = state[1] / 20
        state = torch.from_numpy(state)
        state = state.to(device)
        return state

    def get_state(self):
        """Get the complete state, including ball position/velocity and flipper/plunger toggle values (7 values)."""
        action_state_tensor = torch.tensor([self.left_flipper_up, self.right_flipper_up, self.plunger_down], dtype=torch.float32).to(device)
        return torch.cat((self.get_ball_state(), action_state_tensor), dim=0)

    def update_toggles(self, action):
        if action == 0:
            self.left_flipper_up = not self.left_flipper_up
        elif action == 1:
            self.right_flipper_up = not self.right_flipper_up
        elif action == 2:
            self.plunger_down = not self.plunger_down

    def step(self, action):
        # Action is one of 4 possible values:
        # 0: toggle left flipper
        # 1: toggle right flipper
        # 2: toggle plunger
        # 3: do nothing
        # if self.frame_id % 100 == 0:
            # print(f"      Frame {self.frame_id}, Score {self.score[0]}")
        self.action[:] = self.int_to_c_action(action)[:]
        # Update our internal representation of flipper and plunger
        self.update_toggles(action)
        while self.sem[0] != 1:
            if self.sem[0] < 0:
                break
        # sem is either < 0 or 4 here
        if self.sem[0] == 1:
            self.sem[:] = self.init_sem[:]
        state = self.get_state()
        reward, score_diff = self.get_reward()
        is_done, is_stuck = self.is_done(), self.is_stuck()
        # Negative reward if we lose
        if is_done or is_stuck:
            reward -= 1
        # Negative reward if we take actions
        if action in range(3):
            reward -= 0.1
        self.frame_id += 1
        return state, reward, score_diff, is_done, is_stuck

# def start_game():
#     return GameEnvironment(600, 416)
