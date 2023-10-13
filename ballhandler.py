import subprocess
import numpy as np
from multiprocessing import shared_memory
import torch
import random
from cnn import get_device

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
        self.action_space = ActionSpace(7)

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
        self.get_state()

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
        if self.sem[0] < 0 or self.same_reward_counter > 500 or self.ball_info[1]>14.0: #bumper bug?
            if self.same_reward_counter > 500:
                print("Bumper bug...", end=" ")
            return True
        return False

    def int_to_c_action(self, int_action):
        # right flipper, left flipper, plunger, tilt left, tilt right, no action
        action =  ["R", "r", "L", "l", "!", ".", "p", "X", "x", "Y", "y"][int_action]
        self.extra_additive_reward = 0 if action in "RrLl!." else 0
        self.extra_multiplicative_reward = 1.2 if action in "p" else 1
        if action == self.prev_action:
            action = "p"
        self.prev_action = action
        return np.array([ord(action)], dtype=np.uint8)

    def get_reward(self):
        reward = self.score[0] - self.prev_score[0]
        if reward == 0:
            self.same_reward_counter += 1
        else:
            self.same_reward_counter = 0
        reward = min(reward, self.extra_multiplicative_reward)
        reward = torch.tensor(reward, dtype=torch.float32)
        reward.to(get_device())
        self.prev_score[:] = self.score[:]
        return reward

    def get_state(self):
        state = self.ball_info.astype(np.float32)
        state = state[[0,1,4,5]]
        state[0] = state[0] / 10
        state[1] = state[1] / 20
        state = torch.from_numpy(state)
        state = state.to(get_device())
        return state

    def step(self, action):
        # if self.frame_id % 100 == 0:
            # print(f"      Frame {self.frame_id}, Score {self.score[0]}")
        self.action[:] = self.int_to_c_action(action)[:]
        # for i in range(self.k_skip):
        while self.sem[0] != 4:
            if self.sem[0] < 0:
                break
        # sem is either < 0 or 4 here
        if self.sem[0] == 4:
            self.sem[:] = self.init_sem[:]
        state, reward = self.get_state(), self.get_reward()
        self.frame_id += 4
        return state, reward


def start_game():
    return GameEnvironment(600, 416)
