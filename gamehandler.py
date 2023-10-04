import subprocess
import os
import ctypes
import numpy as np
import time
from multiprocessing import shared_memory, Semaphore
from PIL import Image
import torch
import skimage.measure;
from train_cnn import get_device


class GameEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.save_width = width//2
        self.save_height = height//2
        self.frame_id = 0

        self.internal_state = torch.zeros((3, self.save_width, self.save_height, 4))
        self.prev_score = np.array([0], dtype=np.int32)
        self.external_state = None
        
        ### INIT SHM
        self.shm_objs = []
        # ball info is not used but needs to be opened atm since we modified the C++ code.
        init_ball_info = np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
        ball_info = self.init_shared_memory("ball_info", init_ball_info, np.float32)
        self.init_sem = np.array([0], dtype=np.int32)
        self.sem = self.init_shared_memory("sem", self.init_sem, np.int32)
        init_action = np.array([33], dtype=np.uint8)
        self.action = self.init_shared_memory("action", init_action, np.uint8)
        init_score = np.array([0], dtype=np.int32)
        self.score = self.init_shared_memory("score", init_score, np.int32)
        init_pixels = np.zeros([height*width*4], dtype=np.uint8)
        self.pixels = self.init_shared_memory("pixels", init_pixels, np.uint8)

        # START GAME AND FAST FORWARD 
        self.process = self.start_game()
        #self.fast_forward_frames(500)
        self.get_external_state()
        self.external_state.to(get_device())

    def __del__(self):
        self.process.kill()
        for shm in self.shm_objs:
            shm.close()
            shm.unlink()
        time.sleep(0.1)
        self.process.terminate()
        print("      Shared memory unlinked")

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
        print(f"   Fast forwarding {n} frames...")
        for _ in range(n):
            self.sem[:] = self.init_sem[:] # Tell C to proceed
            self.frame_id += 1
            time.sleep(0.001)
        
    def is_done(self):
        #if self.sem[0] == -9:
        if self.frame_id > 15000:
            return True
        return False

    def update_internal_state(self):
        tensor_idx = self.frame_id % 4
        reshaped_pixels = self.pixels.reshape(self.height, self.width, 4)[:,:,:3]
        reshaped_pixels = skimage.measure.block_reduce(reshaped_pixels, (2,2,1), np.mean)
        self.internal_state[:,:,:,3] = self.internal_state[:,:,:,2] # put the last frame first #TODO: do this in a more efficient way.
        self.internal_state[:,:,:,2] = self.internal_state[:,:,:,1] # put the last frame first
        self.internal_state[:,:,:,1] = self.internal_state[:,:,:,0] # put the last frame first
        self.internal_state[:,:,:,0] = torch.from_numpy(reshaped_pixels).permute(2,1,0) #replace the first frame
        self.sem[:] = self.init_sem[:] # Tell C to proceed
        self.frame_id += 1

    def int_to_c_action(self, int_action):
        # right flipper, left flipper, plunger, tilt left, tilt right, no action
        action =  ["R", "r", "L", "l", "!", ".", "p", "X", "x", "Y", "y"][int_action]
        return np.array([ord(action)], dtype=np.uint8)

    def get_reward(self):
        reward = torch.tensor(self.score[0] - self.prev_score[0], dtype=torch.int32)
        self.prev_score[:] = self.score[:]
        reward.to(get_device())
        return reward

    def get_external_state(self):
        # Transform 1
        self.external_state = self.internal_state.float() / 255.0
        # Transform 2
        self.external_state = self.external_state.view(
            self.external_state.shape[0]*self.external_state.shape[3],
            self.external_state.shape[1],
            self.external_state.shape[2]
        )
        self.external_state.to(get_device())
        return self.external_state

    def step(self, action):
        if self.frame_id % 100 == 0:
            print(f"      Frame {self.frame_id}")
        # Simulate the game step and return the next internal_state and reward
        self.action[:] = self.int_to_c_action(action)[:]
        self.update_internal_state()
        return self.get_external_state(), self.get_reward()


def start_game():
    return GameEnvironment(600, 416)
"""
game = start_game()
game.step(0)
game.step(0)
game.step(0)
game.step(0)
game.step(0)
"""
