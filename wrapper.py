import subprocess
import os
import ctypes
import numpy as np
import time
from multiprocessing import shared_memory, Semaphore
from PIL import Image
import torch
import torch.nn.functional as F

WIDTH = 600;
HEIGHT = 416;

HIDDEN_TEST = False

# Semaphore is 1 when C has written to pixel array and then set to 0 by python
init_sem = np.array([0], dtype=np.int32)
shm_sem = shared_memory.SharedMemory("sem", create=True, size=init_sem.nbytes)
sem = np.ndarray(init_sem.shape, dtype=np.int32, buffer=shm_sem.buf)
sem[:] = init_sem[:]

# ASCII cheatsheet: 76 = L, 82 = R, 33 = !    All other characters means no input
init_action = np.array([33], dtype=np.uint8)
shm_action = shared_memory.SharedMemory("action", create=True, size=init_action.nbytes)
action = np.ndarray(init_action.shape, dtype=np.uint8, buffer=shm_action.buf)
action[:] = init_action[:]

init_score = np.array([-99], dtype=np.int32)
shm_score = shared_memory.SharedMemory("score", create=True, size=init_score.nbytes)
score = np.ndarray(init_score.shape, dtype=np.int32, buffer=shm_score.buf)
score[:] = init_score[:]

# Contains [xpos, ypos, prev_xpos, prev_ypos, xdir, ydir, speed]
init_ball_info = np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32)
shm_ball_info = shared_memory.SharedMemory("ball_info", create=True, size=init_ball_info.nbytes)
ball_info = np.ndarray(init_ball_info.shape, dtype=np.float32, buffer=shm_ball_info.buf)
ball_info[:] = init_ball_info[:]

init_pixels = np.zeros([HEIGHT//2*WIDTH//2*4], dtype=np.uint8) # 4 for 4 frames
shm_pixels = shared_memory.SharedMemory("pixels", create=True, size=init_pixels.nbytes)
pixels = np.ndarray(init_pixels.shape, dtype=np.uint8, buffer=shm_pixels.buf)
pixels[:] = init_pixels[:]

c_program_path = './bin/SpaceCadetPinball'
process = subprocess.Popen([c_program_path])

def close_shms():
    shm_objs = [shm_sem, shm_action, shm_score, shm_ball_info, shm_pixels]
    for shm_obj in shm_objs:
        shm_obj.close()
        shm_obj.unlink()

def save_pixels():
    for i in range(4):
        save_array = pixels.astype(np.uint8).reshape((4, HEIGHT//2, WIDTH//2))[i,:,:]
        im = Image.fromarray(save_array)
        im.save(f'pixels{i}.png')

def game_loop():
    try:
        save_num = 0
        while True:
            while sem[0] != 4:
                pass
            # Update image tensor
            pixels_reshaped = torch.from_numpy(pixels.reshape(4, HEIGHT//2, WIDTH//2))
            #pixels_reshaped = F.max_pool2d(pixels_reshaped, kernel_size=2, stride=2)
            # Update position tensor
            ball_pos = torch.from_numpy(ball_info[np.array([0, 1, 4, 5])])
            print(f"saving data nr {save_num}")
            if save_num == 500:
                save_pixels()
            torch.save(pixels_reshaped, f"img_data/{str(save_num)}.pth")
            torch.save(ball_pos, f"lbl_data/{str(save_num)}.pth")
            save_num += 1
            # Generate 4 new frames
            sem[:] = init_sem[:]
    finally:
        process.kill()
        close_shms()
        time.sleep(1)
        process.terminate()


def main():
    game_loop()
main()
