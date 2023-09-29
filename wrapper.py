import subprocess
import os
import ctypes
import numpy as np
import time
from multiprocessing import shared_memory, Semaphore

WIDTH = 600;
HEIGHT = 416;

def main():
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

    init_pixels = np.zeros([HEIGHT*WIDTH*4], dtype=np.uint8) # 4 for rgba
    shm_pixels = shared_memory.SharedMemory("pixels", create=True, size=init_pixels.nbytes)
    pixels = np.ndarray(init_pixels.shape, dtype=np.uint8, buffer=shm_pixels.buf)
    pixels[:] = init_pixels[:]

    # Run C code with varying input
    c_program_path = './bin/SpaceCadetPinball'
    process = subprocess.Popen([c_program_path])
    for i in range(3):
        sem[:] = init_sem[:]
        time.sleep(1)

    #time.sleep(2)
    #action[:] = np.array([76], dtype=np.uint8)[:]
    #time.sleep(2)

    # Extract pixels and score
    reshaped_array = pixels.astype(np.uint8).reshape((HEIGHT, WIDTH, 4))
    plot_ascii_table = True
    if plot_ascii_table:
        ascii_characters = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
        ascii_art = np.vectorize(lambda x: ascii_characters[x])(reshaped_array[::10, ::10, 0]//26)
        ascii_art = '\n'.join([''.join(row) for row in ascii_art])
        print(ascii_art)

    print(f"\n                    SCORE: {score[0]} ")
    
    # Kill the game
    process.terminate()
    time.sleep(1)
    process.kill()
    
    # Unlink our shared memory
    shm_objs = [shm_sem, shm_action, shm_score, shm_ball_info, shm_pixels]
    for shm_obj in shm_objs:
        shm_obj.unlink()

main()
