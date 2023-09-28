import subprocess
import os
import ctypes
import numpy as np
import time
from multiprocessing import shared_memory, Semaphore

WIDTH = 600;
HEIGHT = 416;

def main():
    """
    init_sem = np.array([1], dtype=np.uint8)
    shm_sem = shared_memory.SharedMemory("sem", create=True, size=init_sem.nbytes)
    sem = np.ndarray(init_sem.shape, dtype=np.uint8, buffer=shm_sem.buf)
    sem[:] = init_sem[:]

    """
    # 76 = L, 82 = R, 33 = !
    init_action = np.array([82], dtype=np.uint8)
    shm_action = shared_memory.SharedMemory("action", create=True, size=init_action.nbytes)
    action = np.ndarray(init_action.shape, dtype=np.uint8, buffer=shm_action.buf)
    action[:] = init_action[:]

    init_score = np.array([-99], dtype=np.int32)
    shm_score = shared_memory.SharedMemory("score", create=True, size=init_score.nbytes)
    score = np.ndarray(init_score.shape, dtype=np.int32, buffer=shm_score.buf)
    score[:] = init_score[:]

    init_pixels = np.zeros([HEIGHT*WIDTH*4], dtype=np.uint8) # 4 for rgba
    shm_pixels = shared_memory.SharedMemory("pixels", create=True, size=init_pixels.nbytes)
    pixels = np.ndarray(init_pixels.shape, dtype=np.uint8, buffer=shm_pixels.buf)
    pixels[:] = init_pixels[:]

    # Run C code
    c_program_path = './bin/SpaceCadetPinball'
    process = subprocess.Popen([c_program_path])
    time.sleep(1)
    action[:] = np.array([76], dtype=np.uint8)[:]
    time.sleep(1)
    # Extract pixels and score
    reshaped_array = pixels.astype(np.uint8).reshape((HEIGHT, WIDTH, 4))
    plot_ascii_table = True
    if plot_ascii_table:
        ascii_characters = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
        ascii_art = np.vectorize(lambda x: ascii_characters[x])(reshaped_array[::10, ::10, 0]//26)
        ascii_art = '\n'.join([''.join(row) for row in ascii_art])
        print(ascii_art)

    print(f"\n                    SCORE: {score[0]} ")
    
    #Kill the game
    process.terminate()
    time.sleep(1)
    process.kill()

    """
    shm_sem.close()
    shm_sem.unlink()
    """
    # close shm stuff
    shm_action.close()
    shm_action.unlink()

    shm_score.close()
    shm_score.unlink()

    shm_pixels.close()
    shm_pixels.unlink()


main()
