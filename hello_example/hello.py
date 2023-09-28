import subprocess
import os
import ctypes
import numpy as np
from multiprocessing import shared_memory, Semaphore

WIDTH = 128;
HEIGHT = 128;

def make():
    compile_command = ["gcc", "hello.c", "-o", "hello", "-lrt", "-pthread"]
    try:
        subprocess.run(compile_command, check=True)
        print("Compilation successful")
    except subprocess.CalledProcessError:
        print("Compilation failed")

def set_environ():
    os.environ["DISP_WIDTH"] = str(WIDTH)
    os.environ["DISP_HEIGHT"] = str(HEIGHT)

def main():
    make()

    # set this so we can use it in the c program.
    set_environ()


    init_sem = np.array([1], dtype=np.int32)
    shm_sem = shared_memory.SharedMemory("sem", create=True, size=init_sem.nbytes)
    sem = np.ndarray(init_sem.shape, dtype=np.int32, buffer=shm_sem.buf)
    sem[:] = init_sem[:]

    init_score = np.array([0])
    shm_score = shared_memory.SharedMemory("score", create=True, size=init_score.nbytes)
    score = np.ndarray(init_score.shape, dtype=np.double, buffer=shm_score.buf)
    score[:] = init_score[:]
    
    init_pixels = np.zeros([HEIGHT*WIDTH])
    shm_pixels = shared_memory.SharedMemory("pixels", create=True, size=init_pixels.nbytes)
    pixels = np.ndarray(init_pixels.shape, dtype=np.double, buffer=shm_pixels.buf)
    pixels[:] = init_pixels[:]

    # Run C code
    c_program_path = './hello'
    process = subprocess.Popen([c_program_path])
    for i in range(10):
        while (True):
            if sem[0]==0:
                print(f"Hi from Python: {i}")
                sem[0]=1
                break;
    process.wait()

    print(f"Pixels after hello finished: {pixels}")
    print(f"Score after hello finished: {score}")

    shm_score.close()
    shm_score.unlink()

    shm_sem.close()
    shm_sem.unlink()

    shm_pixels.close()
    shm_pixels.unlink()


main()
