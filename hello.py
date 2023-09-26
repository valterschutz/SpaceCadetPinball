import subprocess
import numpy as np
from multiprocessing import shared_memory, Semaphore

WIDTH = 100;
HEIGHT = 100;

init_pixels = np.zeros(HEIGHT, WIDTH)
init_score = np.array([0])
shm_pixels = shared_memory.SharedMemory("pixels", create=True, size=a.nbytes)
shm_score = shared_memory.SharedMemory("score", create=True, size=a.nbytes)
shm_action = shared_memory.SharedMemory("action", create=True, size=a.nbytes)
b = np.ndarray(a.shape, dtype=np.double, buffer=shm.buf)
b[:] = a[:]  # Copy the original data into shared memory

# Run C code
c_program_path = './hello'  # Replace with the actual path to your C program
process = subprocess.Popen([c_program_path])
process.wait()

print(b)


shm.close()
shm.unlink()
