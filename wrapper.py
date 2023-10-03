import subprocess
import os
import ctypes
import numpy as np
import time
from multiprocessing import shared_memory, Semaphore
from PIL import Image
import torch
import skimage.measure;

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

init_pixels = np.zeros([HEIGHT*WIDTH*4], dtype=np.uint8) # 4 for rgba
shm_pixels = shared_memory.SharedMemory("pixels", create=True, size=init_pixels.nbytes)
pixels = np.ndarray(init_pixels.shape, dtype=np.uint8, buffer=shm_pixels.buf)
pixels[:] = init_pixels[:]


c_program_path = './bin/SpaceCadetPinball'
process = subprocess.Popen([c_program_path])

def avg_pooling(input_array, pool_size):
    # Assume 3 input channels
    # Perform average pooling
    result = np.zeros((input_array.shape[0] // pool_size[0], input_array.shape[1] // pool_size[1], 3))

    for i in range(0, input_array.shape[0], pool_size[0]):
        for j in range(0, input_array.shape[1], pool_size[1]):
            block = input_array[i:i + pool_size[0], j:j + pool_size[1],:]
            result[i // pool_size[0], j // pool_size[1],:] = np.mean(block, axis=(0,1))

    return result

def close_shms():
    shm_objs = [shm_sem, shm_action, shm_score, shm_ball_info, shm_pixels]
    for shm_obj in shm_objs:
        shm_obj.close()
        shm_obj.unlink()

def print_pixels():
    reshaped_array = pixels.astype(np.uint8).reshape((HEIGHT, WIDTH, 4))
    plot_ascii_table = True
    if plot_ascii_table:
        ascii_characters = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
        ascii_art = np.vectorize(lambda x: ascii_characters[x])(reshaped_array[::10, ::10, 0]//26)
        ascii_art = '\n'.join([''.join(row) for row in ascii_art])
        print(ascii_art)

def save_pixels(filename):
    save_array = pixels.astype(np.uint8).reshape((HEIGHT, WIDTH, 4))[:,:,:3]
    im = Image.fromarray(save_array)
    pooled_im = im.resize((WIDTH//2,HEIGHT//2))
    pooled_im.save('pixels.png')


def chartoarray(c):
    return np.array([ord(c)], dtype=np.uint8)

def game_loop(every_frame_fun, i_frame_funs, end_fun):
    try:
        j = 0
        while True:
            every_frame_fun()
            for (i,i_frame_fun) in i_frame_funs:
                if j % i == 0:
                    i_frame_fun()
            sem[:] = init_sem[:]
            time.sleep(0.01)
            j = j + 1
    # except Exception as e:
    #     print(e)
    #     pass
    finally:
        process.kill()
      
        end_fun()

        time.sleep(1)
        process.terminate()


def main():
    # Run C code with varying input
    def every_frame_fun():
        action[:] = chartoarray('a')
    actions = ['L','l','R','r','!','.','p']
    ai = 0
    img_tensor = torch.zeros(4,3,HEIGHT//2,WIDTH//2, dtype=torch.uint8)
    pos_tensor = torch.zeros(4,2,dtype=torch.float32)
    def add_data_to_tensors(i):
        # Update image tensor
        pixels_reshaped = pixels.reshape(HEIGHT, WIDTH, 4)[:,:,:3]
        thing1 = skimage.measure.block_reduce(pixels_reshaped, (2,2,1), np.mean)
        thing2 = torch.from_numpy(thing1).permute(2,0,1)
        img_tensor[i,:,:,:] = thing2

        # Update position tensor
        pos_tensor[i,:] = torch.from_numpy(ball_info[:2])
    def i_frame_fun():
        nonlocal ai
        action[:] = chartoarray(actions[ai])
        ai = (ai+1) % 7
    def end_fun():
        print_pixels()
        print(f"\n                    SCORE: {score[0]} ")
        close_shms()
    save_num = 0
    def save_fun():
        nonlocal save_num
        print(f"saving data nr {save_num}")
        torch.save(img_tensor, f"img_data/{str(save_num)}.pth")
        torch.save(pos_tensor, f"lbl_data/{str(save_num)}.pth")
        save_num += 1
    game_loop(every_frame_fun=every_frame_fun, i_frame_funs=[
        (5,lambda: add_data_to_tensors(0)),
        (6,lambda: add_data_to_tensors(1)),
        (7,lambda: add_data_to_tensors(2)),
        (8,lambda: add_data_to_tensors(3)),
        (9,save_fun),
    ], end_fun=end_fun)

main()
