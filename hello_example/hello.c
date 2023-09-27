#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>

#define SCORE_BUF_SIZE 1
#define DISP_WIDTH_ENV "DISP_WIDTH"
#define DISP_HEIGHT_ENV "DISP_HEIGHT"
#define WIDTH (atoi(getenv(DISP_WIDTH_ENV)))
#define HEIGHT (atoi(getenv(DISP_HEIGHT_ENV)))
#define PIXELS_BUF_SIZE (HEIGHT * WIDTH)

int main(int argc, char *argv[])
{
    /////// SEMAPHORE ///////
    sem_t *sem = sem_open("/semaphore", O_CREAT, 0644, 1);
    if (sem == SEM_FAILED) {
        perror("sem_open");
        return 1;
    }
    // Init shared memory
    int score_fd = shm_open("/score", O_RDWR, 0666); 
    if (score_fd == -1) {
        perror("shm_open");
        exit(1);
    }
    int pixels_fd = shm_open("/pixels", O_RDWR, 0666); 
    if (pixels_fd == -1) {
        perror("shm_open");
        exit(1);
    }
    // Map the shared memory into the address space
    void *score_ptr = mmap(NULL, sizeof(double) * SCORE_BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, score_fd, 0);
    if (score_ptr == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    void *pixels_ptr = mmap(NULL, sizeof(double) * PIXELS_BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, pixels_fd, 0);
    if (pixels_ptr == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    // Access and modify the shared memory
    double *score = (double *)score_ptr;
    double *pixels = (double *)pixels_ptr;
    for (int j = 0; j < 3; j++) {
	    printf("asa");
	    //sem_wait(sem); # TODO
	    for (int i = 0; i < SCORE_BUF_SIZE; i++) {
		score[i] += 5.0;
	    }
	    for (int i = 0; i < PIXELS_BUF_SIZE; i++) {
		 pixels[i] += 7.0;
	    }
	    //sem_post(sem); #TODO
    }

    // Unmap and close the shared memory
    munmap(score_ptr, sizeof(double) * SCORE_BUF_SIZE);
    munmap(pixels_ptr, sizeof(double) * PIXELS_BUF_SIZE);
    close(score_fd);
    close(pixels_fd);
    
    // Close semaphore
    sem_close(sem);
    return EXIT_SUCCESS;
}
