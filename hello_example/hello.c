#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>

#define SCORE_BUF_SIZE 1
#define SEM_BUF_SIZE 1
#define DISP_WIDTH_ENV "DISP_WIDTH"
#define DISP_HEIGHT_ENV "DISP_HEIGHT"
#define WIDTH (atoi(getenv(DISP_WIDTH_ENV)))
#define HEIGHT (atoi(getenv(DISP_HEIGHT_ENV)))
#define PIXELS_BUF_SIZE (HEIGHT * WIDTH)

int main(int argc, char *argv[])
{
    // Init shared memory
    int sem_fd = shm_open("/sem", O_RDWR, 0666); 
    int score_fd = shm_open("/score", O_RDWR, 0666); 
    int pixels_fd = shm_open("/pixels", O_RDWR, 0666); 
    if (sem_fd == -1 | score_fd == -1 | pixels_fd == -1) {
        perror("shm_open");
        exit(1);
    }

    // Map the shared memory into the address space
    void *sem_ptr = mmap(NULL, sizeof(double) * SEM_BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, sem_fd, 0);
    void *score_ptr = mmap(NULL, sizeof(double) * SCORE_BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, score_fd, 0);
    void *pixels_ptr = mmap(NULL, sizeof(double) * PIXELS_BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, pixels_fd, 0);
    if (sem_ptr == MAP_FAILED | score_ptr == MAP_FAILED | score_ptr == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    // Access and modify the shared memory
    int *sem = (int *)sem_ptr;
    double *score = (double *)score_ptr;
    double *pixels = (double *)pixels_ptr;

    for (int j = 0; j < 10; j++) {
	while (1){
	    if (*sem) {
		*sem = 0;
		printf("From c: %d\n", j);
		for (int i = 0; i < SCORE_BUF_SIZE; i++) {
		    score[i] += 5.0;
		}
		for (int i = 0; i < PIXELS_BUF_SIZE; i++) {
		    pixels[i] += 7.0;
		}
		break;
	    }
	}
    }

    // Unmap and close the shared memory
    munmap(sem_ptr, sizeof(double) * SEM_BUF_SIZE);
    munmap(score_ptr, sizeof(double) * SCORE_BUF_SIZE);
    munmap(pixels_ptr, sizeof(double) * PIXELS_BUF_SIZE);
    close(sem_fd);
    close(score_fd);
    close(pixels_fd);
    return EXIT_SUCCESS;
}
