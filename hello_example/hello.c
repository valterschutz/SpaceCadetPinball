#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

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
    int *sem = mmap(NULL, sizeof(double) * SEM_BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, sem_fd, 0);
    double *score = mmap(NULL, sizeof(double) * SCORE_BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, score_fd, 0);
    double *pixels = mmap(NULL, sizeof(double) * PIXELS_BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, pixels_fd, 0);
    if (sem == MAP_FAILED | score == MAP_FAILED | score == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

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
    munmap(sem, sizeof(double) * SEM_BUF_SIZE);
    munmap(score, sizeof(double) * SCORE_BUF_SIZE);
    munmap(pixels, sizeof(double) * PIXELS_BUF_SIZE);
    close(sem_fd);
    close(score_fd);
    close(pixels_fd);
    return EXIT_SUCCESS;
}
