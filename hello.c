#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    int fd = shm_open("/shm", O_RDWR, 0666); 
    if (fd == -1) {
        perror("shm_open");
        exit(1);
    }
    
    // Map the shared memory into the address space
    void *ptr = mmap(NULL, sizeof(double) * 6, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    
    // Access and modify the shared memory
    double *shared_data = (double *)ptr;
    for (int i = 0; i < 6; i++) {
        shared_data[i] = 2 * shared_data[i];  // Modify the data as needed
    }
    
    // Unmap and close the shared memory
    munmap(ptr, sizeof(double) * 6);
    close(fd);
    
    return EXIT_SUCCESS;
}
