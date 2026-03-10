#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void multiply_matrices(int N, double *A, double *B, double *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) {
                // Accessing A[i][k] and B[k][j]
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int N = 1024; // Matrix size (N x N)
    size_t size = N * N * sizeof(double);
    
    double *A = malloc(size);
    double *B = malloc(size);
    double *C = malloc(size);

    // Initialize matrices with dummy data
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0; B[i] = 2.0;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    multiply_matrices(N, A, B, C);

    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate time in seconds
    double time_taken = (end.tv_sec - start.tv_sec) + 
                        (end.tv_nsec - start.tv_nsec) / 1e9;

    // Calculate GFLOPS
    // Matrix mult performs N^3 multiplications and N^3 additions = 2 * N^3 operations
    double ops = 2.0 * N * N * N;
    double gflops = (ops / time_taken) / 1e9;

    printf("Time: %f seconds\n", time_taken);
    printf("GFLOPS: %f\n", gflops);

    free(A); free(B); free(C);
    return 0;
}