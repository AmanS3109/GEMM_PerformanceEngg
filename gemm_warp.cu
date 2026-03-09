#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024

// Tile Sizes
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 8

// Thread coarsening sizes (Each thread computes an 8x8 block)
#define THREAD_M 8
#define THREAD_N 8

// Macro for catching CUDA errors
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error:\nFile: %s\nLine: %d\nError: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void gemm_register_tiled(float *A, float *B, float *C, int n)
{
    // Block row and col
    int rowOffset = blockIdx.y * BLOCK_M;
    int colOffset = blockIdx.x * BLOCK_N;

    // Shared memory tiles
    __shared__ float As[BLOCK_M][BLOCK_K];
    __shared__ float Bs[BLOCK_K][BLOCK_N];

    // Thread-local registers to hold the 8x8 computed block
    float accum[THREAD_M][THREAD_N] = {0.0f};

    // Thread-local registers to cache shared memory reads
    float a_reg[THREAD_M];
    float b_reg[THREAD_N];

    // 1D Thread ID within the block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Loop over the K dimension
    for (int k = 0; k < n; k += BLOCK_K)
    {
        // -----------------------------------------------------------
        // 1. Collaborative Loading into Shared Memory
        // 64 threads work together to load 512 elements of A and B
        // -----------------------------------------------------------
        
        // Load As (64x8)
        int a_row = tid / BLOCK_K;
        int a_col = tid % BLOCK_K;
        int a_stride = (blockDim.x * blockDim.y) / BLOCK_K;
        for (int i = 0; i < BLOCK_M; i += a_stride) {
            As[a_row + i][a_col] = A[(rowOffset + a_row + i) * n + (k + a_col)];
        }

        // Load Bs (8x64)
        int b_row = tid / BLOCK_N;
        int b_col = tid % BLOCK_N;
        int b_stride = (blockDim.x * blockDim.y) / BLOCK_N;
        for (int i = 0; i < BLOCK_K; i += b_stride) {
            Bs[b_row + i][b_col] = B[(k + b_row + i) * n + (colOffset + b_col)];
        }

        __syncthreads();

        // -----------------------------------------------------------
        // 2. Compute Phase (Register Tiling)
        // -----------------------------------------------------------
        for (int dotIdx = 0; dotIdx < BLOCK_K; ++dotIdx)
        {
            // Pull data from shared memory into registers
            for (int i = 0; i < THREAD_M; ++i) {
                a_reg[i] = As[threadIdx.y * THREAD_M + i][dotIdx];
            }
            for (int j = 0; j < THREAD_N; ++j) {
                b_reg[j] = Bs[dotIdx][threadIdx.x * THREAD_N + j];
            }

            // Perform 64 Fused Multiply-Adds (FMAs) directly in registers
            for (int i = 0; i < THREAD_M; ++i) {
                for (int j = 0; j < THREAD_N; ++j) {
                    accum[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // -----------------------------------------------------------
    // 3. Write Results Back to Global Memory
    // -----------------------------------------------------------
    for (int i = 0; i < THREAD_M; ++i)
    {
        for (int j = 0; j < THREAD_N; ++j)
        {
            int global_row = rowOffset + threadIdx.y * THREAD_M + i;
            int global_col = colOffset + threadIdx.x * THREAD_N + j;
            if (global_row < n && global_col < n) {
                C[global_row * n + global_col] = accum[i][j];
            }
        }
    }
}

int main()
{
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N*N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;

    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // A 64x64 block handled by threads computing 8x8 each means 8x8 threads per block
    dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);
    dim3 grid(N / BLOCK_N, N / BLOCK_M);

    // WARM-UP RUN (To negate initialization overhead)
    gemm_register_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // BENCHMARK RUN
    int iterations = 100;
    CHECK_CUDA(cudaEventRecord(start));

    for(int i = 0; i < iterations; i++) {
        gemm_register_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    double avg_ms = ms / iterations;
    double seconds = avg_ms / 1000.0;
    double flops = 2.0 * (double)N * (double)N * (double)N;
    double gflops = flops / (seconds * 1e9);

    printf("Average Time per GEMM: %f seconds\n", seconds);
    printf("GFLOPS: %f\n", gflops);
    printf("C[0] = %f\n", h_C[0]);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
