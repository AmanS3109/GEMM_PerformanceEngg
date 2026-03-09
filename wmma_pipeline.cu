#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define N 1024

// Tensor Core 16x16x16 math tile
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block tile size (32x32). Handled by 4 Warps (128 threads)
#define BLOCK_M 32
#define BLOCK_N 32
#define BLOCK_K 32

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error:\nFile: %s\nLine: %d\nError: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void init_half_matrices(half *A, half *B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = __float2half(1.0f);
        B[idx] = __float2half(1.0f);
    }
}

__global__ void gemm_wmma_double_buffered(half *A, half *B, float *C, int n) {
    // DOUBLE BUFFERING: Notice the extra [2] dimension
    __shared__ half As[2][BLOCK_M][BLOCK_K];
    __shared__ half Bs[2][BLOCK_K][BLOCK_N];

    int warpId = threadIdx.x / 32;
    int warpRow = (warpId / 2) * WMMA_M;
    int warpCol = (warpId % 2) * WMMA_N;

    int blockRow = blockIdx.y * BLOCK_M;
    int blockCol = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // ==========================================
    // PROLOGUE: Load the very first tile into Buffer 0
    // ==========================================
    for (int i = tid; i < BLOCK_M * BLOCK_K; i += num_threads) {
        int row = i / BLOCK_K;
        int col = i % BLOCK_K;
        As[0][row][col] = A[(blockRow + row) * n + col];
    }
    for (int i = tid; i < BLOCK_K * BLOCK_N; i += num_threads) {
        int row = i / BLOCK_N;
        int col = i % BLOCK_N;
        Bs[0][row][col] = B[row * n + (blockCol + col)];
    }
    __syncthreads();

    int write_idx = 1;
    int read_idx = 0;

    // ==========================================
    // MAIN PIPELINE
    // ==========================================
    for (int k = 0; k < n; k += BLOCK_K) {
        
        // 1. Asynchronously load the NEXT tile into the 'write' buffer
        if (k + BLOCK_K < n) {
            for (int i = tid; i < BLOCK_M * BLOCK_K; i += num_threads) {
                int row = i / BLOCK_K;
                int col = i % BLOCK_K;
                As[write_idx][row][col] = A[(blockRow + row) * n + (k + BLOCK_K + col)];
            }
            for (int i = tid; i < BLOCK_K * BLOCK_N; i += num_threads) {
                int row = i / BLOCK_N;
                int col = i % BLOCK_N;
                Bs[write_idx][row][col] = B[(k + BLOCK_K + row) * n + (blockCol + col)];
            }
        }

        // 2. Compute the CURRENT tile from the 'read' buffer
        for (int step = 0; step < BLOCK_K / WMMA_K; step++) {
            wmma::load_matrix_sync(a_frag, &As[read_idx][warpRow][step * WMMA_K], BLOCK_K);
            wmma::load_matrix_sync(b_frag, &Bs[read_idx][step * WMMA_K][warpCol], BLOCK_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Wait for both the compute and the memory loads to finish before swapping
        __syncthreads();

        // Swap the buffers using a bitwise XOR
        write_idx ^= 1;
        read_idx ^= 1;
    }

    // ==========================================
    // EPILOGUE: Store the results
    // ==========================================
    int globalRow = blockRow + warpRow;
    int globalCol = blockCol + warpCol;
    wmma::store_matrix_sync(C + globalRow * n + globalCol, c_frag, n, wmma::mem_row_major);
}

int main() {
    size_t float_bytes = N * N * sizeof(float);
    size_t half_bytes = N * N * sizeof(half);

    float *h_C = (float*)malloc(float_bytes);
    half *d_A, *d_B;
    float *d_C;

    CHECK_CUDA(cudaMalloc(&d_A, half_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, half_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, float_bytes));

    int threadsPerBlockInit = 256;
    int blocksPerGridInit = (N * N + threadsPerBlockInit - 1) / threadsPerBlockInit;
    init_half_matrices<<<blocksPerGridInit, threadsPerBlockInit>>>(d_A, d_B, N * N);
    CHECK_CUDA(cudaDeviceSynchronize());

    dim3 block(128); 
    dim3 grid(N / BLOCK_N, N / BLOCK_M);

    // Warm-up
    gemm_wmma_double_buffered<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int iterations = 100;
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        gemm_wmma_double_buffered<<<grid, block>>>(d_A, d_B, d_C, N);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaMemcpy(h_C, d_C, float_bytes, cudaMemcpyDeviceToHost));

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
    free(h_C);

    return 0;
}
