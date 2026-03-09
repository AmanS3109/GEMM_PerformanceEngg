#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // Required for 'half' precision
#include <mma.h>       // Required for Tensor Cores

using namespace nvcuda;

#define N 1024

// Tensor Cores operate on fixed tile sizes. 16x16x16 is the standard.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error:\nFile: %s\nLine: %d\nError: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Helper kernel to initialize FP16 matrices on the device
__global__ void init_half_matrices(half *A, half *B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = __float2half(1.0f);
        B[idx] = __float2half(1.0f);
    }
}

// The Tensor Core Kernel
__global__ void gemm_wmma(half *A, half *B, float *C, int n) {
    // 1 Block = 1 Warp (32 threads). 
    // This warp will compute a 16x16 tile of the output matrix C.
    int warpM = blockIdx.y * WMMA_M;
    int warpN = blockIdx.x * WMMA_N;

    // Declare the fragments (opaque Tensor Core registers)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the accumulator to 0
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over the K-dimension
    for (int i = 0; i < n; i += WMMA_K) {
        // The entire warp collaboratively loads 16x16 tiles into the fragments
        wmma::load_matrix_sync(a_frag, A + warpM * n + i, n);
        wmma::load_matrix_sync(b_frag, B + i * n + warpN, n);

        // The Magic: A single hardware instruction to multiply and accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // The warp collaboratively stores the result back to global memory
    wmma::store_matrix_sync(C + warpM * n + warpN, c_frag, n, wmma::mem_row_major);
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

    // Initialize A and B with 1.0f in FP16 directly on the device
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    init_half_matrices<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N * N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Grid config: 1 warp (32 threads) computes a 16x16 tile
    dim3 block(32); 
    dim3 grid(N / WMMA_N, N / WMMA_M);

    // Warm-up
    gemm_wmma<<<grid, block>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int iterations = 100;
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        gemm_wmma<<<grid, block>>>(d_A, d_B, d_C, N);
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
