#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define TILE 16
#define RTILE 4

#define CHECK_CUDA(call)                                      \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            printf("CUDA error %s:%d : %s\n",                  \
                   __FILE__, __LINE__,                         \
                   cudaGetErrorString(err));                   \
            exit(1);                                           \
        }                                                      \
    } while (0)

__global__ void gemm_reg(float *A, float *B, float *C, int n)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE * RTILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum[RTILE] = {0};

    for (int t = 0; t < n / TILE; t++)
    {
        for (int i = 0; i < RTILE; i++)
        {
            As[ty + i * TILE/RTILE][tx] =
                A[(row + i * TILE/RTILE) * n + t * TILE + tx];
        }

        Bs[ty][tx] =
            B[(t * TILE + ty) * n + col];

        __syncthreads();

        for (int k = 0; k < TILE; k++)
        {
            float b = Bs[k][tx];

            for (int i = 0; i < RTILE; i++)
            {
                sum[i] += As[ty + i * TILE/RTILE][k] * b;
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < RTILE; i++)
    {
        C[(row + i * TILE/RTILE) * n + col] = sum[i];
    }
}

int main()
{
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++)
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

    dim3 block(TILE, TILE/RTILE);
    dim3 grid(N/TILE, N/(TILE*RTILE));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    cudaEventRecord(start);

    gemm_reg<<<grid, block>>>(d_A, d_B, d_C, N);

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    double seconds = ms / 1000.0;
    double flops = 2.0 * N * N * N;
    double gflops = flops / (seconds * 1e9);

    printf("Time: %f seconds\n", seconds);
    printf("GFLOPS: %f\n", gflops);
    printf("C[0] = %f\n", h_C[0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
