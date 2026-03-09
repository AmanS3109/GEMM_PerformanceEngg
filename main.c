#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define N 1024
#define BS 128

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

float A[N][N] __attribute__((aligned(64)));
float B[N][N] __attribute__((aligned(64)));
float C[N][N] __attribute__((aligned(64)));

float Bpack[BS][BS] __attribute__((aligned(64)));

void GEMM(float (*restrict A)[N],
          float (*restrict B)[N],
          float (*restrict C)[N])
{
    for (int ii = 0; ii < N; ii += BS) {
        int i_max = (ii + BS < N) ? ii + BS : N;

        for (int kk = 0; kk < N; kk += BS) {
            int k_max = (kk + BS < N) ? kk + BS : N;

            for (int jj = 0; jj < N; jj += BS) {
                int j_max = (jj + BS < N) ? jj + BS : N;

                /* ---- Pack B block ---- */
                for (int k = kk; k < k_max; k++)
                    for (int j = jj; j < j_max; j++)
                        Bpack[k-kk][j-jj] = B[k][j];

                /* ---- Compute block ---- */
                for (int i = ii; i < i_max; i += 4) {

                    float *Crow0 = &C[i][0];
                    float *Crow1 = &C[i+1][0];
                    float *Crow2 = &C[i+2][0];
                    float *Crow3 = &C[i+3][0];

                    for (int j = jj; j < j_max; j += 8) {

                        __m256 c0 = _mm256_load_ps(&Crow0[j]);
                        __m256 c1 = _mm256_load_ps(&Crow1[j]);
                        __m256 c2 = _mm256_load_ps(&Crow2[j]);
                        __m256 c3 = _mm256_load_ps(&Crow3[j]);

                        for (int k = kk; k < k_max; k++) {

                            __m256 b = _mm256_load_ps(&Bpack[k-kk][j-jj]);

                            __m256 a0 = _mm256_set1_ps(A[i][k]);
                            __m256 a1 = _mm256_set1_ps(A[i+1][k]);
                            __m256 a2 = _mm256_set1_ps(A[i+2][k]);
                            __m256 a3 = _mm256_set1_ps(A[i+3][k]);

                            c0 = _mm256_fmadd_ps(a0, b, c0);
                            c1 = _mm256_fmadd_ps(a1, b, c1);
                            c2 = _mm256_fmadd_ps(a2, b, c2);
                            c3 = _mm256_fmadd_ps(a3, b, c3);
                        }

                        _mm256_store_ps(&Crow0[j], c0);
                        _mm256_store_ps(&Crow1[j], c1);
                        _mm256_store_ps(&Crow2[j], c2);
                        _mm256_store_ps(&Crow3[j], c3);
                    }
                }
            }
        }
    }
}

int main()
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0f;
            B[i][j] = 1.0f;
            C[i][j] = 0.0f;
        }

    double start = get_time();

    GEMM(A, B, C);

    double end = get_time();

    double time = end - start;
    double flops = 2.0 * N * N * N;
    double gflops = flops / (time * 1e9);

    printf("Time: %f seconds\n", time);
    printf("GFLOPS: %f\n", gflops);
    printf("C[0][0] = %f\n", C[0][0]);

    return 0;
}
