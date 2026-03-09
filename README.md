# Zero to 6.7 TeraFLOPS: Matrix Multiplication Optimization Journey

This repository documents a step-by-step performance engineering journey of optimizing a Matrix Multiplication (GEMM) algorithm. It starts from baseline CPU and NumPy implementations and scales all the way up to a state-of-the-art, hardware-asynchronous CUDA Tensor Core kernel.



## 🚀 The Journey & File Structure

Each file in this repository represents a specific architectural optimization, demonstrating how to bypass different hardware bottlenecks on modern processors and NVIDIA GPUs.

### Baselines
* **`main.c`**: Standard C implementation running on the CPU.
* **`main.py`**: Python implementation utilizing highly-optimized NumPy (OpenBLAS/MKL) under the hood.
* **`cublas_gemm.cu`**: The GPU baseline benchmark using NVIDIA's standard `cublasSgemm` (Highly optimized library).

### Custom CUDA Kernels
* **`gemm.cu`**: The naive global memory approach. High latency, massive memory traffic.
* **`gemm_warp.cu`**: Implementing **Shared Memory Tiling** to minimize global memory reads.
* **`gemm_reg.cu`**: Implementing **Register Tiling (Thread Coarsening)** to push standard CUDA cores to their physical limits and break the shared memory bottleneck.
* **`wmma_gemm.cu`**: The paradigm shift. Utilizing the `nvcuda::wmma` API to unlock **Tensor Cores** (FP16 inputs, FP32 accumulate).
* **`wmma_shared.cu`**: Feeding Tensor Cores using **Shared Memory** tiles to prevent memory starvation.
* **`wmma_pipeline.cu`**: Implementing **Software Pipelining (Double Buffering)**. *(Note: Performance drops here due to increased shared memory usage tanking Streaming Multiprocessor occupancy).*
* **`wmma_async.cu`**: The final boss. Using Ampere architecture **Hardware Asynchronous Copies (`cp.async`)** to overlap memory loads and math without sacrificing occupancy.

---

## 📊 Performance Metrics

*Hardware Target: NVIDIA RTX 3000/4000 series or newer (Ampere+ Architecture)* *Matrix Size: 1024 x 1024*

| Platform | Stage / Implementation | File | Strategy | GFLOPS |
| :--- | :--- | :--- | :--- | :--- |
| **CPU** | 1. Naive C | `main.c` | Standard Triple-For-Loop | ~ 75 |
| **CPU** | 2. NumPy (Python) | `main.py` | AVX/SIMD Vectorized BLAS | ~ 320 - 350 |
| **GPU** | 3. Shared Memory Tiled | `gemm_warp.cu` | L1 Cache / Shared Mem | ~ 637 |
| **GPU** | 4. Register Tiled | `gemm_reg.cu` | Thread Coarsening | ~ 1,864 |
| **GPU** | 5. Tensor Cores (Naive) | `wmma_gemm.cu` | Global Mem -> Tensor Cores | ~ 4,025 |
| **GPU** | 6. Tensor Cores + Shared | `wmma_shared.cu` | Global -> Shared -> Tensor Cores | ~ 4,250 |
| **GPU** | 7. cuBLAS SGEMM | `cublas_gemm.cu` | `cublasSgemm` (Standard Cores) | ~ 5,280 |
| **GPU** | 8. Software Pipelining | `wmma_pipeline.cu` | Double Buffering *(Occupancy Drop)* | ~ 3,404 |
| **GPU** | **9. Hardware Async Pipeline** | **`wmma_async.cu`** | **`cp.async` Bypassing Registers** | **~ 6,779** |

---

## 🥊 Beating the Baselines: Architectural Takeaways

By tracing the performance table, you can see exactly how hardware awareness dictates speed:

1. **Beating the CPU (75 -> 350 GFLOPS):** NumPy crushes raw C code by utilizing CPU-specific SIMD (Single Instruction, Multiple Data) vector instructions like AVX-512.
2. **Beating standard CUDA Cores (637 -> 1864 GFLOPS):** Moving from shared memory to registers (`gemm_reg.cu`) ensures the math units aren't waiting on the L1 cache.
3. **Beating cuBLAS SGEMM (5280 -> 6779 GFLOPS):** The `cublasSgemm` function is strictly defined to perform Single Precision (FP32) math using standard CUDA cores. To surpass this 5.2 TFLOP limit, we pivoted to mixed-precision math via the `wmma` API. 
    
    
    
    By feeding the GPU `half` precision (FP16) inputs and accumulating the results in `float` (FP32), we unlocked the **Tensor Cores**, performing matrix math in a single clock cycle, fundamentally bypassing the physical floating-point limitations that standard cuBLAS SGEMM is bound to.
4. **The Final Leap (4250 -> 6779 GFLOPS):** Even with Tensor Cores, threads wasting time moving memory stalls performance. 

    

    By utilizing `cp.async` in Ampere hardware, we commanded the GPU's memory controller to fetch data directly into Shared Memory, bypassing the registers entirely and keeping the Tensor Cores fed at 100% utilization. This solved the occupancy drop seen in the software pipelining stage.

---

## 🛠️ How to Build and Run

A simple `Makefile` is included to compile all targets with the correct architecture flags required for Tensor Cores and async memory copies.

```bash
# Compile all kernels
make

# Run the final optimized kernel
./wmma_async

# Clean up build artifacts
make clean
