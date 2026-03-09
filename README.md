# Zero to 6.7 TeraFLOPS: CUDA Matrix Multiplication Optimization

This repository documents a step-by-step performance engineering journey of optimizing a Matrix Multiplication (GEMM) kernel in CUDA. It starts from a naive implementation and scales up to a state-of-the-art, hardware-asynchronous Tensor Core kernel.

## 🚀 The Journey & File Structure

Each file in this repository represents a specific architectural optimization, demonstrating how to bypass different hardware bottlenecks on modern NVIDIA GPUs.

* **`gemm.cu`**: The naive global memory approach. High latency, massive memory traffic.
* **`cublas_gemm.cu`**: The baseline benchmark using NVIDIA's standard `cublasSgemm` (FP32 on standard CUDA cores).
* **`gemm_warp.cu` & `gemm_reg.cu`**: Implementing **Shared Memory Tiling** and **Register Tiling** (Thread Coarsening) to minimize global memory reads and push standard CUDA cores to their physical limits.
* **`wmma_gemm.cu`**: The paradigm shift. Utilizing the `nvcuda::wmma` API to unlock **Tensor Cores** (FP16 inputs, FP32 accumulate).
* **`wmma_shared.cu`**: Feeding Tensor Cores using **Shared Memory** tiles to prevent memory starvation.
* **`wmma_pipeline.cu`**: Implementing **Software Pipelining (Double Buffering)**. (Note: Performance dropped here due to increased shared memory usage tanking SM occupancy).
* **`wmma_async.cu`**: The final boss. Using Ampere architecture **Hardware Asynchronous Copies (`cp.async`)** to overlap memory loads and math without sacrificing occupancy.

## 📊 Performance Metrics

*Hardware Target: NVIDIA RTX 3000/4000 series or newer (Ampere+ Architecture)*
*Matrix Size: 1024 x 1024*

| Optimization Stage | File | Strategy | GFLOPS |
| :--- | :--- | :--- | :--- |
| 1. Baseline CUDA Cores | `gemm_warp.cu` | Shared Memory Tiling | ~530 |
| 2. Tensor Cores (Naive) | `wmma_gemm.cu` | Global Mem -> Tensor Cores | *Skipped/Intermediary* |
| 3. Tensor Cores + Shared | `wmma_shared.cu` | Global -> Shared -> Tensor Cores | ~4,255 |
| 4. Software Pipelining | `wmma_pipeline.cu` | Double Buffering (Thread-driven) | ~3,404 *(Occupancy Drop)* |
| **5. Hardware Async Pipeline** | **`wmma_async.cu`** | **`cp.async` Bypassing Registers** | **~6,779** |

---

## 🥊 Did we beat cuBLAS?

During testing, the fully optimized asynchronous Tensor Core kernel (`wmma_async.cu`) achieved **~6.7 TeraFLOPS**, entirely eclipsing the standard `cublasSgemm` benchmark. 

**How is this possible?**
We didn't just write faster C++ code; we played an architectural trump card. The `cublasSgemm` function is strictly defined to perform Single Precision (FP32) math using standard CUDA cores. 

To achieve 6.7 TFLOPS, we pivoted our kernel to use mixed-precision math via the `wmma` API. By feeding the GPU `half` precision (FP16) inputs and accumulating the results in `float` (FP32), we were able to invoke the **Tensor Cores**. Tensor Cores perform matrix multiplication at a hardware level in a single clock cycle, fundamentally bypassing the physical floating-point limitations that standard cuBLAS SGEMM is bound to. 

To truly compare this code against NVIDIA's best, it would need to be benchmarked against `cublasGemmEx` or `cuBLASLt` (which also utilize Tensor Cores). However, building a custom kernel that touches the physical ceiling of the hardware—and successfully orchestrates asynchronous memory controllers—stands as a massive success in GPU performance engineering.
