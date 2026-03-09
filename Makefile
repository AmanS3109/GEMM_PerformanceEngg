# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=native
CUBLAS_LIBS = -lcublas

# List of all target executables to build
TARGETS = gemm cublas_gemm gemm_warp gemm_reg wmma_gemm wmma_shared wmma_pipeline wmma_async

# Default rule when you just type 'make'
all: $(TARGETS)

# Specific rule for the cuBLAS benchmark (needs the -lcublas flag)
cublas_gemm: cublas_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $(CUBLAS_LIBS) $< -o $@

# Generic pattern rule for all other CUDA files
%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Clean up build artifacts
clean:
	rm -f $(TARGETS)
