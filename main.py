import numpy as np
import time

# np.__config__.show()

N = 1024

A = np.ones((N, N), dtype=np.float32)
B = np.ones((N, N), dtype=np.float32)

# Warm up
C = A @ B

start = time.perf_counter()
C = A @ B
end = time.perf_counter()

time_taken = end - start

flops = 2 * N * N * N
gflops = flops / (time_taken * 1e9)

print("Time:", time_taken)
print("GFLOPS:", gflops)
print("C[0,0]:", C[0,0])
