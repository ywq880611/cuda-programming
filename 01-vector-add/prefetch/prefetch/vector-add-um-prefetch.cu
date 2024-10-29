#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// The time cost for prefetch is about 920 ms, the prefetch is about 1020 ms.
// ~10% improvemenet for prefetch and memadvise.

using namespace std;
const int N = 1 << 26;

__global__ void vector_add(int* a, int* b, int* c, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    // CUDA's printf has certain limitations, including how many outputs can be
    // printed depending on the execution context and available resources.
    // If a large number of threads are executing, not all print statements may
    // be visible. This issue can become more prominent in larger grids or
    // blocks.
    // printf("kernel id: %dd\n", tid);
    c[tid] = a[tid] + b[tid];
  }
}

void check_result(int* a, int* b, int* c) {
  for (int i = 0; i < N; i++) {
    if (c[i] != a[i] + b[i]) {
      printf("i is: %d\n", i);
      printf("a[i]: %d, b[i]: %d, c[i]: %d is not match!\n", a[i], b[i], c[i]);
      abort();
    }
  }
}

int main() {
  const int bytes = N * sizeof(int);

  int id = cudaGetDevice(&id);

  int* a;
  int* b;
  int* c;
  // Allocation memory for these pointers
  cudaMallocManaged(&a, bytes);
  cudaMallocManaged(&b, bytes);
  cudaMallocManaged(&c, bytes);

  cudaMemAdvise(a, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemAdvise(b, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemPrefetchAsync(c, bytes, id);

  for (int i = 0; i < N; i++) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
  }

  cudaMemAdvise(a, bytes, cudaMemAdviseSetReadMostly, id);
  cudaMemAdvise(b, bytes, cudaMemAdviseSetReadMostly, id);
  cudaMemPrefetchAsync(a, bytes, id);
  cudaMemPrefetchAsync(b, bytes, id);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int n_thread = 1 << 10;  // 1024 threads
  int n_block = (N + n_thread - 1) / n_thread;

  // Record start event
  cudaEventRecord(start);
  for (int i = 0; i < 1000; i++) {
    vector_add<<<n_block, n_thread>>>(a, b, c, N);
  }

  // Record stop event
  cudaEventRecord(stop);

  cudaDeviceSynchronize();

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Kernel execution time: " << milliseconds << " ms\n";

  cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

  check_result(a, b, c);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
