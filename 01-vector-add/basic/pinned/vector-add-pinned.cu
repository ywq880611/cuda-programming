#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using namespace std;
const int N = 1 << 26;

// Compare with base, the pinned shows same kernel execution time.
// but if we use `time pinned.o` and `time base.o` to measure the
// total time cost, it shows the base version take ~6.8 sec, but
// the pinned version take ~4.1 sec.
//
// So the CudaMallocHost could obvious improve the memory I/O speed
// between host and device!
// According to the chat-gpt, it seems CudaMallocaHost could pin the
// memory on the physical memory of host to avoid switched it to disk.

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
  // Larger N to show cudaMallocHost make program more efficience.
  const int bytes = N * sizeof(int);

  int* a_h;
  int* b_h;
  int* c_h;

  cudaMallocHost(&a_h, bytes);
  cudaMallocHost(&b_h, bytes);
  cudaMallocHost(&c_h, bytes);

  for (int i = 0; i < N; i++) {
    a_h[i] = rand() % 100;
    b_h[i] = rand() % 100;
  }

  int* a_d;
  int* b_d;
  int* c_d;
  cudaMalloc(&a_d, bytes);
  cudaMalloc(&b_d, bytes);
  cudaMalloc(&c_d, bytes);

  cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int n_thread = 1 << 10;  // 1024 threads
  int n_block = (N + n_thread - 1) / n_thread;

  // Record start event
  cudaEventRecord(start);
  for (int i = 0; i < 1000; i++) {
    vector_add<<<n_block, n_thread>>>(a_d, b_d, c_d, N);
  }

  // Record stop event
  cudaEventRecord(stop);

  // cudaDeviceSynchronize();

  // This method will sync the previous launched kernel.
  cudaMemcpy(c_h, c_d, bytes, cudaMemcpyDeviceToHost);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Kernel execution time: " << milliseconds << " ms\n";

  check_result(a_h, b_h, c_h);

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
