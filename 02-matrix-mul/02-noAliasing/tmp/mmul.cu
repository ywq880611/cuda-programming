#include <stdio.h>

const int N = 1 << 10;
const int bytes = N * N * sizeof(N);
int h_a[N * N];
int h_b[N * N];
int h_c[N * N];

// On RTX 3090, the if we use tmp to avoid alias (or avoid to access cache), it
// shows `25ms vs 10ms`. If we also add `__restrict__` here, we couldn't see
// more benefits, so I guess `tmp` do same thing as `__restrict__`, maybe I
// could check PTX later?
__global__ void matrixMul(int* a, int* b, int* c, int N) {
  // take N as both no. of rows and coulums, so here is
  // a sqare matrix.

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int tmp = 0;
  for (int i = 0; i < N; i++) {
    tmp += a[row * N + i] * b[N * i + col];
  }
  c[row * N + col] = tmp;
}

void verify_results(int* a, int* b, int* c, int N) {
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      int a_times_b = 0;
      for (int i = 0; i < N; i++) {
        a_times_b += a[row * N + i] * b[N * i + col];
      }
      if (a_times_b != c[row * N + col]) {
        printf("the result is wrong at row: %d, column: %d\n", row, col);
        printf("it should be %d, but it's %d\n", a_times_b, c[row * N + col]);
        abort();
      }
    }
  }
}

int main() {
  // Initialize h_a and h_b firstly.
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      h_a[row * N + col] = 1;
      h_b[row * N + col] = 1;
      // h_a[row * N + col] = rand() % 100;
      // h_b[row * N + col] = rand() % 100;
    }
  }

  int* d_a;
  int* d_b;
  int* d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  const int THREAD = 32;
  const int BLOCK = (N + THREAD - 1) / THREAD;

  const dim3 threads(THREAD, THREAD);
  const dim3 blocks(BLOCK, BLOCK);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record start event
  cudaEventRecord(start);

  for (int i = 0; i < 10; i++) {
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
  }

  // Record stop event
  cudaEventRecord(stop);

  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  printf("Kernel execution time: %f ms\n", milliseconds);

  verify_results(h_a, h_b, h_c, N);

  printf("COMPLETED SUCCESSFULLY\n");

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}