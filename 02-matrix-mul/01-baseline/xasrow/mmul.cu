#include <stdio.h>

#include "../../00-cuBLAS/mmul.cuh"

const int N = 1 << 12;
const int bytes = N * N * sizeof(float);
float h_a[N * N];
float h_b[N * N];
float h_c[N * N];

const int test_round = 10;

__global__ void matrixMul(float* a, float* b, float* c) {
  // take N as both no. of rows and coulums, so here is
  // a sqare matrix.

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  c[row * N + col] = 0;
  for (int i = 0; i < N; i++) {
    c[row * N + col] += a[row * N + i] * b[N * i + col];
  }
}

int main() {
  // Initialize h_a and h_b firstly.
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      h_a[row * N + col] = rand() % 100;
      h_b[row * N + col] = rand() % 100;
    }
  }

  float* d_a;
  float* d_b;
  float* d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  const int THREAD = 32;
  const int BLOCK = (N + THREAD - 1) / THREAD;

  const dim3 threads(THREAD, THREAD);
  const dim3 blocks(BLOCK, BLOCK);

  // warm up
  for (int i = 0; i < test_round; i++) {
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record start event
  cudaEventRecord(start);

  for (int i = 0; i < test_round; i++) {
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);
  }

  // Record stop event
  cudaEventRecord(stop);

  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double FLOPs = 2.0 * N * N * N * test_round;
  float GFLOPS = FLOPs / (milliseconds * 1e6);

  printf("Kernel execution time: %.02f ms\n", milliseconds);
  printf("GFLOPS: %.02f gops\n", GFLOPS);

  verify_with_cublas(N, N, N, d_a, d_b, d_c);

  printf("COMPLETED SUCCESSFULLY\n");

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}