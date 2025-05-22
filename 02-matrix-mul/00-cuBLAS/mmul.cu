#include "mmul.cuh"
#include <cassert>

// There is just about ~8% improvement compared with previous
// version kernel on RTX 3080Ti.

// NOTE: the 8% gain from the writeback to C matrix in kernel,
// it was compiled with SIMD inst, but why we didn't use SIMD
// inst in the process of loading GMEM into SMEM? it's a question.

#define data_type float

// Matrix dimensions, two matrixs:
// (M, K) and (K, N)
constexpr int M = 1 << 12;
constexpr int N = 1 << 12;
constexpr int K = 1 << 12;

data_type h_a[M * K];
data_type h_b[K * N];
data_type h_c[M * N];

const int a_bytes = M * K * sizeof(data_type);
const int b_bytes = K * N * sizeof(data_type);
const int c_bytes = M * N * sizeof(data_type);

static const int test_round = 100;

void matrixMul(data_type* a, data_type* b, data_type* c) {
  cublass_gemm(M, N, K, a, b, c);
}

void verify_results(data_type* a, data_type* b, data_type* c) {
  float epsilon = 1e-3f;
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      data_type a_times_b = 0;
      for (int i = 0; i < K; i++) {
        a_times_b += a[row * K + i] * b[N * i + col];
      }
      if (abs(a_times_b - c[row * N + col]) > epsilon) {
        printf("the result is wrong at row: %d, column: %d\n", row, col);
        printf("it should be %f, but it's %f\n", a_times_b, c[row * N + col]);
        abort();
      }
    }
  }
}

int main() {
  // Initialize h_a and h_b firstly.
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_a[row * N + col] = rand() % 100;
      h_b[row * N + col] = rand() % 100;
    }
  }

  data_type* d_a;
  data_type* d_b;
  data_type* d_c;
  cudaMalloc(&d_a, a_bytes);
  cudaMalloc(&d_b, b_bytes);
  cudaMalloc(&d_c, c_bytes);

  cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice);

  // warm up
  for (int i = 0; i < test_round; i++) {
    matrixMul(d_a, d_b, d_c);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record start event
  cudaEventRecord(start);

  for (int i = 0; i < test_round; i++) {
    matrixMul(d_a, d_b, d_c);
  }

  // Record stop event
  cudaEventRecord(stop);

  cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double FLOPs = 2.0 * N * M * K * test_round;
  float GFLOPS = FLOPs / (milliseconds * 1e6);

  printf("Kernel execution time: %.02f ms\n", milliseconds);
  printf("GFLOPS: %.02f gops\n", GFLOPS);

  verify_results(h_a, h_b, h_c);

  printf("COMPLETED SUCCESSFULLY\n");

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}