#include <stdio.h>

#include <cassert>

// There is just about ~40% improvement compared with basic shared
// memory on RTX 3080Ti.

// NOTE: whether we use tmp or not in the 2rd loop didn't improve
// perf at all, because even we didn't use it, the compiler will
// optimize it if we have a smaller 3rd loop.

#define data_type float

// Matrix dimensions, two matrixs:
// (M, K) and (K, N)
constexpr int M = 1 << 10;
constexpr int N = 1 << 10;
constexpr int K = 1 << 10;

const int BM = 64;
const int BN = 64;
const int BK = 8;
const int TM = 8;

data_type h_a[M * K];
data_type h_b[K * N];
data_type h_c[M * N];

const int a_bytes = M * K * sizeof(data_type);
const int b_bytes = K * N * sizeof(data_type);
const int c_bytes = M * N * sizeof(data_type);

__global__ void matrixMul(data_type* a, data_type* b, data_type* c) {
  int iRow = threadIdx.y * TM;  // For threads in block: range(0, 64, 8)
  int iCol = threadIdx.x;       // For threads in block: range(0, 64, 1)

  int eRow = blockIdx.y * blockDim.y * TM;
  int eCol = blockIdx.x * blockDim.x;

  // start row and col for the 1st loop.
  int row = iRow + eRow;
  int col = iCol + eCol;

  __shared__ data_type s_a[BM * BK];
  __shared__ data_type s_b[BK * BN];

  int iRowA = threadIdx.x;
  int iColA = threadIdx.y;
  int iRowB = threadIdx.y;
  int iColB = threadIdx.x;

  data_type tmps[TM] = {0};
  // 1st loop is for iterating over the two whole matrixs.
  for (int i = 0; i < K; i += BK) {
    if (iColA < BK) {
      s_a[iRowA * BK + iColA] = a[(eRow + iRowA) * K + i + iColA];
    }
    if (iRowB < BK) {
      s_b[iRowB * BN + iColB] = b[(iRowB + i) * N + eCol + iColB];
    }

    __syncthreads();

    // 2nd loop interate over a block.
    for (int j = 0; j < BK; j++) {
      // NOTE: The below code show same perf with below code in comment.
      // Hence whether we use the tmp variable or not, it didn't improve
      // perf at all, because in the below case, the compiler will
      // optimize the kernel PTX code same as above.
      data_type tmp = s_b[j * BN + iCol];
      // 3rd loop for iterate row over matrix A.
      for (int k = 0; k < TM; k++) {
        tmps[k] += tmp * s_a[(TM * threadIdx.y + k) * BK + j];
      }

      /*
      for(int k = 0; k < TM; k ++) {
          tmps[k] += s_b[j * BN + iCol] * s_a[(TM * threadIdx.y + k) * BK + j];
      }
      */
    }
    __syncthreads();
  }

  for (int r = 0; r < TM; r++) {
    if (r + row < M && col < N) {
      c[(r + row) * N + col] = tmps[r];
      // printf("c[%d][%d]: %f\n", r, col, tmps[r]);
    }
  }
}

void verify_results(data_type* a, data_type* b, data_type* c, int N) {
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      data_type a_times_b = 0;
      for (int i = 0; i < K; i++) {
        a_times_b += a[row * K + i] * b[N * i + col];
      }
      if (a_times_b != c[row * N + col]) {
        printf("the result is wrong at row: %d, column: %d\n", row, col);
        printf("it should be %f, but it's %f\n", a_times_b, c[row * N + col]);
        abort();
      }
    }
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

  data_type* d_a;
  data_type* d_b;
  data_type* d_c;
  cudaMalloc(&d_a, a_bytes);
  cudaMalloc(&d_b, b_bytes);
  cudaMalloc(&d_c, c_bytes);

  cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice);

  // TODO: rethink the BLOCK_X and BLOCK_Y order. I thought it's
  // not important, we could switch them.
  const int BLOCK_X = N / BN;
  const int BLOCK_Y = M / BM;

  const dim3 threads(BN, BM / TM);
  const dim3 blocks(BLOCK_X, BLOCK_Y);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record start event
  cudaEventRecord(start);

  for (int i = 0; i < 100; i++) {
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);
  }

  // Record stop event
  cudaEventRecord(stop);

  cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double FLOPs = 2.0 * N * M * K;
  float GFLOPS = FLOPs / (milliseconds * 1e6);

  printf("Kernel execution time: %f ms\n", milliseconds);
  printf("GFLOPS: %f gops\n", GFLOPS);

  verify_results(h_a, h_b, h_c, N);

  printf("COMPLETED SUCCESSFULLY\n");

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}