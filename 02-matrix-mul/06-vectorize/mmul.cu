#include <stdio.h>

#include <cassert>

// There is just about ~8% improvement compared with previous
// version kernel on RTX 3080Ti.

// NOTE: the 8% gain from the writeback to C matrix in kernel,
// it was compiled with SIMD inst, but why we didn't use SIMD
// inst in the process of loading GMEM into SMEM? it's a question.

#define data_type float

// Matrix dimensions, two matrixs:
// (M, K) and (K, N)
constexpr int M = 1 << 10;
constexpr int N = 1 << 10;
constexpr int K = 1 << 10;

const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

data_type h_a[M * K];
data_type h_b[K * N];
data_type h_c[M * N];

const int a_bytes = M * K * sizeof(data_type);
const int b_bytes = K * N * sizeof(data_type);
const int c_bytes = M * N * sizeof(data_type);

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    matrixMul(data_type* a, data_type* b, data_type* c) {
  int iRow = threadIdx.y * TM;  // For threads in block: range(0, 128, 8)
  int iCol = threadIdx.x * TN;  // For threads in block: range(0, 128, 8)

  int eRow = blockIdx.y * blockDim.y * TM;
  int eCol = blockIdx.x * blockDim.x * TN;

  __shared__ data_type s_a[BM * BK];
  __shared__ data_type s_b[BK * BN];

  data_type tmps[TM * TN] = {0};
  data_type regA[TM] = {0};
  data_type regB[TN] = {0};
  // 1st loop is for iterating over the two whole matrixs.
  for (int i = 0; i < K; i += BK) {
    // Load GMEM into SMEM
    // NOTE: This vectorized optimization based on an assumption:
    // There is (BM / TM) * (BN / TN) = 256 threads per block
    // For SMEM s_a and s_b, each of them have same number of
    // elements BM(or BN) * BK = 1024.
    // So we could see each thread should load 1024/256 = 4
    // elements, in the previous 05 kernel, we already do this,
    // but we did it by a loop, which should load float32
    // 4 times and the 4 float32 element were not adjacent.

    // NOTE: Hence there is an optimizaion idea, we could load
    // 4 adjacent elements once in a thread by a load.128b inst
    // which will improve perf.

    // TODP: it seems the below code didn't be compiled with
    // load.128b inst, so it didn't improve perf why?
    // instigate it later!!!
    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);  // range(0, 256)
    int iRowA = threadId / (BK / 4);
    int iColA = (threadId % (BK / 4)) * 4;
    int iRowB = threadId / (BN / 4);
    int iColB = (threadId % (BN / 4)) * 4;

    reinterpret_cast<float4*>(&s_a[iRowA * BK + iColA])[0] =
        reinterpret_cast<float4*>(&a[(iRowA + eRow) * K + i + iColA])[0];

    reinterpret_cast<float4*>(&s_b[iRowB * BN + iColB])[0] =
        reinterpret_cast<float4*>(&b[(iRowB + i) * N + eCol + iColB])[0];

    __syncthreads();

    // 2nd loop interate over a block.
    for (int j = 0; j < BK; j++) {
      // Load the element from SMEM into register, but I understand
      // for 3rd loop the register array regA is redudant, we could
      // just use a register to do it
      for (int ra = 0; ra < TM; ra++) {
        regA[ra] = s_a[(iRow + ra) * BK + j];
      }
      for (int rb = 0; rb < TN; rb++) {
        // Maybe removed, because we could just use one register
        // in 3rd loop.
        regB[rb] = s_b[j * BN + iCol + rb];
      }

      // 3rd loop for iterate column over matrix B.
      for (int k = 0; k < TN; k++) {
        // 4th loop for iterator row over matrix A.
        for (int s = 0; s < TM; s++) {
          tmps[s * TN + k] += regA[s] * regB[k];
        }
      }
    }
    __syncthreads();
  }

  /*
  for(int i = 0; i < TM * TN; i ++) {
      int iiRow = i / TN;
      int iiCol = i % TN;
      c[(eRow + iRow + iiRow) * N + eCol + iCol + iiCol] = tmps[i];
  }
  */

  // Writeback could also be vectorized.
  // NOTE: this writeback is better than above writeback, it could
  // utilize `st.global.v4.f32` inst, and it shows ~8% improve aginst
  // previous kernel.
  // Check whether the process of loading GMEM into SMEM didn't use
  // `ld.global.v4.f32` inst!!!! if it was used, we may get a better
  // performance!!
  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j += 4) {
      reinterpret_cast<float4*>(
          &c[(eRow + iRow + i) * N + +eCol + iCol + j])[0] =
          reinterpret_cast<float4*>(&tmps[i * TN + j])[0];
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

  // TODO: rethink the BLOCK_X and BLOCK_Y order. I thought it's
  // not important, we could switch them.
  const int BLOCK_X = N / BN;
  const int BLOCK_Y = M / BM;

  const dim3 threads(BN / TN, BM / TM);
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