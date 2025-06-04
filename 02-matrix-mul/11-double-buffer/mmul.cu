#include <stdio.h>

#include <cassert>

#include "../00-cuBLAS/mmul.cuh"

#define data_type float

// Matrix dimensions, two matrixs:
// (M, K) and (K, N)
constexpr int M = 1 << 12;
constexpr int N = 1 << 12;
constexpr int K = 1 << 12;

const int buffer_stages = 2;
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;
const int NUM_THREADS = buffer_stages * (BM / TM) * (BN / TN);
const int NUM_THREADS_PER_BUFFER = NUM_THREADS / buffer_stages;

data_type h_a[M * K];
data_type h_b[K * N];
data_type h_c[M * N];

const int a_bytes = M * K * sizeof(data_type);
const int b_bytes = K * N * sizeof(data_type);
const int c_bytes = M * N * sizeof(data_type);

const int test_round = 100;

__device__ __forceinline__ void loadSMEMA(data_type* smem, data_type* gmem, const int stride, const int iRow, 
                          const int iCol, const int eRow, const int i) {
  for (int sar = 0; sar < BM; sar += stride) {
    smem[((iRow + sar) * BK + iCol)] = gmem[(eRow + iRow + sar) * K + i + iCol];
  }
}

__device__ __forceinline__ void loadSMEMB(data_type* smem, data_type* gmem, const int stride, const int iRow, 
                          const int iCol, const int eCol, const int i) {
  for (int sbr = 0; sbr < BK; sbr += stride) {
      smem[(iRow + sbr) * BN + iCol] = gmem[(i + iRow + sbr) * N + eCol + iCol];
    }
}

__global__ void __launch_bounds__(buffer_stages * (BM * BN) / (TM * TN), 1)
    matrixMul(data_type* a, data_type* b, data_type* c) {
  int iRow = threadIdx.y * TM;  // For threads in block: range(0, 128, 8)
  int iCol = threadIdx.x * TN;  // For threads in block: range(0, 128, 8)

  int eRow = blockIdx.y * blockDim.y * TM;
  int eCol = blockIdx.x * blockDim.x * TN;

  __shared__ data_type s_a[2 * BM * BK];
  __shared__ data_type s_b[2 * BK * BN];

  data_type tmps[TM * TN] = {0};
  data_type regA[TM] = {0};
  data_type regB[TN] = {0};

  int threadBufferId = threadIdx.x + (threadIdx.y * blockDim.x);
  // TODO: the 1 below should be related buffer_stages, it's 2 in our case.
  bool isLowerDoubleBuffer = threadIdx.z < 1;
  int iRowA = threadBufferId / BK;
  int iColA = threadBufferId % BK;
  int strideA = NUM_THREADS_PER_BUFFER / BK;
  int iRowB = threadBufferId / BN;
  int iColB = threadBufferId % BN;
  int strideB = NUM_THREADS_PER_BUFFER / BN;

  loadSMEMA(s_a, a, strideA, iRowA, iColA, eRow, 0);
  loadSMEMB(s_b, b, strideB, iRowB, iColB, eCol, 0);
  __syncthreads();
  // 1st loop is for iterating over the two whole matrixs.
  for (int i = 0; i < K; i += 2 * BK) {
    if(isLowerDoubleBuffer) {
      // process i firstly
      for (int j = 0; j < BK; j++) {
        for (int ra = 0; ra < TM; ra++) {
          regA[ra] = s_a[(iRow + ra) * BK + j];
        }
        for (int rb = 0; rb < TN; rb++) {
          regB[rb] = s_b[j * BN + iCol + rb];
        }

        for (int k = 0; k < TN; k++) {
          for (int s = 0; s < TM; s++) {
            tmps[s * TN + k] += regA[s] * regB[k];
          }
        }
      }
      __syncthreads();

      // load i + 2
      if(i + 2 * BK < K) {
        loadSMEMA(s_a, a, strideA, iRowA, iColA, eRow, i + 2 * BK);
        loadSMEMB(s_b, b, strideB, iRowB, iColB, eCol, i + 2 * BK);
      }
      __syncthreads();
    } else {
      // load i + 1 to upper-half of s_a and s_b
      loadSMEMA(s_a + BM * BK, a, strideA, iRowA, iColA, eRow, i + BK);
      loadSMEMB(s_b + BK * BN, b, strideB, iRowB, iColB, eCol, i + BK);
      __syncthreads();

      // process i + 1
      for (int j = 0; j < BK; j++) {
        for (int ra = 0; ra < TM; ra++) {
          regA[ra] = s_a[BM * BK + (iRow + ra) * BK + j];
        }
        for (int rb = 0; rb < TN; rb++) {
          regB[rb] = s_b[BK * BN + j * BN + iCol + rb];
        }

        for (int k = 0; k < TN; k++) {
          for (int s = 0; s < TM; s++) {
            tmps[s * TN + k] += regA[s] * regB[k];
          }
        }
      }
      __syncthreads();
    }
  }

  if(isLowerDoubleBuffer) {
    // we store tmp for lower half firstly.
    for (int i = 0; i < TM * TN; i++) {
      int iiRow = i / TN;
      int iiCol = i % TN;
      // store to s_a firstly.
      c[(eRow + iRow + iiRow) * N + eCol + iCol + iiCol] = tmps[i];
    }
    __syncthreads();
  } else {
    // wait lower half store finished.
    __syncthreads();
    // then we **accumulate** the tmp to c again.
    // TODO: try not to load c twice, like use shared memory to do it.
    for (int i = 0; i < TM * TN; i++) {
      int iiRow = i / TN;
      int iiCol = i % TN;
      c[(eRow + iRow + iiRow) * N + eCol + iCol + iiCol] += tmps[i];
    }
  }
  __syncthreads();
}

int main() {
  // Initialize h_a and h_b firstly.
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_a[row * K + col] = rand() % 100;
      h_b[row * N + col] = rand() % 100;
      // h_a[row * K + col] = 1;
      // h_b[row * N + col] = 1;
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

  const dim3 threads(BN / TN, BM / TM, buffer_stages);
  const dim3 blocks(BLOCK_X, BLOCK_Y);

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

  cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double FLOPs = 2.0 * M * N * K * test_round;
  float GFLOPS = FLOPs / (milliseconds * 1e6);

  printf("Kernel execution time: %.02f ms\n", milliseconds);
  printf("GFLOPS: %.02f gops\n", GFLOPS);

  verify_with_cublas(M, N, K, d_a, d_b, d_c);

  printf("COMPLETED SUCCESSFULLY\n");

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}