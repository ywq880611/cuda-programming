#include <stdio.h>

#include <cassert>

#include "../00-cuBLAS/mmul.cuh"

// There is just about ~120% improvement compared with 1d tile
// version on RTX 3080Ti.

// NOTE: The kernel perf exceed the expection compared with
// the previous kernel. Because the yestday kernel waste some
// thread in process of loading GMEM into SMEM, now we optimize
// it, and improve the perf gain from ~50% to ~120%.

// NOTE: If we increase the matrix size to 4096, we could see
// ~50% gain, on our local test the kernel 4 and kernel 5 in
// https://siboehm.com/articles/22/CUDA-MMM shows ~90% gain.
// We should investigate later.

// the improvement is not meet our expectation, in the
// blog, the author claim this kernel will bring 2x perf gain
// compared with 1d tile kernel.

#define data_type float

// Matrix dimensions, two matrixs:
// (M, K) and (K, N)
constexpr int M = 1 << 12;
constexpr int N = 1 << 12;
constexpr int K = 1 << 12;

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

const int test_round = 100;

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
    /*
    // Load GMEM into SMEM
    // The two asserts is nesscessry for the below load, otherwise
    // we couldn't load the data into SMEM fully.
    assert(blockDim.y >= BK);
    assert(blockDim.x >= BK);
    for(int sar = 0; sar < TM; sar ++) {
        if(threadIdx.x < BK) {
            s_a[(iRow + sar) * BK + threadIdx.x] = a[(iRow + eRow + sar) * K + i
    + threadIdx.x];
        }
    }
    for(int sbc = 0; sbc < TN; sbc ++) {
        if(threadIdx.y < BK) {
            s_b[threadIdx.y * BN + iCol + sbc] = b[(i + threadIdx.y) * N + eCol
    + iCol + sbc];
        }
    }
    */

    // NOTE: the reason for yestday kernel's perf didn't meet expection:
    // The above solution for load GMEM into SMEM waste some resources,
    // because the if(threadIdx.x < BK) and if(threadIdx.y < BK) check
    // forced some thread to do nothing, but wait for other thread load
    // from GMEM, so the below solution I would like to ask each thread
    // to load an element from GMEM.
    // In this approch, the thread may not enought to load all elements
    // we want from GMEM into SMEM, so we have to do some loop to achive
    // our purpose.
    int threadCnt = (BN / TN) * (BM / TM);
    int threadId = threadIdx.x + (threadIdx.y * blockDim.x);
    int iRowA = threadId / BK;
    int iColA = threadId % BK;
    int strideA = threadCnt / BK;
    int iRowB = threadId / BN;
    int iColB = threadId % BN;
    int strideB = threadCnt / BN;

    for (int sar = 0; sar < BM; sar += strideA) {
      s_a[((iRowA + sar) * BK + iColA)] =
          a[(eRow + iRowA + sar) * K + i + iColA];
    }
    for (int sbr = 0; sbr < BK; sbr += strideB) {
      s_b[(iRowB + sbr) * BN + iColB] = b[(i + iRowB + sbr) * N + eCol + iColB];
    }

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
      /*
      // 3rd loop for iterate column over matrix A.
      for(int k = 0; k < TM; k ++) {
          // 4th loop for iterator row over matrix B.
          for(int s= 0; s < TN; s ++) {
              tmps[k * TN + s] += regA[k] * regB[s];
          }
      }
      */

      // TODO: the below version perf is better than the
      // above version, should investigate later.
      // Maybe it's related to memory coalescing?
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

  for (int i = 0; i < TM * TN; i++) {
    int iiRow = i / TN;
    int iiCol = i % TN;
    c[(eRow + iRow + iiRow) * N + eCol + iCol + iiCol] = tmps[i];
  }
}

int main() {
  // Initialize h_a and h_b firstly.
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      h_a[row * K + col] = rand() % 100;
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