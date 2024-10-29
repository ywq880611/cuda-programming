#include <stdio.h>

#include <cassert>

// TODO: this kernel is wrong, it couldn't get the correct
// result, and didn't show any perf gain.

// Please rewrite it later!!!!

#define data_type float

// Matrix dimensions, two matrixs:
// (M, K) and (K, N)
constexpr int M = 1 << 10;
constexpr int N = 1 << 10;
constexpr int K = 1 << 10;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int WN = 64;
constexpr int WM = 64;
constexpr int TM = 8;
constexpr int TN = 4;
constexpr int WNITER = 4;
// NOTE: it's calculated, couldn't assign a random value!!!
constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);

// the read M for a warp was executed simutaneous
constexpr int WSUBM = WM / WMITER;
// the read N for a warp was executed simutaneous
constexpr int WSUBN = WN / WNITER;

constexpr int WARP_SIZE = 32;
// NOTE: Warps per block, it's calculated, couldn't assign a random value!!!
constexpr int NUM_WARPS = (BM * BN) / (WM * WN);
// NOTE :Threads per block, it's calculated, couldn't assign a random value!!!
constexpr int NUM_THREADS = WARP_SIZE * NUM_WARPS;

constexpr int BLOCK_X = N / BN;
constexpr int BLOCK_Y = M / BM;

data_type h_a[M * K];
data_type h_b[K * N];
data_type h_c[M * N];

constexpr int a_bytes = M * K * sizeof(data_type);
constexpr int b_bytes = K * N * sizeof(data_type);
constexpr int c_bytes = M * N * sizeof(data_type);

__device__ void loadFromGmem(int iRowA, int iColA, int iRowB, int iColB,
                             int offset, int eRow, int eCol, float *a, float *b,
                             float *s_a, float *s_b) {
  int rowStrideA = NUM_THREADS / (BK / 4);
  int rowStrideB = NUM_THREADS / (BN / 4);

  static_assert((BM * BK) % NUM_THREADS == 0);
  // The below check for it could be vectorized load;
  static_assert(((BM * BK) % NUM_THREADS) % 4 == 0);

  static_assert((BK * BN) % NUM_THREADS == 0);
  // The below check for it could be vectorized load;
  static_assert(((BK * BN) % NUM_THREADS) % 4 == 0);

  // each thread load thread_load_M_iter float4 in thread_load_M_iter row.
  for (int mr = iRowA; mr < BM; mr += rowStrideA) {
    reinterpret_cast<float4 *>(&s_a[(mr)*BK + iColA * 4])[0] =
        reinterpret_cast<float4 *>(&a[(mr + eRow) * K + iColA * 4])[0];
  }

  for (int nr = iRowB; nr < BK; nr += rowStrideB) {
    reinterpret_cast<float4 *>(&s_b[(nr)*BN + iColB * 4])[0] =
        reinterpret_cast<float4 *>(&b[(nr + offset) * N + eCol + iColB * 4])[0];
  }
}

__global__ void __launch_bounds__(NUM_THREADS)
    matrixMul(data_type *a, data_type *b, data_type *c) {
  int cRow = blockIdx.y * BM;
  int cCol = blockIdx.x * BN;

  int threadId = threadIdx.x;  // range(0, 128)
  const int warpIdx = threadId / WARP_SIZE;
  // The warpRow and WarpCol inside a (BM * BN) block.
  const int warpRow = warpIdx / (BN / WN);
  const int warpCol = warpIdx % (BN / WN);

  const int iWarpIdx = threadId % WARP_SIZE;
  int iRowSubC = (iWarpIdx / (WSUBN / TN)) * TM;
  int iColSubC = (iWarpIdx % (WSUBN / TN)) * TN;

  __shared__ data_type s_a[BM * BK];
  __shared__ data_type s_b[BK * BN];

  data_type tmps[WMITER * WNITER * TM * TN] = {0};
  data_type regA[WMITER * TM] = {0};
  data_type regB[WNITER * TN] = {0};
  // 1st loop is for iterating over the two whole matrixs.
  for (int i = 0; i < K; i += BK) {
    // Load GMEM into SMEM
    int iRowA = threadId / (BK / 4);
    int iColA = (threadId % (BK / 4));
    int iRowB = threadId / (BN / 4);
    int iColB = (threadId % (BN / 4));
    loadFromGmem(iRowA, iColA, iRowB, iColB, i, cRow, cCol, a, b, s_a, s_b);

    __syncthreads();

    // 2nd loop interate over a block.
    for (int j = 0; j < BK; j++) {
      // Load the element from SMEM into register.
      for (int wmi = 0; wmi < WMITER; wmi++) {
        for (int ra = 0; ra < TM; ra++) {
          regA[wmi * TM + ra] =
              s_a[(warpRow * WM + wmi * WSUBM + iRowSubC + ra) * BK + j];
        }
      }
      for (int wni = 0; wni < WNITER; wni++) {
        for (int rb = 0; rb < TN; rb++) {
          regB[wni * TN + rb] =
              s_b[j * BN + warpCol * WN + wni * WSUBN + iColSubC + rb];
        }
      }

      for (int wsubrow = 0; wsubrow < WMITER; wsubrow++) {
        for (int wsubcol = 0; wsubcol < WNITER; wsubcol++) {
          for (int k = 0; k < TN; k++) {
            for (int s = 0; s < TM; s++) {
              tmps[(wsubrow * TM + s) * (TN * WNITER) + wsubcol * TN + k] +=
                  regA[wsubrow * TM + s] * regB[wsubcol * TN + k];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  for (int wsubrow = 0; wsubrow < WMITER; wsubrow++) {
    for (int wsubcol = 0; wsubcol < WNITER; wsubcol++) {
      for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j += 4) {
          reinterpret_cast<float4 *>(
              &c[(cRow + warpRow * WM + wsubrow * WSUBM + iRowSubC + i) * N +
                 cCol + warpCol * WN + wsubcol * WSUBN + iColSubC + j])[0] =
              reinterpret_cast<float4 *>(
                  &tmps[(wsubrow * TM + i) * (TN * WNITER) + wsubcol * TN +
                        j])[0];
        }
      }
    }
  }
}

void verify_results(data_type *a, data_type *b, data_type *c, int N) {
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
      // h_a[row * N + col] = row;
      // h_b[row * N + col] = col;
    }
  }

  data_type *d_a;
  data_type *d_b;
  data_type *d_c;
  cudaMalloc(&d_a, a_bytes);
  cudaMalloc(&d_b, b_bytes);
  cudaMalloc(&d_c, c_bytes);

  cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice);

  // TODO: rethink the BLOCK_X and BLOCK_Y order. I thought it's
  // not important, we could switch them.

  static_assert(N % BLOCK_X == 0);
  static_assert(M % BLOCK_Y == 0);

  static_assert((BN % WN == 0) && (BM % WM == 0));
  static_assert((WN * WM) % (WNITER * WARP_SIZE * TN * TM) == 0);
  // TODO: I suspect the NUM_THREADS should be calculated rather than define
  // it directly, so the below formula should be a equation, but not a great
  // or equal, right???
  static_assert(NUM_THREADS * WNITER * WMITER * TM * TN >= BN * BM);

  const dim3 threads(NUM_THREADS);
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

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      // printf("h_c[%d][%d] is %f\n", i, j, h_c[i * N + j]);
    }
  }

  verify_results(h_a, h_b, h_c, N);

  printf("COMPLETED SUCCESSFULLY\n");

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}