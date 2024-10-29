#include <stdio.h>

// NOTE: Compare to 05 kernel, this kernel shos about 3% gain...
// It do improve performance with using sequenctial add, but the
// gain is too small..

const int N = 1 << 20;
const int BLOCK_NUM = 1 << 10;
const int THREAD_NUM = 1 << 10;
// block length should equal to thread length.
const int BLOCK_LENGTH = N / BLOCK_NUM;
const int ITERATION = 1000;

float h_a[N];

__global__ void sumReuction(float* a, float* r) {
  int threadId = threadIdx.x;
  int blockId = blockIdx.x;
  int offset = blockId * blockDim.x;

  __shared__ float smem[BLOCK_LENGTH];
  smem[threadId] = a[offset + threadId];
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadId < stride) {
      smem[threadId] += smem[threadId + stride];
    }
    __syncthreads();
  }

  __syncthreads();
  // printf("a[offset] is: %f\n", a[offset]);
  if (threadId == 0) atomicAdd(r, smem[0]);
}

void verify_result(float a, float b) {
  if (abs(a - b) > 1e-5 * abs(a)) {
    printf("res is wrong! it's %5f, it should be %5f\n", a, b);
  } else {
    printf("it's OK!\n");
  }
}

int main() {
  static_assert(N == BLOCK_NUM * THREAD_NUM);
  // Initialize host array first, and store the final sum result.
  double final_res = 0;
  for (int i = 0; i < N; i++) {
    // Couldn't use too big number here, otherwise the result may
    // not be very percise.
    // h_a[i] = rand() % 100;
    h_a[i] = 1;
    final_res += h_a[i];
  }

  float* d_a;
  float* d_r;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_r, sizeof(float));

  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

  const dim3 blocks(BLOCK_NUM);
  const dim3 threads(THREAD_NUM);

  // Run it once for verify result.
  sumReuction<<<blocks, threads>>>(d_a, d_r);

  float res = 0;
  cudaMemcpy(&res, d_r, sizeof(float), cudaMemcpyDeviceToHost);
  verify_result(res, final_res);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record start event
  cudaEventRecord(start);

  for (int i = 0; i < ITERATION; i++) {
    sumReuction<<<blocks, threads>>>(d_a, d_r);
  }

  // Record stop event
  cudaEventRecord(stop);
  cudaDeviceSynchronize();

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double FLOPs = N * ITERATION;
  float GFLOPS = FLOPs / (milliseconds * 1e6);

  printf("Kernel execution time: %f ms\n", milliseconds);
  printf("GFLOPS: %f gflops\n", GFLOPS);
  return 0;
}