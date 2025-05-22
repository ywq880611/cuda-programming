#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>


void cublass_gemm(const int M, const int N, const int K, float* d_a, float* d_b, float* d_c) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  const float alpha = 1.0;
  const float beta = 0.0;

  // cublass use column-major tensor as input, therefore we have to convert the
  // A @ B = (B^T @ A^T)^T
  // (M, K) @ (K, N) = [(N, K) @ (K, M)]^T = (N, M)^T
  cublasGemmEx(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,
    &alpha,
    d_b, CUDA_R_32F, N,
    d_a, CUDA_R_32F, K,
    &beta,
    d_c, CUDA_R_32F, N,
    CUDA_R_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cudaDeviceSynchronize();
  cublasDestroy(handle);
}

void verify_with_cublas(const int M, const int N, const int K, float* d_a, float* d_b, float* d_c) {
  const size_t size_c = M * N * sizeof(float);

  float* cublas_d_c;
  cudaMalloc(&cublas_d_c, size_c);
  cublass_gemm(M, N, K, d_a, d_b, cublas_d_c);
  
  float* h_c;
  h_c = reinterpret_cast<float*>(malloc(size_c));
  cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

  float* cublas_c;
  cublas_c = reinterpret_cast<float*>(malloc(size_c));
  cudaMemcpy(cublas_c, cublas_d_c, size_c, cudaMemcpyDeviceToHost);
  float epsilon = 1e-5;

  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      const int offset = row * N + col;
      float cublass_res = cublas_c[offset];
      float verify_res = h_c[offset];

      if (abs(cublass_res - verify_res) > epsilon) {
        printf("the result is wrong at row: %d, column: %d\n", row, col);
        printf("it should be %f, but it's %f\n", cublass_res, verify_res);
        abort();
      }
    }
  }
}