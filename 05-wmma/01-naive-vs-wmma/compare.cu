#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <chrono>

using namespace nvcuda;
using namespace std::chrono;

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

const int M = 4096;
const int N = 4096;
const int K = 4096;

__global__ void naive_gemm_half(const half* A, const half* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = __half2float(A[row * K + k]);
            float b = __half2float(B[k * N + col]);
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

__global__ void wmma_gemm(const half* A, const half* B, float* C, int M, int N, int K) {
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < K; i += 16) {
        int aRow = warpM * 16;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * 16;

        if (aRow < M && bCol < N) {
            const half* tileA = A + aRow * K + aCol;
            const half* tileB = B + bRow * N + bCol;

            wmma::load_matrix_sync(a_frag, tileA, K);
            wmma::load_matrix_sync(b_frag, tileB, N);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    if (warpM * 16 < M && warpN * 16 < N) {
        float* tileC = C + warpM * 16 * N + warpN * 16;
        wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
    }
}

void compare_gemm_kernels() {
    float *h_C_naive, *h_C_wmma;
    half *h_A, *h_B;
    half *d_A, *d_B;
    float *d_C_naive, *d_C_wmma;

    size_t size_half_AB = M * K * sizeof(half);
    size_t size_half_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);

    h_A = new half[M * K];
    h_B = new half[K * N];
    h_C_naive = new float[M * N];
    h_C_wmma = new float[M * N];

    for (int i = 0; i < M * K; ++i)
        h_A[i] = __float2half(static_cast<float>(i % 3 - 1));

    for (int i = 0; i < K * N; ++i)
        h_B[i] = __float2half(static_cast<float>((i % 5) - 2));

    CHECK_CUDA(cudaMalloc(&d_A, size_half_AB));
    CHECK_CUDA(cudaMalloc(&d_B, size_half_B));
    CHECK_CUDA(cudaMalloc(&d_C_naive, size_C));
    CHECK_CUDA(cudaMalloc(&d_C_wmma, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_half_AB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_half_B, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    // Time Naive
    auto start_naive = high_resolution_clock::now();
    naive_gemm_half<<<gridDim, blockDim>>>(d_A, d_B, d_C_naive, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_naive = high_resolution_clock::now();

    // Time WMMA
    dim3 grid_wmma(N / 16, M / 16);
    auto start_wmma = high_resolution_clock::now();
    wmma_gemm<<<grid_wmma, dim3(32)>>>(d_A, d_B, d_C_wmma, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_wmma = high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_C_naive, d_C_naive, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_wmma, d_C_wmma, size_C, cudaMemcpyDeviceToHost));

    // Validate
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(h_C_naive[i] - h_C_wmma[i]);
        if (diff > max_diff) max_diff = diff;
    }

    auto dur_naive = duration_cast<duration<double>>(end_naive - start_naive).count();
    auto dur_wmma  = duration_cast<duration<double>>(end_wmma  - start_wmma).count();

    // Compute TFLOPs: 2*M*N*K operations
    double ops = 2.0 * M * N * K;
    double tflops_naive = ops / (dur_naive * 1e12);
    double tflops_wmma  = ops / (dur_wmma * 1e12);

    std::cout << "================ Performance Report ================\n";
    std::cout << "Matrix size: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "Naive Kernel Time: " << dur_naive * 1e3 << " ms\n";
    std::cout << "Naive Kernel TFLOPs: " << tflops_naive << "\n";
    std::cout << "WMMA  Kernel Time: " << dur_wmma * 1e3 << " ms\n";
    std::cout << "WMMA  Kernel TFLOPs: " << tflops_wmma << "\n";
    std::cout << "Max absolute difference between results: " << max_diff << "\n";
    std::cout << "====================================================\n";

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_naive;
    delete[] h_C_wmma;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_wmma);
}

int main() {
    compare_gemm_kernels();
    return 0;
}
