#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>

#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __LINE__ << ": "               \
                      << cudaGetErrorString(err) << std::endl;              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

// Naive GEMM kernel (global memory only, simple 4-level loop)
__global__ void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = A[row * K + k];
            float b = B[k * N + col];
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

float benchmark_naive(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    naive_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}


using Element = float;
using Layout = cutlass::layout::RowMajor;
using Gemm = cutlass::gemm::device::Gemm<Element, Layout, Element, Layout,
                                        Element, Layout>;

float benchmark_cutlass(Gemm::Arguments args) {
    Gemm gemm_op;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    cutlass::Status status = gemm_op(args);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed.\n";
        exit(EXIT_FAILURE);
    }

    return ms;
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    float alpha = 1.0f, beta = 0.0f;

    // Create CUTLASS host tensors
    cutlass::HostTensor<Element, Layout> A({M, K});
    cutlass::HostTensor<Element, Layout> B({K, N});
    cutlass::HostTensor<Element, Layout> C({M, N});
    cutlass::HostTensor<Element, Layout> D({M, N});

    cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 1, 0.0f, 1.0f, 0);
    cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 1, 0.0f, 1.0f, 17);
    cutlass::reference::host::TensorFill(C.host_view(), Element(0));
    cutlass::reference::host::TensorFill(D.host_view(), Element(0));

    A.sync_device();
    B.sync_device();
    C.sync_device();
    D.sync_device();

    // Setup CUTLASS GEMM
    Gemm::Arguments cutlass_args(
        {M, N, K},
        A.device_ref(),
        B.device_ref(),
        C.device_ref(),
        D.device_ref(),
        {alpha, beta}
    );

    // Benchmark CUTLASS
    float cutlass_time = benchmark_cutlass(cutlass_args);

    // Allocate and copy for naive GEMM
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeof(float) * M * K));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(float) * K * N));
    CHECK_CUDA(cudaMalloc(&d_C, sizeof(float) * M * N));
    CHECK_CUDA(cudaMemcpy(d_A, A.device_data(), sizeof(float) * M * K, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B.device_data(), sizeof(float) * K * N, cudaMemcpyDeviceToDevice));

    float naive_time = benchmark_naive(d_A, d_B, d_C, M, N, K);

    // Compute and print performance
    float gflops = 2.0f * M * N * K / 1e9f;
    std::cout << "CUTLASS Time: " << cutlass_time << " ms, " << (gflops / (cutlass_time / 1e3f)) << " GFLOPs\n";
    std::cout << "Naive   Time: " << naive_time   << " ms, " << (gflops / (naive_time / 1e3f)) << " GFLOPs\n";

    // Validate output
    D.sync_host();

    std::vector<float> host_naive(M * N);
    CHECK_CUDA(cudaMemcpy(host_naive.data(), d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    bool all_close = true;
    float eps = 1e-3f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(D.host_data()[i] - host_naive[i]);
        if (diff > eps) {
            std::cerr << "❌ Mismatch at index " << i << ": CUTLASS = "
                      << D.host_data()[i] << ", Naive = " << host_naive[i]
                      << ", Diff = " << diff << std::endl;
            all_close = false;
            break;
        }
    }

    if (all_close) {
        std::cout << "✅ Outputs match within tolerance " << eps << std::endl;
    } else {
        std::cerr << "❌ Outputs do NOT match!" << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
