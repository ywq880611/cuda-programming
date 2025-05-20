#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <chrono>

using namespace nvcuda;
using namespace std::chrono;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

const int M = 4096;
const int N = 4096;
const int K = 4096;

// NOTE: make sure M, N and K is multiple of BM, BN and BK 
const int BM = 128;
const int BN = 128;
const int BK = 16;

const int TM = 8;
const int TN = 8;


__global__ void classic_gemm_half(const half* A, const half* B, float* C, int M, int N, int K) {
    const int bRow = blockIdx.x * BM;
    const int bCol = blockIdx.y * BN;
    // NOTE: because both A and B is row major tensor, so we could just utilize
    // memory coalescing for columns (adjacent thread will read data in same
    // row but adjacent columns), so we have to make columns index for thread
    // index at x.
    // TODO: but we could optimize it by transpose another matrix with column
    // major.
    const int tRow = threadIdx.y * TM;
    const int tCol = threadIdx.x * TN;

    const int block_size = blockDim.y * blockDim.x;

    __shared__ half sh_a[BM * BK];
    __shared__ half sh_b[BK * BN];
    float tmp_output[TM * TN] = {0};
    // 1st loop to load BM * BK or BK * BN data from HBM to SRAM.
    for(int i = 0; i < K; i += BK) {
        // NOTE: given `M == N == K`, and `BM == BN == BK`, and `BM * BK ==
        // number of threads in a block * TM * TN`, so each thread just load
        // TM * TN elements in to SRAM.
        int sIdx = threadIdx.y * blockDim.x + threadIdx.x;
        for(; sIdx < BM * BK; sIdx += block_size) {
            // TODO: add more loop or check here to support when BK != BM.
            int sRowA = sIdx / BK;
            int sColA = sIdx % BK;
            int sRowB = sIdx / BN;
            int sColB = sIdx % BN;

            sh_a[sIdx] = A[(bRow + sRowA) * K + sColA + i];
            sh_b[sIdx] = B[(sRowB + i) * N + bCol + sColB];
        }
        __syncthreads();

        float reg_a[TM] = {0};
        float reg_b[TN] = {0};
        // 2nd loop for each thread over BK and load data from SRAM to
        // registers, for matrix A, it loads TM elements, for matrix B,
        // it loads TN elements, so if we dot the TM * TN, we will get
        // a matrix with TM * TN as tmp output.
        for (int j = 0; j < BK; j ++) {
            for (int ra = 0; ra < TM; ra ++) {
                // we also convert type here.
                reg_a[ra] = __half2float(sh_a[(tRow + ra) * BK + j]);
            }
            for(int rb = 0; rb < TN; rb ++) {
                // we also convert type here.
                reg_b[rb] = __half2float(sh_b[j * BN + tCol + rb]);
            }

            // 3rd and 4th loop for calculate the dot product for the TM
            // mutiply by TN.
            // TODO: How about change the TM and TN loop? whether there is a
            // better performance?
            for(int k = 0; k < TM; k ++) {
                for(int s = 0; s < TN; s ++) {
                    tmp_output[k * TN + s] += reg_a[k] * reg_b[s];
                }
            }
        }
        __syncthreads();
    }

    // write back to C
    for(int i = 0; i < TM * TN; i ++) {
        int iRow = i / TN;
        int iCol = i % TN;

        C[(bRow + tRow + iRow) * N + bCol + tCol + iCol] = tmp_output[i];
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
    const int round = 5;

    float *h_C_classic, *h_C_wmma;
    half *h_A, *h_B;
    half *d_A, *d_B;
    float *d_C_classic, *d_C_wmma;

    size_t size_half_AB = M * K * sizeof(half);
    size_t size_half_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(float);

    h_A = new half[M * K];
    h_B = new half[K * N];
    h_C_classic = new float[M * N];
    h_C_wmma = new float[M * N];

    for (int i = 0; i < M * K; ++i)
        h_A[i] = __float2half(static_cast<float>(i % 3 - 1));

    for (int i = 0; i < K * N; ++i)
        h_B[i] = __float2half(static_cast<float>((i % 5) - 2));

    CHECK_CUDA(cudaMalloc(&d_A, size_half_AB));
    CHECK_CUDA(cudaMalloc(&d_B, size_half_B));
    CHECK_CUDA(cudaMalloc(&d_C_classic, size_C));
    CHECK_CUDA(cudaMalloc(&d_C_wmma, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_half_AB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_half_B, cudaMemcpyHostToDevice));

    dim3 blockDim(BM / TM, BN / TN);
    dim3 gridDim(M / BM, N / BN);

    // warm up
    for(int i = 0; i < round; i ++) {
        classic_gemm_half<<<gridDim, blockDim>>>(d_A, d_B, d_C_classic, M, N, K);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Time Classic
    auto start_classic = high_resolution_clock::now();
    for(int i = 0; i < round; i ++) {
        classic_gemm_half<<<gridDim, blockDim>>>(d_A, d_B, d_C_classic, M, N, K);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    auto end_classic = high_resolution_clock::now();

    // Time WMMA
    dim3 grid_wmma(N / 16, M / 16);
    auto start_wmma = high_resolution_clock::now();
    for(int i = 0; i < round; i ++) {
        wmma_gemm<<<grid_wmma, dim3(32)>>>(d_A, d_B, d_C_wmma, M, N, K);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    auto end_wmma = high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_C_classic, d_C_classic, size_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_wmma, d_C_wmma, size_C, cudaMemcpyDeviceToHost));

    // Validate
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(h_C_classic[i] - h_C_wmma[i]);
        if (diff > max_diff) max_diff = diff;
    }

    auto dur_classic = duration_cast<duration<double>>(end_classic - start_classic).count();
    auto dur_wmma  = duration_cast<duration<double>>(end_wmma  - start_wmma).count();

    // Compute TFLOPs: 2*M*N*K operations
    double ops = 2.0 * M * N * K * round;
    double tflops_classic = ops / (dur_classic * 1e12);
    double tflops_wmma  = ops / (dur_wmma * 1e12);

    std::cout << "================ Performance Report ================\n";
    std::cout << "Matrix size: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "Classic Kernel Time: " << dur_classic * 1e3 << " ms\n";
    std::cout << "Classic Kernel TFLOPs: " << tflops_classic << "\n";
    std::cout << "WMMA  Kernel Time: " << dur_wmma * 1e3 << " ms\n";
    std::cout << "WMMA  Kernel TFLOPs: " << tflops_wmma << "\n";
    std::cout << "Max absolute difference between results: " << max_diff << "\n";
    std::cout << "====================================================\n";

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_classic;
    delete[] h_C_wmma;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_classic);
    cudaFree(d_C_wmma);
}

int main() {
    compare_gemm_kernels();
    return 0;
}
