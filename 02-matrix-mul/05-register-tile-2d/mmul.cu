#include <stdio.h>
#include <cassert> 

// There is just about ~25% improvement compared with 1d tile
// version on RTX 3080Ti.

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

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) matrixMul(data_type* a, data_type* b, data_type* c) {
    int iRow = threadIdx.y * TM; // For threads in block: range(0, 128, 8)
    int iCol = threadIdx.x * TN; // For threads in block: range(0, 128, 8)

    int eRow = blockIdx.y * blockDim.y * TM;
    int eCol = blockIdx.x * blockDim.x * TN;

    __shared__ data_type s_a[BM * BK];
    __shared__ data_type s_b[BK * BN];

    data_type tmps[TM * TN] = {0};
    data_type regA[TM] = {0};
    data_type regB[TN] = {0};
    // 1st loop is for iterating over the two whole matrixs.
    for(int i = 0; i < K; i += BK) {
        // Load GMEM into SMEM
        // The two asserts is nesscessry for the below load, otherwise
        // we couldn't load the data into SMEM fully.
        assert(blockDim.y >= BK);
        assert(blockDim.x >= BK);
        for(int sar = 0; sar < TM; sar ++) {
            if(threadIdx.x < BK) {
                s_a[(iRow + sar) * BK + threadIdx.x] = a[(iRow + eRow + sar) * K + i + threadIdx.x];
            }
        }
        for(int sbc = 0; sbc < TN; sbc ++) {
            if(threadIdx.y < BK) {
                s_b[threadIdx.y * BN + iCol + sbc] = b[(i + threadIdx.y) * N + eCol + iCol + sbc];
            }
        }

        __syncthreads();

        // 2nd loop interate over a block.
        for(int j = 0; j < BK; j ++) {
            // Load the element from SMEM into register, but I understand
            // for 3rd loop the register array regA is redudant, we could
            // just use a register to do it
            for(int ra = 0; ra < TM; ra ++) {
                regA[ra] = s_a[(iRow + ra) * BK + j];
            }
            for(int rb = 0; rb < TN; rb ++) {
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
            for(int k = 0; k < TN; k ++) {
                // 4th loop for iterator row over matrix A.
                for(int s= 0; s < TM; s ++) {
                    tmps[s * TN + k] += regA[s] * regB[k];
                }
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < TM * TN; i ++) {
        int iiRow = i / TN;
        int iiCol = i % TN;
        c[(eRow + iRow + iiRow) * N + eCol + iCol + iiCol] = tmps[i];
    }
}

void verify_results(data_type* a, data_type* b, data_type* c, int N){
    for(int row = 0; row < M; row ++){
        for(int col = 0; col < N; col ++){
            data_type a_times_b = 0;
            for(int i = 0; i < K; i++){
                a_times_b += a[row * K + i] * b[N * i + col];
            }
            if(a_times_b != c[row * N + col]){
                printf("the result is wrong at row: %d, column: %d\n", row, col);
                printf("it should be %f, but it's %f\n", a_times_b, c[row * N + col]);
                abort();
            }
        }
    }
}

int main(){
    // Initialize h_a and h_b firstly.
    for(int row = 0; row < N; row ++){
        for(int col = 0; col < N; col ++){
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

    const dim3 threads(BN/TN, BM/TM);
    const dim3 blocks(BLOCK_X, BLOCK_Y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    for(int i = 0; i < 100; i ++){
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