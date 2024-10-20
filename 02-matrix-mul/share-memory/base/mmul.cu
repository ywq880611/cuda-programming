#include <stdio.h>
#include <cassert> 

#define data_type float

// Matrix dimensions, two matrixs:
// (M, K) and (K, N)
constexpr int M = (1 << 11) + 7;
constexpr int N = (1 << 10) + 7;
constexpr int K = (1 << 11) + 7;

const int THREAD_X = 32;
const int THREAD_Y = 32;

constexpr int K_stride = 32;

// Padded matrix dimensions
constexpr int M_padded = M + (THREAD_Y - M % THREAD_Y) % THREAD_Y;
constexpr int N_padded = N + (THREAD_X - N % THREAD_X) % THREAD_X;
constexpr int K_padded = K + (K_stride - K % K_stride) % K_stride;

data_type h_a[M_padded * K_padded];
data_type h_b[K_padded * N_padded];
data_type h_c[M * N];

const int a_bytes = M_padded * K_padded * sizeof(data_type);
const int b_bytes = K_padded * N_padded * sizeof(data_type);
const int c_bytes = M * N * sizeof(data_type);

__global__ void matrixMul(data_type* a, data_type* b, data_type* c) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    data_type tmp = 0;
    for(int i = 0; i < K; i ++){
        tmp += a[row * K_padded + i] * b[N_padded * i + col];
    }
    if (row < M && col < N) c[row * N + col] = tmp;
}

void verify_results(data_type* a, data_type* b, data_type* c, int N){
    for(int row = 0; row < M; row ++){
        for(int col = 0; col < N; col ++){
            data_type a_times_b = 0;
            for(int i = 0; i < K; i++){
                a_times_b += a[row * K_padded + i] * b[N_padded * i + col];
            }
            if(abs(a_times_b - c[row * N + col]) > 1e-3){
                printf("the result is wrong at row: %d, column: %d\n", row, col);
                //printf("it should be %d, but it's %d\n", a_times_b, c[row * N + col]);
                abort();
            }
        }
    }
}

int main(){
    // Initialize h_a and h_b firstly.
    for(int row = 0; row < N; row ++){
        for(int col = 0; col < N; col ++){
            h_a[row * N_padded + col] = rand() % 100;
            h_b[row * N_padded + col] = rand() % 100;
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
    const int BLOCK_X = N_padded / THREAD_X;
    const int BLOCK_Y = M_padded / THREAD_Y;

    const dim3 threads(THREAD_X, THREAD_Y);
    const dim3 blocks(BLOCK_X, BLOCK_Y);

    // NOTE: a detail, K_stride should be less or equal to blockDim.x
    // and blockDim.y, otherwise in the below loop, the s_a and s_b
    // shared memory couldn't be fully filled within blockDim.x * blockDim.y
    // threads.
    assert(K_stride <= threads.y);
    assert(K_stride <= threads.x);

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
    long long int FLOPs = 2LL * M * N * K;
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