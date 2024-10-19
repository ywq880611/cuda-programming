#include <stdio.h>

// Matrix dimensions, two matrixs:
// (M, K) and (K, N)
constexpr int M = (1 << 9) + 7;
constexpr int N = (1 << 10) + 7;
constexpr int K = (1 << 11) + 7;

const int THREAD = 32;

const int SHMEM_SIZE = THREAD * THREAD;

// Padded matrix dimensions
constexpr int M_padded = M + (THREAD - M % THREAD) % THREAD;
constexpr int N_padded = N + (THREAD - N % THREAD) % THREAD;
constexpr int K_padded = K + (THREAD - K % THREAD) % THREAD;

int h_a[M_padded * K_padded];
int h_b[K_padded * N_padded];
int h_c[M * N];

const int a_bytes = M_padded * K_padded * sizeof(int);
const int a_bytes = K_padded * N_padded * sizeof(int);
const int a_bytes = M * N * sizeof(int);

__global__ void matrixMul(int* a, int* b, int* c) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Statically allocated shared memory
    __shared__ int s_a[SHMEM_SIZE];
    __shared__ int s_b[SHMEM_SIZE];

    int tmp = 0;
    for(int i = 0; i < K_padded; i += blockDim.x){
        c[row * N + col] += a[row * N + i] * b[N * i + col];
    }
}

void verify_results(int* a, int* b, int* c, int N){
    for(int row = 0; row < M_padded; row ++){
        for(int col = 0; col < N_padded; col ++){
            int a_times_b = 0;
            for(int i = 0; i < K; i++){
                a_times_b += a[row * K_padded + i] * b[N_padded * i + col];
            }
            if(a_times_b != c[row * N + col]){
                printf("the result is wrong at row: %d, column: %d\n", row, col);
                printf("it should be %d, but it's %d\n", a_times_b, c[row * N + col]);
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

    int* d_a;
    int* d_b;
    int* d_c;
    cudaMalloc(&d_a, a_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes);

    cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice);

    // TODO: rethink the BLOCK_X and BLOCK_Y order. I thought it's
    // not important, we could switch them.
    const int BLOCK_X = N_padded / THREAD;
    const int BLOCK_Y = M_padded / THREAD;

    const dim3 threads(THREAD, THREAD);
    const dim3 blocks(BLOCK_X, BLOCK_Y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    for(int i = 0; i < 10; i ++){
        matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
    }

    // Record stop event
    cudaEventRecord(stop);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %f ms\n", milliseconds);

    verify_results(h_a, h_b, h_c, N);

    printf("COMPLETED SUCCESSFULLY\n");

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}