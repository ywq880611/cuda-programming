#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

__global__ void vector_add(int* a, int* b, int* c, int N){
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < N) {
        // CUDA's printf has certain limitations, including how many outputs can be 
        // printed depending on the execution context and available resources. 
        // If a large number of threads are executing, not all print statements may
        // be visible. This issue can become more prominent in larger grids or blocks.
        //printf("kernel id: %dd\n", tid);
        c[tid] = a[tid] + b[tid];
    }
}

void check_result(vector<int>& a, vector<int>& b, vector<int>& c) {
    for(int i = 0; i < a.size(); i ++){
        if(c[i] != a[i] + b[i]){
            printf("i is: %d\n", i);
            printf("a[i]: %d, b[i]: %d, c[i]: %d is not match!\n", a[i], b[i], c[i]);
            abort();
        }
    }
}

int main(){
    const int N = 1 << 26;
    const int bytes = N * sizeof(int);

    vector<int> a;
    a.reserve(N);
    vector<int> b;
    b.reserve(N);
    vector<int> c;
    c.reserve(N);

    for(int i = 0; i < N; i ++){
        a.push_back(rand() % 100);
        b.push_back(rand() % 100);
    }

    int* a_d;
    int* b_d;
    int* c_d;
    cudaMalloc(&a_d, bytes);
    cudaMalloc(&b_d, bytes);
    cudaMalloc(&c_d, bytes);

    cudaMemcpy(a_d, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n_thread = 1 << 10; // 1024 threads
    int n_block = (N + n_thread - 1) / n_thread;

    // Record start event
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        vector_add<<<n_block, n_thread>>>(a_d, b_d, c_d, N);
    }
    
    // Record stop event
    cudaEventRecord(stop);

    //cudaDeviceSynchronize();

    // This method will sync the previous launched kernel.
    cudaMemcpy(c.data(), c_d, bytes, cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds << " ms\n";

    check_result(a, b, c);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    return 0;
}

