## WMMA
This folder will show some case about `wmma (tensor core)`.

### How to build
We must explictly assign a flag `-arch=sm_86` (>70 required) to compile with `wmma`, like `nvcc -arch=sm_86 -o a.o compare.cu`.

### Key APIs
```cuda
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);
```

The above class and method were the key APIs for `wmma`, for more details about these APIs please refer to this [doc](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-description).

### A very naive implement
Here is the example output for this [file](./01-naive-vs-wmma/compare.cu):
```
================ Performance Report ================
Matrix size: M=4096, N=4096, K=4096
Naive Kernel Time: 351.055 ms
Naive Kernel TFLOPs: 1.95751
WMMA  Kernel Time: 78.7553 ms
WMMA  Kernel TFLOPs: 8.7257
Max absolute difference between results: 0
====================================================
```
It shows **4x** gain for baseline, but both `wmma` and baseline implement are very naive, we could do futher experiment later.


### Classic GEMM implement
There is another GEMM, it utilize the classic 4 loops 2d tile GEMM, the example output came from this [file](./02-classic-gemm-vs-wmma/compare.cu):
```
================ Performance Report ================
Matrix size: M=4096, N=4096, K=4096
Classic Kernel Time: 51.1002 ms
Classic Kernel TFLOPs: 13.448
WMMA  Kernel Time: 83.3077 ms
WMMA  Kernel TFLOPs: 8.24888
Max absolute difference between results: 0
====================================================
```
It seems the `wmma` version is worse than the classic GEMM, but we could try to optimize it later.