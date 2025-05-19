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
Naive Kernel Time: 80.2572 ms
Naive Kernel TFLOPs: 1.71248
WMMA  Kernel Time: 18.6568 ms
WMMA  Kernel TFLOPs: 7.36668
Max absolute difference between results: 0
====================================================
```
It shows **4x** gain for baseline, but both `wmma` and baseline implement are very naive, we could do futher experiment later.