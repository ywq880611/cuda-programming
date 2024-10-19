# Study notes for cuda programming

## vector add
### basic
1. pinned: it could accelerate the time cost from host to device or from device to host.  
in the case for vector add it improve time cost from `~6.8s` to `~4.1s`.
2. `__restrict__`: it didn't show perf gain for vector add case.

### prefetch
1. Use `cudaMallocManaged` could make the memory automative transfported between host and device, which helps us easy to write code, but it will hurt perf, because the CUDA runtime couldn't do the trans well.
2. use `cudaMemAdvise` and `cudaMemPrefetchAsync` could mitigate the regression mentioned above, but it's still not recommend to use managed memory in CUDA.

## Matrix multiply
### baseline
1. the navie implement of matrix multiply of `C = A * B` is like (assume all of `A, B and C` have shape `N * N`)  
    ```cpp
    for(int i = 0; i < N; i ++){
        c[row * N + col] += a[row * N + i] * b[N * i + col];
    }
    ```
2. How to define `row` and `col` is curial to CUDA perf, it's recommend to define them as:
    ```
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    ```
    Why we take `blockIdx.y` and `threadIdx.y` for `row`, but not `x`?
    Because in CUDA, a warp of thread always have a same `threadIdx.y`, but different `threadIdx.x`, so if we take `threadIdx.y` as row will reduce memory access. Please see detail in iPad notes for `Native natrix multiply`.

### noAliasing
1. `__restrict__`: it will benefits matrix multiply, it's `10ms vs 25ms`.  
maybe we could check `PTX` code for it later and compare with vector-add case to inspect why it didn't benefit vector-add.
2. tmp: use a tmp variable to avoid to add to matrix c for multiple times.
    ```cpp
    int tmp = 0;
    for(int i = 0; i < N; i ++){
        tmp += a[row * N + i] * b[N * i + col];
    }
    c[row * N + col] = tmp;
    ```
    It shows same perf gain as `__restrict__`, maybe they have same `PTX` code? check them later.