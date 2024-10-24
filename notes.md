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

### share memory
1. Use share memory to accelearte memory access, but it still have to access share memory / L1 cache, we could use register to make it run faster.
2. How to calculate the upper bound limitation of how much blocks could run on a SM.

### register-tile
1. use a little inner loop or a tmp variable to hold a element in register to avoid access share memory.
2. Why we use register tile improve perf? just register? it's a part, but from another point, we could do a comparasion for 2 case (take all of A, B and C with shape N * N):
    1. 1 thread calculates an element in final matrix C: in that case, for a specific `column 0` in C, we have to load `column 0` in B for `once` in `N` threads, so we load `column 0` in B for `N` times.
    2. 1 thread calculates a column in final matrix C (like 8 elements in a column): same as above case, we just need to load `column 0` in B for `once` in `N/8` threads, so we just load `column 0` in B for `N/8` times.

### register tile 2d
1. `__launch_bounds__` could use to tell compiler the maxmium of how many threads in a block we would like to launch, that could keep each threads could be allocated to enough registers to avoid load from cache or memory.
2. `NOTE`: make sure we didn't waste resource!!! In previous kernel of 05-register-2d kernel, we waste some thread in `if(threadIdx.x < BK)` and `if(threadIdx.y < BK)`. Hence if the thread didn't meet these condition, it will do nothing, so we couldn't load data from GMEM to SMEM ASAP.  
in such a case that the share memory size was not same as our thread count in a block, we could try to use a loop to load GMEM into SMEM to make all thread to load an element at least! please refer to how current `05-register-tile-2d` kernel load GMEM.

### vectorize
1. vectorize by loading/writing a 128 byte float between GMEM and SMEM, it could benefits perf.
2. On my kernel, the 128 byte SIMD code only works on writing process, I didn't know why it didn't work on loading process, but it also bring `~8%` gain from writing, maybe I could investigate on loading later.

## auto-tuning
1. Try to tune `BM, BN, BK, TM and TN` to see which combination could bring best performance to our kernel, it dependences on specific GPU spec.