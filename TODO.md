### TODO
1. ALL `TODO` in code.
2. ALL `NOTE` in code.
3. Check why `__restrict__` and `tmp` improve perf for matrix multiply but not for vector add, maybe could check from PTX code.
4. Think about why:
    ```cpp
    const int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    ```
    is sightly better than 

    ```
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    ```
    in some case(espicially for some big matrix 4096 * 4096) for coalescing.