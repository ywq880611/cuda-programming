### TODO
1. ALL `TODO` in code.
2. ALL `NOTE` in code.
3. Check why `__restrict__` and `tmp` improve perf for matrix multiply but not for vector add, maybe could check from PTX code.
4. Think about why:
    ```cpp
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    ```
    is sightly better than 

    ```cpp
    uint x = blockIdx.y * blockDim.y + threadIdx.y;
    uint y = blockIdx.x * blockDim.x + threadIdx.x;
    ```
    in some case(espicially for some big matrix 4096 * 4096) for coalescing. Maybe `uint` takes more inst to cast type? check for PTX code later.