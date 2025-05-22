# GEMM

GEMM is important part of CUDA, I will add a markdown for more details here.

Refer to [blog](https://chiemon.github.io/2020/02/06/CUDA-%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95-%E4%BC%98%E5%8C%96%E5%8F%8A%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90-%E4%B8%8A.html) and [code](https://github.com/siboehm/SGEMM_CUDA).

## How to build
Given we use `cublas` for verify correctness, so we have to build each `mmul.cu` file with `nvcc -arch=sm_86 -O3 -lcublas mmul.cu`.

## Performance table:
There is a table for GFLOPS on my local RTX 3080, the shape is **(4096, 4096) @ (4096, 4096)**.

| Kernel | GFLOPS | Performance relative to cuBLAS |
| ------ | ------ | ------------------------------ |
| cuBLAS | 22265.32 | 100% |
| 1.1: base-x-as-row | 59.83 | 0.3% |
| 1.2: base-y-as-row | 471.31 | 2.1% |
| 2.1: restrict | 1977.79 | 8.9% |
| 2.2: tmp (use register) | 1969.70 | 8.9% |
| 3: shared memory opt | 2410.00 | 10.8% |
| 4: 1d register tile | 5040.12 | 22.6% |
| 5: 2d register tile | 13947.22 | 62.6% |
| 6: vectorized | 11825.50 | 53.1% |
