## Cutlass
This folder contains some case for using cutlass.

Most important for this folder is update cutlass with `git submodule update --init --recursive` before build `.cu` file.

How to build `.cu` file: `nvcc -std=c++17 -arch=sm_86 -I../cutlass/include/ -I../cutlass/tools/util/include/ gemm.cu`, please make sure the path in `-I` flag is correctly point the cutlass include folder.
