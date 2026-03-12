#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

__global__ void naive_matmul(float* A, float* B, float* C, int n);
__global__ void tiled_matmul(float* A, float* B, float* C, int n);
__global__ void optimized_matmul(float* A, float* B, float* C, int n);
__global__ void vectorized_matmul(float* A, float* B, float* C, int n);

#endif