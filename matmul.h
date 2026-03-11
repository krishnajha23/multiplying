#ifndef MATMUL_H
#define MATMUL_H

#define N 1024
#define TILE_SIZE 16
#define OPT_TILE_SIZE 32
#define COARSE_FACTOR 2

void cpu_matmul(float* A, float* B, float* C, int n);

__global__ void naive_matmul(float* A, float* B, float* C, int n);
__global__ void tiled_matmul(float* A, float* B, float* C, int n);
__global__ void optimized_matmul(float* A, float* B, float* C, int n);

#endif
