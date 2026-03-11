#include "matmul.h"

void cpu_matmul(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

__global__ void naive_matmul(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void tiled_matmul(float* A, float* B, float* C, int n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        As[threadIdx.y][threadIdx.x] =
            (row < n && t * TILE_SIZE + threadIdx.x < n)
                ? A[row * n + t * TILE_SIZE + threadIdx.x]
                : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (col < n && t * TILE_SIZE + threadIdx.y < n)
                ? B[(t * TILE_SIZE + threadIdx.y) * n + col]
                : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

__global__ void optimized_matmul(float* A, float* B, float* C, int n) {
    __shared__ float tile_A[OPT_TILE_SIZE][OPT_TILE_SIZE + 1];
    __shared__ float tile_B[OPT_TILE_SIZE][OPT_TILE_SIZE + 1];

    int row = blockIdx.y * OPT_TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * OPT_TILE_SIZE + threadIdx.x * COARSE_FACTOR;
    float sums[COARSE_FACTOR] = {0.0f};

    int num_tiles = (n + OPT_TILE_SIZE - 1) / OPT_TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int a_col = t * OPT_TILE_SIZE + threadIdx.x * COARSE_FACTOR + c;
            tile_A[threadIdx.y][threadIdx.x * COARSE_FACTOR + c] =
                (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        }

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int b_row = t * OPT_TILE_SIZE + threadIdx.y;
            int b_col = blockIdx.x * OPT_TILE_SIZE + threadIdx.x * COARSE_FACTOR + c;
            tile_B[threadIdx.y][threadIdx.x * COARSE_FACTOR + c] =
                (b_row < n && b_col < n) ? B[b_row * n + b_col] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < OPT_TILE_SIZE; k++) {
            float a_val = tile_A[threadIdx.y][k];
            for (int c = 0; c < COARSE_FACTOR; c++) {
                sums[c] += a_val * tile_B[k][threadIdx.x * COARSE_FACTOR + c];
            }
        }

        __syncthreads();
    }

    for (int c = 0; c < COARSE_FACTOR; c++) {
        if (row < n && (col + c) < n) {
            C[row * n + (col + c)] = sums[c];
        }
    }
}
