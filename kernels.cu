#include "kernels.h"
#include "config.h"

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

__global__ void vectorized_matmul(float* A, float* B, float* C, int n) {
    __shared__ float tile_A[OPT_TILE_SIZE][OPT_TILE_SIZE + 4];
    __shared__ float tile_B[OPT_TILE_SIZE][OPT_TILE_SIZE + 4];

    int row = blockIdx.y * OPT_TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * OPT_TILE_SIZE + threadIdx.x * VECTOR_COARSE_FACTOR;

    float sums[VECTOR_COARSE_FACTOR] = {0.0f, 0.0f, 0.0f, 0.0f};

    int num_tiles = (n + OPT_TILE_SIZE - 1) / OPT_TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * OPT_TILE_SIZE + threadIdx.x * VECTOR_WIDTH;
        if (row < n && a_col + 3 < n) {
            float4 a4 = reinterpret_cast<float4*>(&A[row * n + a_col])[0];
            tile_A[threadIdx.y][threadIdx.x * VECTOR_WIDTH + 0] = a4.x;
            tile_A[threadIdx.y][threadIdx.x * VECTOR_WIDTH + 1] = a4.y;
            tile_A[threadIdx.y][threadIdx.x * VECTOR_WIDTH + 2] = a4.z;
            tile_A[threadIdx.y][threadIdx.x * VECTOR_WIDTH + 3] = a4.w;
        } else {
            for (int c = 0; c < VECTOR_WIDTH; c++) {
                tile_A[threadIdx.y][threadIdx.x * VECTOR_WIDTH + c] =
                    (row < n && a_col + c < n) ? A[row * n + a_col + c] : 0.0f;
            }
        }

        int b_row = t * OPT_TILE_SIZE + threadIdx.y;
        int b_col = blockIdx.x * OPT_TILE_SIZE + threadIdx.x * VECTOR_WIDTH;
        if (b_row < n && b_col + 3 < n) {
            float4 b4 = reinterpret_cast<float4*>(&B[b_row * n + b_col])[0];
            tile_B[threadIdx.y][threadIdx.x * VECTOR_WIDTH + 0] = b4.x;
            tile_B[threadIdx.y][threadIdx.x * VECTOR_WIDTH + 1] = b4.y;
            tile_B[threadIdx.y][threadIdx.x * VECTOR_WIDTH + 2] = b4.z;
            tile_B[threadIdx.y][threadIdx.x * VECTOR_WIDTH + 3] = b4.w;
        } else {
            for (int c = 0; c < VECTOR_WIDTH; c++) {
                tile_B[threadIdx.y][threadIdx.x * VECTOR_WIDTH + c] =
                    (b_row < n && b_col + c < n) ? B[b_row * n + b_col + c] : 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < OPT_TILE_SIZE; k++) {
            float a_val = tile_A[threadIdx.y][k];
            for (int c = 0; c < VECTOR_COARSE_FACTOR; c++) {
                sums[c] += a_val * tile_B[k][threadIdx.x * VECTOR_COARSE_FACTOR + c];
            }
        }

        __syncthreads();
    }

    int out_col = col;
    if (row < n) {
        if (out_col + 3 < n) {
            float4 out = make_float4(sums[0], sums[1], sums[2], sums[3]);
            reinterpret_cast<float4*>(&C[row * n + out_col])[0] = out;
        } else {
            for (int c = 0; c < VECTOR_COARSE_FACTOR; c++) {
                if (out_col + c < n) {
                    C[row * n + out_col + c] = sums[c];
                }
            }
        }
    }
}