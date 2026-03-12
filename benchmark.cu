#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "benchmark.h"
#include "kernels.h"
#include "utils.h"
#include "config.h"

BenchmarkResult benchmark_naive(
    float* d_A, float* d_B, float* d_C,
    float* h_C, float* h_ref, int n
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
              (n + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    naive_matmul<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_start);
    naive_matmul<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    BenchmarkResult result;
    cudaEventElapsedTime(&result.time_ms, ev_start, ev_stop);
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    result.gflops = compute_gflops(n, result.time_ms);
    result.correct = matrices_equal(h_ref, h_C, n);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return result;
}

BenchmarkResult benchmark_tiled(
    float* d_A, float* d_B, float* d_C,
    float* h_C, float* h_ref, int n
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
              (n + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    tiled_matmul<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_start);
    tiled_matmul<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    BenchmarkResult result;
    cudaEventElapsedTime(&result.time_ms, ev_start, ev_stop);
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    result.gflops = compute_gflops(n, result.time_ms);
    result.correct = matrices_equal(h_ref, h_C, n);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return result;
}

BenchmarkResult benchmark_optimized(
    float* d_A, float* d_B, float* d_C,
    float* h_C, float* h_ref, int n
) {
    dim3 block(OPT_TILE_SIZE / COARSE_FACTOR, OPT_TILE_SIZE);
    dim3 grid((n + OPT_TILE_SIZE - 1) / OPT_TILE_SIZE,
              (n + OPT_TILE_SIZE - 1) / OPT_TILE_SIZE);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    optimized_matmul<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_start);
    optimized_matmul<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    BenchmarkResult result;
    cudaEventElapsedTime(&result.time_ms, ev_start, ev_stop);
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    result.gflops = compute_gflops(n, result.time_ms);
    result.correct = matrices_equal(h_ref, h_C, n);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return result;
}

BenchmarkResult benchmark_vectorized(
    float* d_A, float* d_B, float* d_C,
    float* h_C, float* h_ref, int n
) {
    dim3 block(VECTOR_THREADS_X, OPT_TILE_SIZE);
    dim3 grid((n + OPT_TILE_SIZE - 1) / OPT_TILE_SIZE,
              (n + OPT_TILE_SIZE - 1) / OPT_TILE_SIZE);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    vectorized_matmul<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_start);
    vectorized_matmul<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    BenchmarkResult result;
    cudaEventElapsedTime(&result.time_ms, ev_start, ev_stop);
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    result.gflops = compute_gflops(n, result.time_ms);
    result.correct = matrices_equal(h_ref, h_C, n);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return result;
}

BenchmarkResult benchmark_cublas(
    float* d_A, float* d_B, float* d_C,
    float* h_C, float* h_ref, int n
) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &alpha,
                d_B, n, d_A, n,
                &beta, d_C, n);
    cudaDeviceSynchronize();

    cudaEventRecord(ev_start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &alpha,
                d_B, n, d_A, n,
                &beta, d_C, n);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    BenchmarkResult result;
    cudaEventElapsedTime(&result.time_ms, ev_start, ev_stop);
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    result.gflops = compute_gflops(n, result.time_ms);
    result.correct = matrices_equal(h_ref, h_C, n);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cublasDestroy(handle);

    return result;
}