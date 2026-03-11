#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "matmul.h"
#include "utils.h"

int main() {
    int size = N * N * sizeof(float);

    double total_flops = 2.0 * (double)N * (double)N * (double)N;
    printf("Matrix size : %d x %d\n", N, N);
    printf("Total FLOPs : %.2e  (%.2f GFLOPs)\n\n", total_flops, total_flops / 1e9);

    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C_cpu = (float*)malloc(size);
    float* C_naive = (float*)malloc(size);
    float* C_tiled = (float*)malloc(size);
    float* C_opt = (float*)malloc(size);
    float* C_cublas = (float*)malloc(size);

    srand(42);
    init_matrix(A, N);
    init_matrix(B, N);

    clock_t start = clock();
    cpu_matmul(A, B, C_cpu, N);
    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    printf("CPU:        %8.2f ms  |  %6.2f GFLOPS\n",
           cpu_time, compute_gflops(N, cpu_time));

    float *d_A, *d_B, *d_C_naive, *d_C_tiled, *d_C_opt, *d_C_cublas;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C_naive, size);
    cudaMalloc(&d_C_tiled, size);
    cudaMalloc(&d_C_opt, size);
    cudaMalloc(&d_C_cublas, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    float naive_gpu_time, tiled_gpu_time, opt_gpu_time, cublas_time;

    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                  (N + TILE_SIZE - 1) / TILE_SIZE);

        naive_matmul<<<grid, block>>>(d_A, d_B, d_C_naive, N);
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        naive_matmul<<<grid, block>>>(d_A, d_B, d_C_naive, N);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        cudaEventElapsedTime(&naive_gpu_time, ev_start, ev_stop);
        cudaMemcpy(C_naive, d_C_naive, size, cudaMemcpyDeviceToHost);

        printf("Naive GPU:  %8.2f ms  |  %6.2f GFLOPS  |  %s\n",
               naive_gpu_time, compute_gflops(N, naive_gpu_time),
               matrices_equal(C_cpu, C_naive, N) ? "CORRECT" : "WRONG");
    }

    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                  (N + TILE_SIZE - 1) / TILE_SIZE);

        tiled_matmul<<<grid, block>>>(d_A, d_B, d_C_tiled, N);
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        tiled_matmul<<<grid, block>>>(d_A, d_B, d_C_tiled, N);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        cudaEventElapsedTime(&tiled_gpu_time, ev_start, ev_stop);
        cudaMemcpy(C_tiled, d_C_tiled, size, cudaMemcpyDeviceToHost);

        printf("Tiled GPU:  %8.2f ms  |  %6.2f GFLOPS  |  %s\n",
               tiled_gpu_time, compute_gflops(N, tiled_gpu_time),
               matrices_equal(C_cpu, C_tiled, N) ? "CORRECT" : "WRONG");
    }

    {
        dim3 block(OPT_TILE_SIZE / COARSE_FACTOR, OPT_TILE_SIZE);
        dim3 grid((N + OPT_TILE_SIZE - 1) / OPT_TILE_SIZE,
                  (N + OPT_TILE_SIZE - 1) / OPT_TILE_SIZE);

        optimized_matmul<<<grid, block>>>(d_A, d_B, d_C_opt, N);
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        optimized_matmul<<<grid, block>>>(d_A, d_B, d_C_opt, N);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        cudaEventElapsedTime(&opt_gpu_time, ev_start, ev_stop);
        cudaMemcpy(C_opt, d_C_opt, size, cudaMemcpyDeviceToHost);

        printf("Optimized:  %8.2f ms  |  %6.2f GFLOPS  |  %s\n",
               opt_gpu_time, compute_gflops(N, opt_gpu_time),
               matrices_equal(C_cpu, C_opt, N) ? "CORRECT" : "WRONG");
    }

    {
        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta = 0.0f;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha,
                    d_B, N, d_A, N,
                    &beta, d_C_cublas, N);
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha,
                    d_B, N, d_A, N,
                    &beta, d_C_cublas, N);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        cudaEventElapsedTime(&cublas_time, ev_start, ev_stop);
        cudaMemcpy(C_cublas, d_C_cublas, size, cudaMemcpyDeviceToHost);

        printf("cuBLAS:     %8.2f ms  |  %6.2f GFLOPS  |  %s\n",
               cublas_time, compute_gflops(N, cublas_time),
               matrices_equal(C_cpu, C_cublas, N) ? "CORRECT" : "WRONG");

        cublasDestroy(handle);
    }

    float N_float = (float)N;
    float bytes_read = 2.0f * N_float * N_float * N_float * sizeof(float);
    float bytes_written = N_float * N_float * sizeof(float);
    float total_bytes = bytes_read + bytes_written;
    float peak_bw = 300.0f;

    float naive_bw = total_bytes / (naive_gpu_time / 1000.0f) / 1e9f;
    float tiled_bw = total_bytes / (tiled_gpu_time / 1000.0f) / 1e9f;
    float opt_bw = total_bytes / (opt_gpu_time / 1000.0f) / 1e9f;
    float cublas_bw = total_bytes / (cublas_time / 1000.0f) / 1e9f;

    printf("\n=== Memory Bandwidth Analysis ===\n");
    printf("Naive global memory reads : %.2f GB\n", bytes_read / 1e9f);
    printf("Output writes             : %.2f GB\n", bytes_written / 1e9f);
    printf("Total data moved (naive)  : %.2f GB\n\n", total_bytes / 1e9f);

    printf("%-12s  %8s  %10s  %12s\n",
           "Kernel", "BW (GB/s)", "Peak util", "Tile reuse");
    printf("%-12s  %8.1f  %9.1f%%  %11s\n",
           "Naive", naive_bw, naive_bw / peak_bw * 100.0f, "none");
    printf("%-12s  %8.1f  %9.1f%%  %8.0fx\n",
           "Tiled", tiled_bw, tiled_bw / peak_bw * 100.0f, (float)N / TILE_SIZE);
    printf("%-12s  %8.1f  %9.1f%%  %8.0fx\n",
           "Optimized", opt_bw, opt_bw / peak_bw * 100.0f, (float)N / OPT_TILE_SIZE);
    printf("%-12s  %8.1f  %9.1f%%  %11s\n",
           "cuBLAS", cublas_bw, cublas_bw / peak_bw * 100.0f, "internal");

    printf("\nNote: low bandwidth util + high GFLOPS = compute bound (good).\n");
    printf("cuBLAS uses Tensor Cores internally — bandwidth analysis understates\n");
    printf("its efficiency since Tensor Core ops don't map 1:1 to DRAM traffic.\n");

    printf("\n=== Gap Analysis vs cuBLAS ===\n");
    printf("cuBLAS GFLOPS          : %6.2f\n", compute_gflops(N, cublas_time));
    printf("Optimized GFLOPS       : %6.2f\n", compute_gflops(N, opt_gpu_time));
    printf("Gap                    : %6.2f GFLOPS (%.1fx slower than cuBLAS)\n",
           compute_gflops(N, cublas_time) - compute_gflops(N, opt_gpu_time),
           cublas_time > 0 ? opt_gpu_time / cublas_time : 0.0f);

    printf("\ncuBLAS closes the gap via:\n");
    printf("  - Tensor Core utilization (not used in our kernels)\n");
    printf("  - Architecture-specific instruction scheduling\n");
    printf("  - Autotuned tile sizes per GPU model\n");
    printf("  - Double buffering to overlap compute and memory loads\n");

    printf("\n=== Final Summary ===\n");
    printf("%-12s  %8s  %10s  %10s\n", "Kernel", "Time(ms)", "GFLOPS", "vs cuBLAS");
    printf("%-12s  %8.2f  %10.2f  %9s\n",
           "CPU", cpu_time, compute_gflops(N, cpu_time), "baseline");
    printf("%-12s  %8.2f  %10.2f  %8.1fx\n",
           "Naive GPU", naive_gpu_time, compute_gflops(N, naive_gpu_time),
           cublas_time / naive_gpu_time);
    printf("%-12s  %8.2f  %10.2f  %8.1fx\n",
           "Tiled GPU", tiled_gpu_time, compute_gflops(N, tiled_gpu_time),
           cublas_time / tiled_gpu_time);
    printf("%-12s  %8.2f  %10.2f  %8.1fx\n",
           "Optimized", opt_gpu_time, compute_gflops(N, opt_gpu_time),
           cublas_time / opt_gpu_time);
    printf("%-12s  %8.2f  %10.2f  %8s\n",
           "cuBLAS", cublas_time, compute_gflops(N, cublas_time), "peak");

    free(A);
    free(B);
    free(C_cpu);
    free(C_naive);
    free(C_tiled);
    free(C_opt);
    free(C_cublas);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_tiled);
    cudaFree(d_C_opt);
    cudaFree(d_C_cublas);

    return 0;
}
