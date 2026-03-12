#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "config.h"
#include "cpu_reference.h"
#include "utils.h"
#include "benchmark.h"
#include "analysis.h"

int main() {
    int size = N * N * sizeof(float);

    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C_cpu = (float*)malloc(size);
    float* C_naive = (float*)malloc(size);
    float* C_tiled = (float*)malloc(size);
    float* C_opt = (float*)malloc(size);
    float* C_vec = (float*)malloc(size);
    float* C_cublas = (float*)malloc(size);

    srand(42);
    init_matrix(A, N);
    init_matrix(B, N);

    double total_flops = compute_total_flops(N);
    printf("Matrix size : %d x %d\n", N, N);
    printf("Total FLOPs : %.2e  (%.2f GFLOPs)\n\n", total_flops, total_flops / 1e9);

    clock_t start = clock();
    cpu_matmul(A, B, C_cpu, N);
    clock_t end = clock();

    double cpu_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    double cpu_gflops = compute_gflops(N, cpu_time_ms);

    printf("CPU:        %8.2f ms  |  %6.2f GFLOPS\n", cpu_time_ms, cpu_gflops);

    float *d_A, *d_B, *d_C_naive, *d_C_tiled, *d_C_opt, *d_C_vec, *d_C_cublas;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C_naive, size);
    cudaMalloc(&d_C_tiled, size);
    cudaMalloc(&d_C_opt, size);
    cudaMalloc(&d_C_vec, size);
    cudaMalloc(&d_C_cublas, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    BenchmarkResult naive = benchmark_naive(d_A, d_B, d_C_naive, C_naive, C_cpu, N);
    printf("Naive GPU:  %8.2f ms  |  %6.2f GFLOPS  |  %s\n",
           naive.time_ms, naive.gflops, naive.correct ? "CORRECT" : "WRONG");

    BenchmarkResult tiled = benchmark_tiled(d_A, d_B, d_C_tiled, C_tiled, C_cpu, N);
    printf("Tiled GPU:  %8.2f ms  |  %6.2f GFLOPS  |  %s\n",
           tiled.time_ms, tiled.gflops, tiled.correct ? "CORRECT" : "WRONG");

    BenchmarkResult optimized = benchmark_optimized(d_A, d_B, d_C_opt, C_opt, C_cpu, N);
    printf("Optimized:  %8.2f ms  |  %6.2f GFLOPS  |  %s\n",
           optimized.time_ms, optimized.gflops, optimized.correct ? "CORRECT" : "WRONG");

    BenchmarkResult vectorized = benchmark_vectorized(d_A, d_B, d_C_vec, C_vec, C_cpu, N);
    printf("Vectorized: %8.2f ms  |  %6.2f GFLOPS  |  %s\n",
           vectorized.time_ms, vectorized.gflops, vectorized.correct ? "CORRECT" : "WRONG");

    BenchmarkResult cublas = benchmark_cublas(d_A, d_B, d_C_cublas, C_cublas, C_cpu, N);
    printf("cuBLAS:     %8.2f ms  |  %6.2f GFLOPS  |  %s\n",
           cublas.time_ms, cublas.gflops, cublas.correct ? "CORRECT" : "WRONG");

    print_bandwidth_analysis(N, naive, tiled, optimized, vectorized, cublas);
    print_gap_analysis(optimized, vectorized, cublas);
    print_final_summary(cpu_time_ms, cpu_gflops, naive, tiled, optimized, vectorized, cublas);

    free(A);
    free(B);
    free(C_cpu);
    free(C_naive);
    free(C_tiled);
    free(C_opt);
    free(C_vec);
    free(C_cublas);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_tiled);
    cudaFree(d_C_opt);
    cudaFree(d_C_vec);
    cudaFree(d_C_cublas);

    return 0;
}