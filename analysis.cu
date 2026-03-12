#include <stdio.h>
#include "analysis.h"
#include "config.h"

void print_bandwidth_analysis(
    int n,
    const BenchmarkResult& naive,
    const BenchmarkResult& tiled,
    const BenchmarkResult& optimized,
    const BenchmarkResult& vectorized,
    const BenchmarkResult& cublas
) {
    float n_float = (float)n;
    float bytes_read = 2.0f * n_float * n_float * n_float * sizeof(float);
    float bytes_written = n_float * n_float * sizeof(float);
    float total_bytes = bytes_read + bytes_written;

    float naive_bw = total_bytes / (naive.time_ms / 1000.0f) / 1e9f;
    float tiled_bw = total_bytes / (tiled.time_ms / 1000.0f) / 1e9f;
    float opt_bw = total_bytes / (optimized.time_ms / 1000.0f) / 1e9f;
    float vec_bw = total_bytes / (vectorized.time_ms / 1000.0f) / 1e9f;
    float cublas_bw = total_bytes / (cublas.time_ms / 1000.0f) / 1e9f;

    printf("\n=== Memory Bandwidth Analysis ===\n");
    printf("Naive global memory reads : %.2f GB\n", bytes_read / 1e9f);
    printf("Output writes             : %.2f GB\n", bytes_written / 1e9f);
    printf("Total data moved (naive)  : %.2f GB\n\n", total_bytes / 1e9f);

    printf("%-12s  %8s  %10s  %12s\n",
           "Kernel", "BW (GB/s)", "Peak util", "Tile reuse");
    printf("%-12s  %8.1f  %9.1f%%  %11s\n",
           "Naive", naive_bw, naive_bw / PEAK_BW_GBPS * 100.0f, "none");
    printf("%-12s  %8.1f  %9.1f%%  %8.0fx\n",
           "Tiled", tiled_bw, tiled_bw / PEAK_BW_GBPS * 100.0f, (float)n / TILE_SIZE);
    printf("%-12s  %8.1f  %9.1f%%  %8.0fx\n",
           "Optimized", opt_bw, opt_bw / PEAK_BW_GBPS * 100.0f, (float)n / OPT_TILE_SIZE);
    printf("%-12s  %8.1f  %9.1f%%  %8.0fx\n",
           "Vectorized", vec_bw, vec_bw / PEAK_BW_GBPS * 100.0f, (float)n / OPT_TILE_SIZE);
    printf("%-12s  %8.1f  %9.1f%%  %11s\n",
           "cuBLAS", cublas_bw, cublas_bw / PEAK_BW_GBPS * 100.0f, "internal");

    printf("\nNote: low bandwidth util + high GFLOPS = compute bound.\n");
    printf("cuBLAS uses Tensor Cores internally, so this bandwidth view is conservative.\n");
}

void print_gap_analysis(
    const BenchmarkResult& optimized,
    const BenchmarkResult& vectorized,
    const BenchmarkResult& cublas
) {
    printf("\n=== Gap Analysis vs cuBLAS ===\n");
    printf("cuBLAS GFLOPS          : %6.2f\n", cublas.gflops);
    printf("Optimized GFLOPS       : %6.2f\n", optimized.gflops);
    printf("Vectorized GFLOPS      : %6.2f\n", vectorized.gflops);
    printf("Optimized gap          : %6.2f GFLOPS (%.1fx slower)\n",
           cublas.gflops - optimized.gflops,
           optimized.time_ms / cublas.time_ms);
    printf("Vectorized gap         : %6.2f GFLOPS (%.1fx slower)\n",
           cublas.gflops - vectorized.gflops,
           vectorized.time_ms / cublas.time_ms);

    printf("\ncuBLAS closes the gap via:\n");
    printf("  - Tensor Core utilization\n");
    printf("  - Architecture-specific instruction scheduling\n");
    printf("  - Autotuned tile sizes per GPU model\n");
    printf("  - Double buffering to overlap compute and memory loads\n");
}

void print_final_summary(
    double cpu_time_ms,
    double cpu_gflops,
    const BenchmarkResult& naive,
    const BenchmarkResult& tiled,
    const BenchmarkResult& optimized,
    const BenchmarkResult& vectorized,
    const BenchmarkResult& cublas
) {
    printf("\n=== Final Summary ===\n");
    printf("%-12s  %8s  %10s  %10s\n", "Kernel", "Time(ms)", "GFLOPS", "vs cuBLAS");
    printf("%-12s  %8.2f  %10.2f  %9s\n",
           "CPU", cpu_time_ms, cpu_gflops, "baseline");
    printf("%-12s  %8.2f  %10.2f  %8.1fx\n",
           "Naive GPU", naive.time_ms, naive.gflops, naive.time_ms / cublas.time_ms);
    printf("%-12s  %8.2f  %10.2f  %8.1fx\n",
           "Tiled GPU", tiled.time_ms, tiled.gflops, tiled.time_ms / cublas.time_ms);
    printf("%-12s  %8.2f  %10.2f  %8.1fx\n",
           "Optimized", optimized.time_ms, optimized.gflops, optimized.time_ms / cublas.time_ms);
    printf("%-12s  %8.2f  %10.2f  %8.1fx\n",
           "Vectorized", vectorized.time_ms, vectorized.gflops, vectorized.time_ms / cublas.time_ms);
    printf("%-12s  %8.2f  %10.2f  %8s\n",
           "cuBLAS", cublas.time_ms, cublas.gflops, "peak");
}