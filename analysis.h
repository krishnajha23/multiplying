#ifndef ANALYSIS_H
#define ANALYSIS_H

#include "benchmark.h"

void print_bandwidth_analysis(
    int n,
    const BenchmarkResult& naive,
    const BenchmarkResult& tiled,
    const BenchmarkResult& optimized,
    const BenchmarkResult& vectorized,
    const BenchmarkResult& cublas
);

void print_gap_analysis(
    const BenchmarkResult& optimized,
    const BenchmarkResult& vectorized,
    const BenchmarkResult& cublas
);

void print_final_summary(
    double cpu_time_ms,
    double cpu_gflops,
    const BenchmarkResult& naive,
    const BenchmarkResult& tiled,
    const BenchmarkResult& optimized,
    const BenchmarkResult& vectorized,
    const BenchmarkResult& cublas
);

#endif