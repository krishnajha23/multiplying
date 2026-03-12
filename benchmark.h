#ifndef BENCHMARK_H
#define BENCHMARK_H

struct BenchmarkResult {
    float time_ms;
    double gflops;
    bool correct;
};

BenchmarkResult benchmark_naive(
    float* d_A, float* d_B, float* d_C,
    float* h_C, float* h_ref, int n
);

BenchmarkResult benchmark_tiled(
    float* d_A, float* d_B, float* d_C,
    float* h_C, float* h_ref, int n
);

BenchmarkResult benchmark_optimized(
    float* d_A, float* d_B, float* d_C,
    float* h_C, float* h_ref, int n
);

BenchmarkResult benchmark_vectorized(
    float* d_A, float* d_B, float* d_C,
    float* h_C, float* h_ref, int n
);

BenchmarkResult benchmark_cublas(
    float* d_A, float* d_B, float* d_C,
    float* h_C, float* h_ref, int n
);

#endif