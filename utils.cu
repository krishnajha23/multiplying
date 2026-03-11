#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

void init_matrix(float* mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

bool matrices_equal(float* A, float* B, int n, float tol) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(A[i] - B[i]) > tol) {
            printf("Mismatch at index %d: %f vs %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

double compute_gflops(int n, double time_ms) {
    double flops = 2.0 * (double)n * (double)n * (double)n;
    double time_s = time_ms / 1000.0;
    return (flops / time_s) / 1e9;
}
