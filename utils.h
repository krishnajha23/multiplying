#ifndef UTILS_H
#define UTILS_H

void init_matrix(float* mat, int n);
bool matrices_equal(float* A, float* B, int n, float tol = 1e-3);
double compute_gflops(int n, double time_ms);

#endif
