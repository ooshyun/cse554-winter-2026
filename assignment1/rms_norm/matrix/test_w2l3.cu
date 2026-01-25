/**
 * CSE 554 Assignment 1 - Section 2: W2L3 Optimization Test
 * Test program for W2L3-inspired kernels
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../../common/gpu_specs.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Function declarations
extern void rms_norm_matrix_basic(const float*, float*, int, int);
extern void rms_norm_matrix_optimized(const float*, float*, int, int);
extern void rms_norm_matrix_fast(const float*, float*, int, int);
extern void rms_norm_matrix_w2l3_reduction(const float*, float*, int, int);
extern void rms_norm_matrix_w2l3_tile(const float*, float*, int, int);
extern void rms_norm_matrix_w2l3_hybrid(const float*, float*, int, int);
extern float measure_rms_norm_time(void (*)(const float*, float*, int, int),
                                   const float*, float*, int, int, int);
extern float calculate_rms_norm_bandwidth(int, int, float);


int main() {
    printf("========================================\n");
    printf("W2L3 Optimization Test\n");
    printf("========================================\n\n");

    const int rows = 8192;
    const int cols = 8192;
    const int n = rows * cols;
    const int num_iterations = 100;

    // Allocate host memory
    float* h_input = (float*)malloc(n * sizeof(float));

    // Initialize input
    srand(42);
    for (int i = 0; i < n; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    }

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                          cudaMemcpyHostToDevice));

    const float peak_bandwidth = GPU_PEAK_BANDWIDTH_DATASHEET;

    printf("Benchmarking kernels on %d x %d matrix (%d iterations)...\n\n",
           rows, cols, num_iterations);

    // Benchmark BASIC kernel
    printf("1. BASIC kernel:\n");
    float time_basic = measure_rms_norm_time(rms_norm_matrix_basic, d_input,
                                             d_output, rows, cols, num_iterations);
    float bandwidth_basic = calculate_rms_norm_bandwidth(rows, cols, time_basic);
    printf("   Time: %.4f ms | Bandwidth: %.2f GB/s (%.1f%% of peak)\n\n",
           time_basic, bandwidth_basic, (bandwidth_basic / peak_bandwidth) * 100.0f);

    // Benchmark OPTIMIZED kernel
    printf("2. OPTIMIZED kernel:\n");
    float time_optimized = measure_rms_norm_time(rms_norm_matrix_optimized,
                                                 d_input, d_output, rows, cols,
                                                 num_iterations);
    float bandwidth_optimized = calculate_rms_norm_bandwidth(rows, cols,
                                                             time_optimized);
    printf("   Time: %.4f ms | Bandwidth: %.2f GB/s (%.1f%% of peak)\n\n",
           time_optimized, bandwidth_optimized,
           (bandwidth_optimized / peak_bandwidth) * 100.0f);

    // Benchmark FAST kernel
    printf("3. FAST kernel:\n");
    float time_fast = measure_rms_norm_time(rms_norm_matrix_fast, d_input,
                                            d_output, rows, cols, num_iterations);
    float bandwidth_fast = calculate_rms_norm_bandwidth(rows, cols, time_fast);
    printf("   Time: %.4f ms | Bandwidth: %.2f GB/s (%.1f%% of peak)\n\n",
           time_fast, bandwidth_fast, (bandwidth_fast / peak_bandwidth) * 100.0f);

    // Benchmark W2L3 REDUCTION kernel
    printf("4. W2L3_REDUCTION kernel (reduction5-inspired):\n");
    printf("   - Multiple elements per thread (ELEMENTS_PER_THREAD = 32)\n");
    printf("   - Sequential addressing in shared memory\n");
    float time_w2l3_reduction = measure_rms_norm_time(rms_norm_matrix_w2l3_reduction,
                                                       d_input, d_output, rows, cols,
                                                       num_iterations);
    float bandwidth_w2l3_reduction = calculate_rms_norm_bandwidth(rows, cols,
                                                                   time_w2l3_reduction);
    printf("   Time: %.4f ms | Bandwidth: %.2f GB/s (%.1f%% of peak)\n",
           time_w2l3_reduction, bandwidth_w2l3_reduction,
           (bandwidth_w2l3_reduction / peak_bandwidth) * 100.0f);
    printf("   Speedup vs BASIC: %.2fx\n\n", time_basic / time_w2l3_reduction);

    // Benchmark W2L3 TILE kernel
    printf("5. W2L3_TILE kernel (transpose_v5-inspired):\n");
    printf("   - Large tile size (128 elements)\n");
    printf("   - Bank conflict avoidance with padding\n");
    float time_w2l3_tile = measure_rms_norm_time(rms_norm_matrix_w2l3_tile,
                                                  d_input, d_output, rows, cols,
                                                  num_iterations);
    float bandwidth_w2l3_tile = calculate_rms_norm_bandwidth(rows, cols,
                                                              time_w2l3_tile);
    printf("   Time: %.4f ms | Bandwidth: %.2f GB/s (%.1f%% of peak)\n",
           time_w2l3_tile, bandwidth_w2l3_tile,
           (bandwidth_w2l3_tile / peak_bandwidth) * 100.0f);
    printf("   Speedup vs BASIC: %.2fx\n\n", time_basic / time_w2l3_tile);

    // Benchmark W2L3 HYBRID kernel
    printf("6. W2L3_HYBRID kernel (combined optimizations):\n");
    printf("   - Multiple elements per thread + float4 vectorization\n");
    printf("   - Bank conflict avoidance\n");
    printf("   - Unrolled reduction loops\n");
    float time_w2l3_hybrid = measure_rms_norm_time(rms_norm_matrix_w2l3_hybrid,
                                                    d_input, d_output, rows, cols,
                                                    num_iterations);
    float bandwidth_w2l3_hybrid = calculate_rms_norm_bandwidth(rows, cols,
                                                                time_w2l3_hybrid);
    printf("   Time: %.4f ms | Bandwidth: %.2f GB/s (%.1f%% of peak)\n",
           time_w2l3_hybrid, bandwidth_w2l3_hybrid,
           (bandwidth_w2l3_hybrid / peak_bandwidth) * 100.0f);
    printf("   Speedup vs BASIC: %.2fx\n\n", time_basic / time_w2l3_hybrid);

    // Summary
    printf("========================================\n");
    printf("SUMMARY\n");
    printf("========================================\n");
    printf("Peak bandwidth: %.2f GB/s\n", peak_bandwidth);
    printf("Target: > 300 GB/s\n\n");

    printf("Kernel Performance Ranking:\n");
    float kernels[][2] = {
        {bandwidth_basic, 1},
        {bandwidth_optimized, 2},
        {bandwidth_fast, 3},
        {bandwidth_w2l3_reduction, 4},
        {bandwidth_w2l3_tile, 5},
        {bandwidth_w2l3_hybrid, 6}
    };

    // Simple bubble sort
    for (int i = 0; i < 6; i++) {
        for (int j = i + 1; j < 6; j++) {
            if (kernels[j][0] > kernels[i][0]) {
                float temp0 = kernels[i][0];
                float temp1 = kernels[i][1];
                kernels[i][0] = kernels[j][0];
                kernels[i][1] = kernels[j][1];
                kernels[j][0] = temp0;
                kernels[j][1] = temp1;
            }
        }
    }

    const char* kernel_names[] = {
        "BASIC", "OPTIMIZED", "FAST",
        "W2L3_REDUCTION", "W2L3_TILE", "W2L3_HYBRID"
    };

    for (int i = 0; i < 6; i++) {
        int idx = (int)kernels[i][1] - 1;
        printf("%d. %-20s: %.2f GB/s (%.1f%% of peak)\n",
               i + 1, kernel_names[idx], kernels[i][0],
               (kernels[i][0] / peak_bandwidth) * 100.0f);
    }

    printf("\nBest W2L3 kernel: ");
    float best_w2l3 = bandwidth_w2l3_reduction;
    const char* best_name = "W2L3_REDUCTION";
    if (bandwidth_w2l3_tile > best_w2l3) {
        best_w2l3 = bandwidth_w2l3_tile;
        best_name = "W2L3_TILE";
    }
    if (bandwidth_w2l3_hybrid > best_w2l3) {
        best_w2l3 = bandwidth_w2l3_hybrid;
        best_name = "W2L3_HYBRID";
    }
    printf("%s (%.2f GB/s)\n", best_name, best_w2l3);
    printf("========================================\n");

    // Cleanup
    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
