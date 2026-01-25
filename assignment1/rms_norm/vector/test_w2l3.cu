/**
 * CSE 554 Assignment 1 - Section 2: W2L3 Vector Optimization Test
 * Test program for W2L3-inspired vector kernels
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
extern void rms_norm_vector_basic(const float*, float*, int);
extern void rms_norm_vector_fast(const float*, float*, int);
extern void rms_norm_vector_w2l3_reduction(const float*, float*, int);
extern void rms_norm_vector_w2l3_hybrid(const float*, float*, int);
extern float measure_rms_norm_vector_time(void (*)(const float*, float*, int),
                                          const float*, float*, int, int);
extern float calculate_rms_norm_vector_bandwidth(int, float);


int main() {
    printf("========================================\n");
    printf("W2L3 Vector Optimization Test\n");
    printf("========================================\n\n");

    const int n = 1024 * 1024;  // 1M elements
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

    printf("Benchmarking vector kernels (%d elements, %d iterations)...\n\n",
           n, num_iterations);

    // Benchmark BASIC kernel
    printf("1. BASIC kernel (two-phase):\n");
    float time_basic = measure_rms_norm_vector_time(rms_norm_vector_basic, d_input,
                                                     d_output, n, num_iterations);
    float bandwidth_basic = calculate_rms_norm_vector_bandwidth(n, time_basic);
    printf("   Time: %.4f ms | Bandwidth: %.2f GB/s (%.1f%% of peak)\n\n",
           time_basic, bandwidth_basic, (bandwidth_basic / peak_bandwidth) * 100.0f);

    // Benchmark FAST kernel
    printf("2. FAST kernel (GPU-only reduction):\n");
    float time_fast = measure_rms_norm_vector_time(rms_norm_vector_fast, d_input,
                                                    d_output, n, num_iterations);
    float bandwidth_fast = calculate_rms_norm_vector_bandwidth(n, time_fast);
    printf("   Time: %.4f ms | Bandwidth: %.2f GB/s (%.1f%% of peak)\n\n",
           time_fast, bandwidth_fast, (bandwidth_fast / peak_bandwidth) * 100.0f);

    // Benchmark W2L3 REDUCTION kernel
    printf("3. W2L3_REDUCTION kernel (reduction5-inspired):\n");
    printf("   - Multiple elements per thread (ELEMENTS_PER_THREAD = 64)\n");
    printf("   - Sequential addressing reduction\n");
    printf("   - Grid-stride loop pattern\n");
    float time_w2l3_reduction = measure_rms_norm_vector_time(
        rms_norm_vector_w2l3_reduction, d_input, d_output, n, num_iterations);
    float bandwidth_w2l3_reduction = calculate_rms_norm_vector_bandwidth(n, time_w2l3_reduction);
    printf("   Time: %.4f ms | Bandwidth: %.2f GB/s (%.1f%% of peak)\n",
           time_w2l3_reduction, bandwidth_w2l3_reduction,
           (bandwidth_w2l3_reduction / peak_bandwidth) * 100.0f);
    printf("   Speedup vs BASIC: %.2fx\n\n", time_basic / time_w2l3_reduction);

    // Benchmark W2L3 HYBRID kernel
    printf("4. W2L3_HYBRID kernel (reduction5 + vectorization):\n");
    printf("   - Multiple elements per thread (32 iterations × float4)\n");
    printf("   - Bank conflict avoidance\n");
    printf("   - Unrolled reduction\n");
    float time_w2l3_hybrid = measure_rms_norm_vector_time(
        rms_norm_vector_w2l3_hybrid, d_input, d_output, n, num_iterations);
    float bandwidth_w2l3_hybrid = calculate_rms_norm_vector_bandwidth(n, time_w2l3_hybrid);
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
        {bandwidth_fast, 2},
        {bandwidth_w2l3_reduction, 3},
        {bandwidth_w2l3_hybrid, 4}
    };

    // Simple bubble sort
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
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
        "BASIC", "FAST", "W2L3_REDUCTION", "W2L3_HYBRID"
    };

    for (int i = 0; i < 4; i++) {
        int idx = (int)kernels[i][1] - 1;
        printf("%d. %-20s: %.2f GB/s (%.1f%% of peak)\n",
               i + 1, kernel_names[idx], kernels[i][0],
               (kernels[i][0] / peak_bandwidth) * 100.0f);
    }

    printf("\nBest W2L3 kernel: ");
    if (bandwidth_w2l3_hybrid > bandwidth_w2l3_reduction) {
        printf("W2L3_HYBRID (%.2f GB/s)\n", bandwidth_w2l3_hybrid);
    } else {
        printf("W2L3_REDUCTION (%.2f GB/s)\n", bandwidth_w2l3_reduction);
    }
    printf("========================================\n");

    // Cleanup
    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
