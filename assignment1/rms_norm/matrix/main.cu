/**
 * CSE 554 Assignment 1 - Section 2: RMS Norm Matrix Test
 * Main test file for matrix RMS Norm implementation
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

#define EPSILON 1e-6f

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

// Picked kernel: rms_norm_matrix_w2l3_hybrid
void (*picked_kernel)(const float*, float*, int, int) = rms_norm_matrix_w2l3_hybrid;


/**
 * CPU reference implementation of RMS Norm
 */
void rms_norm_cpu(const float* input, float* output, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        // Compute sum of squares for this row
        float sum_sq = 0.0f;
        for (int col = 0; col < cols; col++) {
            int idx = row * cols + col;
            float val = input[idx];
            sum_sq += val * val;
        }

        // Compute RMS
        float rms = sqrtf(sum_sq / cols + EPSILON);

        // Normalize
        for (int col = 0; col < cols; col++) {
            int idx = row * cols + col;
            output[idx] = input[idx] / rms;
        }
    }
}


/**
 * Verify CUDA results against CPU reference
 */
bool verify_rms_norm(const float* cuda_result, const float* cpu_result,
                    int rows, int cols, float tolerance = 1e-4) {
    float max_diff = 0.0f;
    int max_diff_row = 0;
    int max_diff_col = 0;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int idx = row * cols + col;
            float diff = fabsf(cuda_result[idx] - cpu_result[idx]);
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_row = row;
                max_diff_col = col;
            }
        }
    }

    printf("  Max difference: %.2e at [%d, %d]\n", max_diff, max_diff_row, max_diff_col);
    printf("    CUDA: %.6f, CPU: %.6f\n",
        cuda_result[max_diff_row * cols + max_diff_col],
        cpu_result[max_diff_row * cols + max_diff_col]);

    return max_diff < tolerance;
}


/**
 * Test small matrix for correctness
 */
void test_correctness() {
    printf("\n");
    printf("================================================================================\n");
    printf("CORRECTNESS TEST\n");
    printf("================================================================================\n");

    const int rows = 4;
    const int cols = 8;
    const int n = rows * cols;

    float h_input[n];
    float h_output_cpu[n];
    float h_output_cuda[n];

    // Initialize test data
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)(i % 10) - 5.0f;  // Values from -5 to 4
    }

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                        cudaMemcpyHostToDevice));

    // Run CPU reference
    printf("\nRunning CPU reference...\n");
    rms_norm_cpu(h_input, h_output_cpu, rows, cols);

    // Run CUDA kernel
    printf("Running CUDA kernel...\n");
    rms_norm_matrix_fast(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaMemcpy(h_output_cuda, d_output, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

    // Display results for first row
    printf("\nFirst row results:\n");
    printf("Input:  ");
    for (int col = 0; col < cols; col++) {
        printf("%6.2f ", h_input[col]);
    }
    printf("\nCPU:    ");
    for (int col = 0; col < cols; col++) {
        printf("%6.3f ", h_output_cpu[col]);
    }
    printf("\nCUDA:   ");
    for (int col = 0; col < cols; col++) {
        printf("%6.3f ", h_output_cuda[col]);
    }
    printf("\n");

    // Verify
    bool passed = verify_rms_norm(h_output_cuda, h_output_cpu, rows, cols);
    printf("\n✓ Correctness test: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


/**
 * Benchmark performance for matrix (8192, 8192)
 */
void benchmark_performance() {
    printf("\n");
    printf("================================================================================\n");
    printf("PERFORMANCE BENCHMARK (8192 x 8192 matrix)\n");
    printf("================================================================================\n");
#if defined(PROFILE_NCUS)
    const int num_iterations = 1;
#else
    const int num_iterations = 100;
#endif
    const int rows = 8192;
    const int cols = 8192;
    const int n = rows * cols;

    printf("Matrix size: %d x %d = %d elements\n", rows, cols, n);
    printf("Memory per matrix: %.2f MB\n", (n * sizeof(float)) / (1024.0f * 1024.0f));
    printf("Total memory (in + out): %.2f MB\n",
        (2 * n * sizeof(float)) / (1024.0f * 1024.0f));
    printf("Iterations: %d\n\n", num_iterations);

    // Allocate host memory
    float* h_input = (float*)malloc(n * sizeof(float));
    float* h_output_cpu = (float*)malloc(n * sizeof(float));
    float* h_output_cuda = (float*)malloc(n * sizeof(float));

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

    // Benchmark picked kernel
    // rms_norm_matrix_basic rms_norm_matrix_optimized rms_norm_matrix_fast 
    // rms_norm_matrix_w2l3_reduction rms_norm_matrix_w2l3_tile rms_norm_matrix_w2l3_hybrid
    printf("Testing RMS Norm Matrix kernel...\n");
    float time_picked = measure_rms_norm_time(picked_kernel, d_input, d_output, rows, cols, num_iterations);
    float bandwidth_picked = calculate_rms_norm_bandwidth(rows, cols, time_picked);
    const float peak_bandwidth_datasheet = GPU_PEAK_BANDWIDTH_DATASHEET;
    float percentage_picked = (bandwidth_picked / peak_bandwidth_datasheet) * 100.0f;
    printf("  Execution time: %.4f ms\n", time_picked);
    printf("  Bandwidth: %.2f GB/s (%.1f%% of peak %.2f GB/s)\n",
    bandwidth_picked, percentage_picked, peak_bandwidth_datasheet); 

#if !defined(PROFILE_NCUS)
    // Verify correctness
    printf("\nVerifying kernel correctness...\n");
    (*picked_kernel)(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaMemcpy(h_output_cuda, d_output, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

    // Compute CPU reference for subset
    const int verify_rows = 100;
    rms_norm_cpu(h_input, h_output_cpu, verify_rows, cols);

    bool passed = verify_rms_norm(h_output_cuda, h_output_cpu, verify_rows, cols);
    printf("✓ Verification: %s\n", passed ? "PASSED" : "FAILED");

    // Summary
    printf("\n");
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Target bandwidth: > 300 GB/s\n");
    printf("Peak memory bandwidth (datasheet): %.2f GB/s\n", peak_bandwidth_datasheet);
    printf("Achieved bandwidth (picked kernel): %.2f GB/s (%.1f%% of peak)\n",
        bandwidth_picked, percentage_picked);
    printf("Status: %s\n", bandwidth_picked > 300.0f ? "✓ PASSED" : "✗ NEEDS OPTIMIZATION");
    printf("================================================================================\n");
#endif

    // Cleanup
    free(h_input);
    free(h_output_cpu);
    free(h_output_cuda);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


int main() {
    printf("CSE 554 Assignment 1 - Section 2: RMS Norm Matrix Implementation\n");
    printf("================================================================================\n");

    // Get GPU properties
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int memClockRate;
    cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, device);

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Peak Memory Bandwidth: %.2f GB/s\n",
        2.0 * memClockRate * (prop.memoryBusWidth / 8) / 1e6);

    // Run tests
#if !defined(PROFILE_NCUS)
    test_correctness();
#endif
    benchmark_performance();

    printf("\n✓ All tests complete!\n");
    return 0;
}
