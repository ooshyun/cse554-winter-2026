/**
 * CSE 554 Assignment 1 - Section 1: SiLU CUDA Test
 * Main test file for CUDA SiLU implementation
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../../common/gpu_specs.h"

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Function declarations from silu.cu
extern void silu_cuda_basic(const float* d_input, float* d_output, int n);
extern void silu_cuda_optimized(const float* d_input, float* d_output, int n);
extern void silu_cuda_fast(const float* d_input, float* d_output, int n);
extern float measure_kernel_time(void (*)(const float*, float*, int),
                                const float*, float*, int, int);
extern float calculate_bandwidth(int n, float time_ms);


/**
 * CPU reference implementation of SiLU
 */
void silu_cpu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[i] = x * sigmoid;
    }
}


/**
 * Verify CUDA results against CPU reference
 */
bool verify_result(const float* cuda_result, const float* cpu_result, int n,
                float tolerance = 1e-5) {
    float max_diff = 0.0f;
    int max_diff_idx = 0;

    for (int i = 0; i < n; i++) {
        float diff = fabsf(cuda_result[i] - cpu_result[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    printf("  Max difference: %.2e at index %d\n", max_diff, max_diff_idx);
    printf("    CUDA: %.6f, CPU: %.6f\n",
        cuda_result[max_diff_idx], cpu_result[max_diff_idx]);

    return max_diff < tolerance;
}


/**
 * Test small input for correctness
 */
void test_correctness() {
    printf("\n");
    printf("================================================================================\n");
    printf("CORRECTNESS TEST\n");
    printf("================================================================================\n");

    const int n = 10;
    float h_input[n] = {-5.0f, -2.0f, -1.0f, -0.5f, 0.0f,
                        0.5f, 1.0f, 2.0f, 5.0f, 10.0f};
    float h_output_cpu[n];
    float h_output_cuda[n];

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                        cudaMemcpyHostToDevice));

    // Run CPU reference
    silu_cpu(h_input, h_output_cpu, n);

    // Run CUDA kernel
    silu_cuda_fast(d_input, d_output, n);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output_cuda, d_output, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

    // Display results
    printf("\nInput -> CPU Output | CUDA Output | Difference\n");
    printf("--------------------------------------------------------------\n");
    for (int i = 0; i < n; i++) {
        printf("%6.2f -> %10.6f | %10.6f | %.2e\n",
            h_input[i], h_output_cpu[i], h_output_cuda[i],
            fabsf(h_output_cpu[i] - h_output_cuda[i]));
    }

    // Verify
    bool passed = verify_result(h_output_cuda, h_output_cpu, n);
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

    const int rows = 8192;
    const int cols = 8192;
    const int n = rows * cols;
#if defined(PROFILE_NCUS)
    const int num_iterations = 1;
#else
    const int num_iterations = 100;
#endif

    printf("Matrix size: %d x %d = %d elements\n", rows, cols, n);
    printf("Element size: %zu bytes\n", sizeof(float));
    printf("Total memory: %.2f MB\n", (n * sizeof(float)) / (1024.0f * 1024.0f));
    printf("Iterations: %d\n\n", num_iterations);

    // Allocate host memory
    float* h_input = (float*)malloc(n * sizeof(float));
    float* h_output_cpu = (float*)malloc(n * sizeof(float));
    float* h_output_cuda = (float*)malloc(n * sizeof(float));

    // Initialize input with random values
    srand(42);
    for (int i = 0; i < n; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;  // Range: [-10, 10]
    }

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float),
                        cudaMemcpyHostToDevice));

    // Benchmark kernel
    // silu_cuda_basic silu_cuda_optimized silu_cuda_fast
    printf("Testing SiLU kernel...\n");
    void (*picked_kernel)(const float*, float*, int) = silu_cuda_basic;
    float time_picked = measure_kernel_time(picked_kernel, d_input, d_output,
                                        n, num_iterations);
    float bandwidth_picked = calculate_bandwidth(n, time_picked);
    const float peak_bandwidth_datasheet = GPU_PEAK_BANDWIDTH_DATASHEET;
    float percentage_picked = (bandwidth_picked / peak_bandwidth_datasheet) * 100.0f;

    printf("  Kernel execution time: %.4f ms\n", time_picked);
    printf("  Bandwidth: %.2f GB/s (%.1f%% of peak %.2f GB/s)\n",
        bandwidth_picked, percentage_picked, peak_bandwidth_datasheet);

#if !defined(PROFILE_NCUS)
    // Verify correctness of fast kernel
    printf("\nVerifying kernel correctness...\n");
    (*picked_kernel)(d_input, d_output, n);
    CUDA_CHECK(cudaMemcpy(h_output_cuda, d_output, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
    // Compute CPU reference for a subset
    const int verify_size = 10000;
    silu_cpu(h_input, h_output_cpu, verify_size);

    bool passed = verify_result(h_output_cuda, h_output_cpu, verify_size);
    printf("✓ Verification: %s\n", passed ? "PASSED" : "FAILED");

    // Summary
    printf("\n");
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Target bandwidth: > 500 GB/s\n");
    printf("Peak memory bandwidth (datasheet): %.2f GB/s\n", peak_bandwidth_datasheet);
    printf("Achieved bandwidth (picked kernel): %.2f GB/s (%.1f%% of peak)\n",
        bandwidth_picked, percentage_picked);
    printf("Status: %s\n", bandwidth_picked > 500.0f ? "✓ PASSED" : "✗ NEEDS OPTIMIZATION");
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
    printf("CSE 554 Assignment 1 - Section 1: SiLU CUDA Implementation\n");
    printf("================================================================================\n");

    // Initialize CUDA runtime and get GPU properties
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaSetDevice(device));  // Initialize CUDA runtime explicitly
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int memClockRate;
    cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, device);

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory Clock Rate: %.2f GHz\n", memClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.2f GB/s\n",
        2.0 * memClockRate * (prop.memoryBusWidth / 8) / 1e6);

    // Run tests
#if !defined(PROFILE_NCUS)
    test_correctness();
#endif
    benchmark_performance();

    printf("\n✓ All tests complete!\n");
    printf("  Run Nsight Compute: ncu -o profiling_results/cuda_silu ./silu_test\n");

    return 0;
}
