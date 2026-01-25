/**
 * CSE 554 Assignment 1 - Section 2: RMS Norm Vector Test
 * Main test file for vector RMS Norm implementation (1, 1024×1024)
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
extern void rms_norm_vector_basic(const float*, float*, int);
extern void rms_norm_vector_fast(const float*, float*, int);
extern void rms_norm_vector_w2l3_reduction(const float*, float*, int);
extern void rms_norm_vector_w2l3_hybrid(const float*, float*, int);
extern void rms_norm_vector_basic_v1(const float*, float*, int);
extern float measure_rms_norm_vector_time(void (*)(const float*, float*, int),
                                          const float*, float*, int, int);
extern float calculate_rms_norm_vector_bandwidth(int, float);

// Picked kernel: rms_norm_vector_basic_v1
void (*picked_kernel)(const float*, float*, int) = rms_norm_vector_basic_v1;


/**
* CPU reference implementation of RMS Norm for vector
*/
void rms_norm_cpu_vector(const float* input, float* output, int n) {
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float val = input[i];
        sum_sq += val * val;
    }

    // Compute RMS
    float rms = sqrtf(sum_sq / n + EPSILON);

    // Normalize
    for (int i = 0; i < n; i++) {
        output[i] = input[i] / rms;
    }
}


/**
* Verify CUDA results against CPU reference
*/
bool verify_rms_norm_vector(const float* cuda_result, const float* cpu_result,
                            int n, float tolerance = 1e-3) {
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
* Test small vector for correctness
*/
void test_correctness() {
    printf("\n");
    printf("================================================================================\n");
    printf("CORRECTNESS TEST\n");
    printf("================================================================================\n");

    const int n = 100;
    float h_input[n];
    float h_output_cpu[n];
    float h_output_cuda[n];

    // Initialize test data
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)(i % 20) - 10.0f;  // Values from -10 to 9
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
    rms_norm_cpu_vector(h_input, h_output_cpu, n);

    // Run CUDA kernel
    printf("Running CUDA kernel...\n");
    rms_norm_vector_fast(d_input, d_output, n);
    CUDA_CHECK(cudaMemcpy(h_output_cuda, d_output, n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Display first 10 results
    printf("\nFirst 10 elements:\n");
    printf("Index | Input   | CPU     | CUDA    | Diff\n");
    printf("-----------------------------------------------\n");
    for (int i = 0; i < 10; i++) {
        printf("%5d | %7.2f | %7.4f | %7.4f | %.2e\n",
              i, h_input[i], h_output_cpu[i], h_output_cuda[i],
              fabsf(h_output_cpu[i] - h_output_cuda[i]));
    }

    // Verify
    bool passed = verify_rms_norm_vector(h_output_cuda, h_output_cpu, n);
    printf("\n✓ Correctness test: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


/**
* Benchmark performance for vector (1, 1024×1024)
*/
void benchmark_performance() {
    printf("\n");
    printf("================================================================================\n");
    printf("PERFORMANCE BENCHMARK (1 x 1048576 vector)\n");
    printf("================================================================================\n");

    const int n = 1024 * 1024;  // 1048576 elements
#if defined(PROFILE_NCUS)
    const int num_iterations = 1;
    printf("[PROFILE MODE ENABLED]\n");
#else
    const int num_iterations = 100;
    printf("[NORMAL MODE]\n");
#endif

    printf("Vector size: 1 x %d = %d elements\n", n, n);
    printf("Memory per vector: %.2f MB\n", (n * sizeof(float)) / (1024.0f * 1024.0f));
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
    // rms_norm_vector_basic rms_norm_vector_optimized rms_norm_vector_fast
    // rms_norm_vector_w2l3_reduction rms_norm_vector_w2l3_tile rms_norm_vector_w2l3_hybrid
    printf("Testing RMS Norm Vector kernel...\n");
    float time_picked = measure_rms_norm_vector_time(
      picked_kernel, d_input, d_output, n, num_iterations);
    float bandwidth_picked = calculate_rms_norm_vector_bandwidth(n, time_picked);
    const float peak_bandwidth_datasheet = GPU_PEAK_BANDWIDTH_DATASHEET;
    float percentage_picked = (bandwidth_picked / peak_bandwidth_datasheet) * 100.0f;
    printf("  Execution time: %.4f ms\n", time_picked);
    printf("  Bandwidth: %.2f GB/s (%.1f%% of peak %.2f GB/s)\n",
      bandwidth_picked, percentage_picked, peak_bandwidth_datasheet); 

#if !defined(PROFILE_NCUS)
    // Verify correctness
    printf("\nVerifying kernel correctness...\n");
    (*picked_kernel)(d_input, d_output, n);
    CUDA_CHECK(cudaMemcpy(h_output_cuda, d_output, n * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Compute CPU reference for subset
    // For vector RMS Norm, the entire vector should have same RMS
    // So we need to compute RMS for full vector on CPU
    const int verify_size = 10000;
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += h_input[i] * h_input[i];
    }
    float rms = sqrtf(sum_sq / n + EPSILON);

    // Normalize subset with full vector RMS
    for (int i = 0; i < verify_size; i++) {
        h_output_cpu[i] = h_input[i] / rms;
    }

    bool passed = verify_rms_norm_vector(h_output_cuda, h_output_cpu, verify_size);
    printf("✓ Verification: %s\n", passed ? "PASSED" : "FAILED");

    // Summary
    printf("\n");
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Target bandwidth: > 200 GB/s\n");
    printf("Peak memory bandwidth (datasheet): %.2f GB/s\n", peak_bandwidth_datasheet);
    printf("Achieved bandwidth (picked kernel): %.2f GB/s (%.1f%% of peak)\n",
          bandwidth_picked, percentage_picked);
    printf("Status: %s\n", bandwidth_picked > 200.0f ? "✓ PASSED" : "✗ NEEDS OPTIMIZATION");
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
    printf("CSE 554 Assignment 1 - Section 2: RMS Norm Vector Implementation\n");
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
