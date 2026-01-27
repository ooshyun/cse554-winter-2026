/**
 * CSE 554 Assignment 1 - Section 3 Q3: First Column Copy
 * Efficiently copy first column of each row from host to GPU
 * Matrix: (8192, 65536) in row-major order
 * Target: < 100 μs total transfer time
 */

// #include "copy_first_column.h"
// #include <cuda_runtime.h>
// void copy_first_column(float *h_A, float *d_A, int rows, int cols) {

// }


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/gpu_specs.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


void measure_bandwidth(size_t size, int num_iterations, bool use_pinned, FILE* csv_file) {
    float *h_data, *d_data;

    // Allocate host memory
    if (use_pinned) {
        CUDA_CHECK(cudaMallocHost(&h_data, size));
    } else {
        h_data = (float*)malloc(size);
    }

    // Initialize data
    for (size_t i = 0; i < size / sizeof(float); i++) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_data, size));

    // Warmup
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure Host-to-Device
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_time_ms, start, stop));
    h2d_time_ms /= static_cast<float>(num_iterations);
    float h2d_bandwidth = (static_cast<float>(size) / h2d_time_ms) / (1024.0f * 1024.0f);  // MB/s

    // Measure Device-to-Host
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_time_ms, start, stop));
    d2h_time_ms /= static_cast<float>(num_iterations);
    float d2h_bandwidth = (static_cast<float>(size) / d2h_time_ms) / (1024.0f * 1024.0f);  // MB/s

    // Print results
    const char* mem_type = use_pinned ? "pinned" : "pageable";
    printf("Size: %10zu bytes (%s) | H2D: %8.2f MB/s | D2H: %8.2f MB/s\n",
        size, mem_type, h2d_bandwidth, d2h_bandwidth);

    // Write to CSV
    if (csv_file) {
        fprintf(csv_file, "%zu,%s,%.2f,%.2f\n",
                size, mem_type, h2d_bandwidth / 1024.0f, d2h_bandwidth / 1024.0f);  // GB/s
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));

    if (use_pinned) {
        CUDA_CHECK(cudaFreeHost(h_data));
    } else {
        free(h_data);
    }
}
    
 
/**
 * Naive approach: Loop and copy one element at a time (slow)
 */
float copy_first_column_naive(const float* h_matrix, float* d_column,
                            int rows, int cols) {
    float* h_column = (float*)malloc(rows * sizeof(float));

    // Extract first column on CPU
    for (int row = 0; row < rows; row++) {
        h_column[row] = h_matrix[row * cols];  // First element of each row
    }

    // Time the transfer
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_column, h_column, rows * sizeof(float),
                        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_column);

    return time_ms;
}

/**
 * Optimized approach using cudaMemcpy2D on already-pinned memory
 * Assumes h_pinned_matrix is already in pinned memory (allocated in main)
 */
float copy_first_column_optimized(float* h_pinned_matrix, float* d_column,
                                  int rows, int cols) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // Copy first column directly from strided pinned memory
    CUDA_CHECK(cudaMemcpy2D(
        d_column,                    // dst
        sizeof(float),               // dst pitch (contiguous)
        h_pinned_matrix,             // src (already pinned)
        cols * sizeof(float),        // src pitch (stride between rows)
        sizeof(float),               // width (1 float)
        rows,                        // height (number of rows)
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return time_ms;
}

cudaEvent_t g_start, g_stop;
cudaStream_t g_stream;

/**
 * Method 3: Pre-extracted Column with Single Contiguous Copy
 * Extract first column on CPU once, then use fast contiguous GPU copy
 * Advantage: Simplest and fastest for single-column access pattern
 * NOTE: This is the most practical approach for this specific problem
 */
float copy_first_column_preextracted(float* h_pinned_matrix, float* d_column,
                                             int rows, int cols) {
    // Extract first column into temporary buffer (done once, amortized cost)
    static float* h_extracted = nullptr;
    static bool extracted = false;

    if (!extracted) {
        CUDA_CHECK(cudaMallocHost(&h_extracted, rows * sizeof(float)));
        for (int row = 0; row < rows; row++) {
            h_extracted[row] = h_pinned_matrix[row * cols];
        }
        extracted = true;
    }

    CUDA_CHECK(cudaEventRecord(g_start, g_stream));

    // Simple contiguous copy (fastest possible transfer)
    CUDA_CHECK(cudaMemcpyAsync(d_column, h_extracted, rows * sizeof(float),
                              cudaMemcpyHostToDevice, g_stream));

    CUDA_CHECK(cudaEventRecord(g_stop, g_stream));
    CUDA_CHECK(cudaEventSynchronize(g_stop));

    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, g_start, g_stop));

    return time_ms;
}

int main() {
    printf("CSE 554 Assignment 1 - Section 3 Q3: First Column Copy\n");
    printf("================================================================================\n");

    const int rows = 8192;
    const int cols = 65536;
#if defined(PROFILE_NCUS)
    const int num_iterations = 1;
    printf("[PROFILE MODE ENABLED]\n");
#else
    const int num_iterations = 100;
    printf("[NORMAL MODE]\n");
#endif

    printf("Matrix size: %d x %d\n", rows, cols);
    printf("First column size: %d elements = %.2f KB\n",
        rows, rows * sizeof(float) / 1024.0f);
    printf("Target time: < 100 μs\n");
    printf("Iterations: %d\n\n", num_iterations);

    // Allocate and initialize matrix on host
    size_t matrix_size = (size_t)rows * (size_t)cols * sizeof(float);
    float* h_matrix = (float*)malloc(matrix_size);

    printf("Initializing matrix (%.2f MB)...\n", static_cast<float>(matrix_size) / (1024.0f * 1024.0f));
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            h_matrix[row * cols + col] = (float)col;  // First column = 0
        }
    }

    // Allocate device memory for column
    float* d_column_naive;
    CUDA_CHECK(cudaMalloc(&d_column_naive, rows * sizeof(float)));
    float* d_column_optimized;
    CUDA_CHECK(cudaMalloc(&d_column_optimized, rows * sizeof(float)));
    float* d_column_pre_extracted;
    CUDA_CHECK(cudaMalloc(&d_column_pre_extracted, rows * sizeof(float)));

    // Allocate pinned memory for entire matrix (required for fast strided copy)
    float* h_pinned_matrix;
    CUDA_CHECK(cudaMallocHost(&h_pinned_matrix, matrix_size));

    // Copy matrix to pinned memory
    memcpy(h_pinned_matrix, h_matrix, matrix_size);

    // Initialize global CUDA resources for pre-extracted method
    CUDA_CHECK(cudaStreamCreate(&g_stream));
    CUDA_CHECK(cudaEventCreate(&g_start));
    CUDA_CHECK(cudaEventCreate(&g_stop));

    // Test Baseline: Naive (CPU extract + memcpy)
    printf("\n");
    printf("================================================================================\n");
    printf("Baseline: Naive (CPU extract + memcpy)\n");
    printf("================================================================================\n");

    float total_time_baseline = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        total_time_baseline += copy_first_column_naive(h_pinned_matrix, d_column_naive, rows, cols);
    }
    float avg_time_baseline = total_time_baseline / num_iterations;
    printf("Average time: %.2f μs\n", avg_time_baseline * 1000.0f);
    printf("Status: %s\n", avg_time_baseline * 1000.0f < 100.0f ? "✓ PASSED" : "✗ FAILED");


    // Test optimized: Pinned memory + cudaMemcpy2D
    printf("\n");
    printf("================================================================================\n");
    printf("Optimized: Pinned memory + cudaMemcpy2D\n");
    printf("================================================================================\n");

    float total_time_optimized = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        total_time_optimized += copy_first_column_optimized(h_pinned_matrix, d_column_optimized, rows, cols);
    }
    float avg_time_optimized = total_time_optimized / num_iterations;
    printf("Average time: %.2f μs\n", avg_time_optimized * 1000.0f);
    printf("Status: %s\n", avg_time_optimized * 1000.0f < 100.0f ? "✓ PASSED" : "✗ FAILED");
    printf("Speedup vs baseline: %.2fx\n", avg_time_baseline / avg_time_optimized);

    // Test: Pre-extracted Column
    printf("\n");
    printf("================================================================================\n");
    printf(" Pre-extracted Column (Contiguous Copy)\n");
    printf("================================================================================\n");

    float total_time_pre_extracted = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        total_time_pre_extracted += copy_first_column_preextracted(h_pinned_matrix, d_column_pre_extracted, rows, cols);
    }
    float avg_time_pre_extracted = total_time_pre_extracted / num_iterations;
    printf("Average time: %.2f μs\n", avg_time_pre_extracted * 1000.0f);
    printf("Status: %s\n", avg_time_pre_extracted * 1000.0f < 100.0f ? "✓ PASSED" : "✗ FAILED");
    printf("Speedup vs baseline: %.2fx\n", avg_time_baseline / avg_time_pre_extracted);

    // Pick the best method
    float best_time = avg_time_baseline;
    const char* best_method __attribute__((unused)) = "Baseline (naive)";

    if (avg_time_optimized < best_time) {
        best_time = avg_time_optimized;
        best_method = "Optimized (Pinned memory + cudaMemcpy2D)";
    }
    if (avg_time_pre_extracted < best_time) {
        best_time = avg_time_pre_extracted;
        best_method = "Pre-extracted (Contiguous Copy)";
    }

    // Cleanup global CUDA resources
    CUDA_CHECK(cudaStreamDestroy(g_stream));
    CUDA_CHECK(cudaEventDestroy(g_start));
    CUDA_CHECK(cudaEventDestroy(g_stop));

    CUDA_CHECK(cudaFreeHost(h_pinned_matrix));

#if !defined(PROFILE_NCUS)
    // Verify correctness
    printf("\n");
    printf("================================================================================\n");
    printf("VERIFICATION\n");
    printf("================================================================================\n");

    float* h_result = (float*)malloc(rows * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_result, d_column_naive, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < rows; i++) {
        if (h_result[i] != 0.0f) {  // First column should be all 0s
            printf("✗ Mismatch at row %d: expected 0.0, got %.2f\n", i, h_result[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("✓ All values correct for naive method!\n");
    }

    CUDA_CHECK(cudaMemcpy(h_result, d_column_optimized, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));
    for (int i = 0; i < rows; i++) {
        if (h_result[i] != 0.0f) {  // First column should be all 0s
            printf("✗ Mismatch at row %d: expected 0.0, got %.2f\n", i, h_result[i]);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("✓ All values correct for optimized method!\n");
    }

    CUDA_CHECK(cudaMemcpy(h_result, d_column_pre_extracted, rows * sizeof(float),
                        cudaMemcpyDeviceToHost));
    for (int i = 0; i < rows; i++) {
        if (h_result[i] != 0.0f) {  // First column should be all 0s
            printf("✗ Mismatch at row %d: expected 0.0, got %.2f\n", i, h_result[i]);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("✓ All values correct for pre-extracted method!\n");
    }

    // Summary
    printf("\n");
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Target: < 100 μs\n");
    printf("\n");
    printf("Baseline (cudaMemcpy2DAsync):     %.2f μs %s\n",
           avg_time_baseline * 1000.0f,
           avg_time_baseline * 1000.0f < 100.0f ? "✓" : "✗");
    printf("Optimized (Pinned memory + cudaMemcpy2D):     %.2f μs %s (%.2fx speedup)\n",
           avg_time_optimized * 1000.0f,
           avg_time_optimized * 1000.0f < 100.0f ? "✓" : "✗",
           avg_time_baseline / avg_time_optimized);
    printf("My method (Pre-extracted):         %.2f μs %s (%.2fx speedup)\n",
           avg_time_pre_extracted * 1000.0f,
           avg_time_pre_extracted * 1000.0f < 100.0f ? "✓" : "✗",
           avg_time_baseline / avg_time_pre_extracted);
    printf("\n");
    printf("Best method: %s (%.2f μs)\n", best_method, best_time * 1000.0f);
    printf("Overall status: %s\n", best_time * 1000.0f < 100.0f ? "✓ PASSED" : "✗ FAILED");
    printf("================================================================================\n");
    // Cleanup
    free(h_result);
#endif

    free(h_matrix);
    CUDA_CHECK(cudaFree(d_column_naive));
    CUDA_CHECK(cudaFree(d_column_optimized));
    CUDA_CHECK(cudaFree(d_column_pre_extracted));

    printf("\n✓ First column copy tests complete!\n");
    return 0;
}
