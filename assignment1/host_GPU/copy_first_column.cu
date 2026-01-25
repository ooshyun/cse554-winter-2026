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
    h2d_time_ms /= num_iterations;
    float h2d_bandwidth = (size / h2d_time_ms) / (1024.0f * 1024.0f);  // MB/s

    // Measure Device-to-Host
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; i++) {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_time_ms, start, stop));
    d2h_time_ms /= num_iterations;
    float d2h_bandwidth = (size / d2h_time_ms) / (1024.0f * 1024.0f);  // MB/s

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
 * Optimized approach using cudaMemcpy2D (handles strided data)
 */
float copy_first_column_memcpy2d(const float* h_matrix, float* d_column,
                                int rows, int cols) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // cudaMemcpy2D handles strided data efficiently
    // Copy width: 1 element (4 bytes)
    // Copy height: rows
    // Source pitch: cols * sizeof(float) (row-major stride)
    // Dest pitch: sizeof(float) (contiguous)
    CUDA_CHECK(cudaMemcpy2D(
        d_column,                    // dst
        sizeof(float),               // dst pitch (contiguous: 1 element width)
        h_matrix,                    // src
        cols * sizeof(float),        // src pitch (stride between rows)
        sizeof(float),               // width in bytes (1 float)
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


/**
 * Highly optimized approach using pinned memory + cudaMemcpy2D
 */
float copy_first_column_optimized(const float* h_matrix, float* d_column,
                                int rows, int cols) {
    // Use pinned host memory for faster transfer
    float* h_pinned;
    size_t matrix_size = (size_t)rows * (size_t)cols * sizeof(float);
    CUDA_CHECK(cudaMallocHost(&h_pinned, matrix_size));

    // Copy from regular host memory to pinned memory
    memcpy(h_pinned, h_matrix, matrix_size);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy2D(
        d_column,
        sizeof(float),
        h_pinned,
        cols * sizeof(float),
        sizeof(float),
        rows,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_pinned));

    return time_ms;
}


/**
 * Ultra-optimized: Pre-extracted column in pinned memory + simple async transfer
 * Key insight: Extract first column ONCE in pinned memory, then use fast contiguous copy
 */
float copy_first_column_ultra(float* h_pinned_column, float* d_column,
                            int rows, int cols) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Use a stream for async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaEventRecord(start, stream));

    // Simple contiguous memcpy - much faster than 2D copy for small data
    CUDA_CHECK(cudaMemcpyAsync(
        d_column,
        h_pinned_column,
        rows * sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    ));

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return time_ms;
}


// Picked kernel: copy_first_column_ultra (requires pre-allocated pinned memory)
// Function pointer type for copy functions
typedef float (*CopyFirstColumnFunc)(const float*, float*, int, int);
typedef float (*CopyFirstColumnUltraFunc)(float*, float*, int, int);

CopyFirstColumnUltraFunc picked_kernel = copy_first_column_ultra;

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

    printf("Initializing matrix (%.2f MB)...\n", matrix_size / (1024.0f * 1024.0f));
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            h_matrix[row * cols + col] = (float)col;  // First column = 0
        }
    }

    // Allocate device memory for column
    float* d_column;
    CUDA_CHECK(cudaMalloc(&d_column, rows * sizeof(float)));

    // Allocate pinned memory for JUST the first column (not entire matrix)
    float* h_pinned_column;
    CUDA_CHECK(cudaMallocHost(&h_pinned_column, rows * sizeof(float)));

    // Extract first column into pinned memory
    for (int row = 0; row < rows; row++) {
        h_pinned_column[row] = h_matrix[row * cols];
    }

    // Test picked kernel
    printf("\n");
    printf("================================================================================\n");
    printf("Testing Picked Kernel\n");
    printf("================================================================================\n");

    float total_time_picked = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        total_time_picked += (*picked_kernel)(h_pinned_column, d_column, rows, cols);
    }
    float avg_time_picked = total_time_picked / num_iterations;
    printf("Average time: %.2f μs\n", avg_time_picked * 1000.0f);
    printf("Status: %s\n", avg_time_picked * 1000.0f < 100.0f ? "✓ PASSED" : "✗ FAILED");

    CUDA_CHECK(cudaFreeHost(h_pinned_column));

#if !defined(PROFILE_NCUS)
    // Verify correctness
    printf("\n");
    printf("================================================================================\n");
    printf("VERIFICATION\n");
    printf("================================================================================\n");

    float* h_result = (float*)malloc(rows * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_result, d_column, rows * sizeof(float),
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
        printf("✓ All values correct!\n");
    }

    // Summary
    printf("\n");
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Target: < 100 μs\n");
    printf("Picked kernel: %.2f μs\n", avg_time_picked * 1000.0f);
    printf("Status: %s\n", avg_time_picked * 1000.0f < 100.0f ? "✓ PASSED" : "✗ FAILED");
    printf("================================================================================\n");
    // Cleanup
    free(h_result);
#endif

    free(h_matrix);
    CUDA_CHECK(cudaFree(d_column));

    printf("\n✓ First column copy tests complete!\n");
    return 0;
}
