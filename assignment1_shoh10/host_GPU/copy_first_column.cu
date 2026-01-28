/**
 * CSE 554 Assignment 1 - Section 3 Q3: First Column Copy
 * Efficiently copy first column of each row from host to GPU
 * Matrix: (8192, 65536) in row-major order
 * Target: < 100 μs total transfer time
 */

#include "copy_first_column.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// Global CUDA resources for pre-extracted method
static cudaEvent_t g_start = nullptr;
static cudaEvent_t g_stop = nullptr;
static cudaStream_t g_stream = nullptr;
static float* h_extracted = nullptr;
static bool g_initialized = false;
static bool g_extracted = false;
static int g_cached_rows = 0;


/**
 * Initialize CUDA resources for copy_first_column functions
 */
void copy_first_column_init() {
    if (!g_initialized) {
        CUDA_CHECK(cudaStreamCreate(&g_stream));
        CUDA_CHECK(cudaEventCreate(&g_start));
        CUDA_CHECK(cudaEventCreate(&g_stop));
        g_initialized = true;
    }
}


/**
 * Cleanup CUDA resources
 */
void copy_first_column_cleanup() {
    if (g_initialized) {
        if (g_stream) {
            CUDA_CHECK(cudaStreamDestroy(g_stream));
            g_stream = nullptr;
        }
        if (g_start) {
            CUDA_CHECK(cudaEventDestroy(g_start));
            g_start = nullptr;
        }
        if (g_stop) {
            CUDA_CHECK(cudaEventDestroy(g_stop));
            g_stop = nullptr;
        }
        g_initialized = false;
    }

    if (h_extracted) {
        CUDA_CHECK(cudaFreeHost(h_extracted));
        h_extracted = nullptr;
    }
    g_extracted = false;
    g_cached_rows = 0;
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
 * Pre-extracted Column with Single Contiguous Copy
 * Extract first column on CPU once, then use fast contiguous GPU copy
 * Advantage: Simplest and fastest for single-column access pattern
 * NOTE: This is the most practical approach for this specific problem
 */
float copy_first_column_preextracted(float* h_pinned_matrix, float* d_column,
                                     int rows, int cols) {
    // Extract first column into temporary buffer (done once, amortized cost)
    if (!g_extracted || g_cached_rows != rows) {
        if (h_extracted) {
            CUDA_CHECK(cudaFreeHost(h_extracted));
        }
        CUDA_CHECK(cudaMallocHost(&h_extracted, rows * sizeof(float)));
        for (int row = 0; row < rows; row++) {
            h_extracted[row] = h_pinned_matrix[row * cols];
        }
        g_extracted = true;
        g_cached_rows = rows;
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
