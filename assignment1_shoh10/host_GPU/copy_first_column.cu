/**
 * CSE 554 Assignment 1 - Section 3 Q3: First Column Copy
 * Efficiently copy first column of each row from host to GPU
 * Matrix: (8192, 65536) in row-major order
 * Target: ~150 μs total transfer time (end-to-end)
 *
 * GIVEN (per TA comment):
 *   - Input matrix on CPU in PAGEABLE (unpinned) memory
 *   - Destination buffer on GPU
 *
 * TIMING MUST INCLUDE:
 *   - Any intermediate buffer allocation
 *   - Data extraction/copying
 *   - GPU transfer
 *   - Any intermediate buffer deallocation
 */

#include "copy_first_column.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// Global CUDA resources (these can be pre-created, not counted in timing)
static cudaEvent_t g_start = nullptr;
static cudaEvent_t g_stop = nullptr;
static cudaStream_t g_stream = nullptr;
static bool g_initialized = false;


/**
 * Initialize CUDA resources for copy_first_column functions
 * These resources can be pre-created as they are reusable infrastructure
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
}


/**
 * Optimized first column copy with end-to-end timing
 *
 * Strategy: Use cudaMemcpy2D for direct strided copy from host to device.
 * This is much faster than CPU extraction because:
 * - CUDA runtime handles strided access efficiently
 * - No intermediate buffer allocation needed
 * - Direct DMA with optimized memory access patterns
 *
 * cudaMemcpy2D parameters:
 * - dst: destination pointer (device column buffer)
 * - dpitch: destination pitch (sizeof(float) for contiguous column)
 * - src: source pointer (host matrix)
 * - spitch: source pitch (row stride in bytes = cols * sizeof(float))
 * - width: bytes to copy per row (sizeof(float) = 4 bytes)
 * - height: number of rows to copy
 *
 * @param h_matrix Host matrix in PAGEABLE memory (row-major)
 * @param d_column Device memory to store first column (GIVEN)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Transfer time in milliseconds
 */
float copy_first_column_optimized(const float* h_matrix, float* d_column,
                                  int rows, int cols) {
    // Use high-resolution CPU timer for end-to-end measurement
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Use cudaMemcpy2D for efficient strided copy
    // This copies the first column directly without intermediate buffers
    CUDA_CHECK(cudaMemcpy2D(
        d_column,                    // dst: device column buffer
        sizeof(float),               // dpitch: destination stride (contiguous)
        h_matrix,                    // src: host matrix (first column starts at offset 0)
        (size_t)cols * sizeof(float), // spitch: source stride (row size in bytes)
        sizeof(float),               // width: bytes per element (one float)
        (size_t)rows,                // height: number of rows
        cudaMemcpyHostToDevice
    ));

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // Calculate elapsed time in milliseconds
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    return (float)elapsed_ms;
}


/**
 * Alternative method: CPU extract + pinned memory copy
 *
 * Strategy: Extract first column on CPU into a pinned buffer,
 * then do a contiguous memcpy to GPU.
 *
 * Timing includes (end-to-end):
 * - cudaMallocHost (pinned buffer allocation)
 * - CPU extraction (strided read from pageable source)
 * - cudaMemcpy to GPU
 * - cudaFreeHost (pinned buffer deallocation)
 *
 * Performance: ~480 μs on test system (slower than cudaMemcpy2D due to
 * CPU strided read bottleneck - stride=256KB causes cache misses)
 *
 * @param h_matrix Host matrix in PAGEABLE memory (row-major)
 * @param d_column Device memory to store first column (GIVEN)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Transfer time in milliseconds
 */
float copy_first_column_cpu_extract(const float* h_matrix, float* d_column,
                                    int rows, int cols) {
    // Use high-resolution CPU timer for end-to-end measurement
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Allocate PINNED temporary buffer (INCLUDED in timing)
    float* h_column;
    CUDA_CHECK(cudaMallocHost(&h_column, rows * sizeof(float)));

    // Extract first column on CPU (strided read from pageable source)
    for (int row = 0; row < rows; row++) {
        h_column[row] = h_matrix[row * cols];  // First element of each row
    }

    // Transfer contiguous data to GPU (fast DMA from pinned memory)
    CUDA_CHECK(cudaMemcpy(d_column, h_column, rows * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Free pinned buffer (INCLUDED in timing)
    CUDA_CHECK(cudaFreeHost(h_column));

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // Calculate elapsed time in milliseconds
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    return (float)elapsed_ms;
}


/**
 * Method 3: CPU extract + pageable memory copy (baseline/naive)
 *
 * Strategy: Extract first column on CPU into a pageable buffer,
 * then do cudaMemcpy to GPU.
 *
 * Timing includes (end-to-end):
 * - malloc (pageable buffer allocation)
 * - CPU extraction (strided read from pageable source)
 * - cudaMemcpy to GPU (pageable → device, internally staged)
 * - free (pageable buffer deallocation)
 *
 * Expected to be slowest due to:
 * - CPU strided read bottleneck
 * - Pageable memory requires internal staging by CUDA driver
 *
 * @param h_matrix Host matrix in PAGEABLE memory (row-major)
 * @param d_column Device memory to store first column (GIVEN)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Transfer time in milliseconds
 */
float copy_first_column_naive(const float* h_matrix, float* d_column,
                              int rows, int cols) {
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Allocate PAGEABLE temporary buffer (INCLUDED in timing)
    float* h_column = (float*)malloc(rows * sizeof(float));

    // Extract first column on CPU (strided read)
    for (int row = 0; row < rows; row++) {
        h_column[row] = h_matrix[row * cols];
    }

    // Transfer to GPU (pageable memory - internally staged by driver)
    CUDA_CHECK(cudaMemcpy(d_column, h_column, rows * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Free pageable buffer (INCLUDED in timing)
    free(h_column);

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    return (float)elapsed_ms;
}
