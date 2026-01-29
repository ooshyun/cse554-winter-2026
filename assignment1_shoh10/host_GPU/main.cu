/**
 * CSE 554 Assignment 1 - Section 3: Memory Transfer and First Column Copy Tests
 * Q1-Q2: Memory Transfer Bandwidth Tests
 * Q3: First Column Copy (via header)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../common/gpu_specs.h"
#include "copy_first_column.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// ============================================================================
// Q1-Q2: Memory Transfer Bandwidth Measurement Functions
// ============================================================================

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


void run_memory_transfer_tests() {
    printf("\n");
    printf("================================================================================\n");
    printf("Q1-Q2: Memory Transfer Bandwidth Tests\n");
    printf("================================================================================\n");

    // Get GPU properties
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Get memory clock via device attribute (CUDA 12.4+)
    int memClockRate;
    cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, device);

    printf("GPU: %s\n", prop.name);
    printf("Peak GPU Memory Bandwidth (calculated): %.2f GB/s\n",
        2.0 * (memClockRate / 1e6) * (prop.memoryBusWidth / 8));
    printf("Peak GPU Memory Bandwidth (datasheet): GPU_PEAK_BANDWIDTH_DATASHEET GB/s\n");
    printf("PCI Express Generation: %d\n", prop.pciDomainID);
    printf("\n");

    // Open CSV file for plotting
    FILE* csv_file = fopen("bandwidth_data.csv", "w");
    if (csv_file) {
        fprintf(csv_file, "size_bytes,memory_type,h2d_gbps,d2h_gbps\n");
    }

    // Test parameters
    const int num_iterations = 100;
    const int max_power = 28;  // Up to 256 MB
    bool use_pinned = false;

    printf("================================================================================\n");
    printf("Q1: Regular (Pageable) Memory\n");
    printf("================================================================================\n");

    // Test sizes from 2^0 to 2^28 (256 MB)
    for (int power = 0; power <= max_power; power++) {
        size_t size = 1 << power;
        // Reduce iterations for very large sizes to avoid timeout
        int iterations = (power > 24) ? 10 : num_iterations;
        measure_bandwidth(size, iterations, use_pinned, csv_file);
    }

    printf("\n");
    printf("================================================================================\n");
    printf("Q2: Pinned (Page-Locked) Memory\n");
    printf("================================================================================\n");

    use_pinned = true;
    for (int power = 0; power <= max_power; power++) {
        size_t size = 1 << power;
        int iterations = (power > 24) ? 10 : num_iterations;
        measure_bandwidth(size, iterations, use_pinned, csv_file);
    }

    if (csv_file) {
        fclose(csv_file);
        printf("\n✓ Data written to: bandwidth_data.csv\n");
        printf("  Use this data to plot bandwidth vs transfer size\n");
    }

    // Find peak bandwidth with larger size
    printf("\n");
    printf("================================================================================\n");
    printf("PEAK BANDWIDTH MEASUREMENT\n");
    printf("================================================================================\n");

    size_t large_size = 256 << 20;  // 256 MB - large enough to saturate bandwidth
    float *h_pinned, *d_data;
    CUDA_CHECK(cudaMallocHost(&h_pinned, large_size));
    CUDA_CHECK(cudaMalloc(&d_data, large_size));

    // Initialize memory
    for (size_t i = 0; i < large_size / sizeof(float); i++) {
        h_pinned[i] = 1.0f;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 5; i++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_pinned, large_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_pinned, d_data, large_size, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure peak H2D
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        CUDA_CHECK(cudaMemcpy(d_data, h_pinned, large_size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    // Convert: (bytes * iterations) / (time_ms / 1000.0) / (1024^3) = GB/s
    float peak_h2d = static_cast<float>((static_cast<double>(large_size) * 100.0 / static_cast<double>(time_ms) * 1000.0) / (1024.0 * 1024.0 * 1024.0));

    // Measure peak D2H
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        CUDA_CHECK(cudaMemcpy(h_pinned, d_data, large_size, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    float peak_d2h = static_cast<float>((static_cast<double>(large_size) * 100.0 / static_cast<double>(time_ms) * 1000.0) / (1024.0 * 1024.0 * 1024.0));

    printf("Transfer the largest size: 256 MB\n");
    printf("Peak Host-to-Device (pinned): %.2f GB/s\n", peak_h2d);
    printf("Peak Device-to-Host (pinned): %.2f GB/s\n", peak_d2h);
    printf("Average bidirectional: %.2f GB/s\n", (peak_h2d + peak_d2h) / 2.0f);
    printf("================================================================================\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_data));

    printf("\n✓ Memory transfer tests complete!\n");
}


// ============================================================================
// Q3: First Column Copy Tests (using functions from copy_first_column.cu)
// ============================================================================

void run_first_column_copy_tests() {
    printf("\n");
    printf("================================================================================\n");
    printf("Q3: First Column Copy Tests\n");
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
    printf("Target time: ~150 μs (end-to-end including buffer allocation/deallocation)\n");
    printf("Iterations: %d\n\n", num_iterations);

    // =========================================================================
    // GIVEN (per TA comment):
    //   1. Input matrix on CPU in PAGEABLE (unpinned) memory
    //   2. Destination buffer on GPU
    //
    // NOT GIVEN (must be included in timing if used):
    //   - Any intermediate buffers (pinned or pageable)
    //   - Any allocation/deallocation operations
    // =========================================================================

    // Allocate and initialize matrix on host (PAGEABLE memory - this is GIVEN)
    size_t matrix_size = (size_t)rows * (size_t)cols * sizeof(float);
    float* h_matrix = (float*)malloc(matrix_size);

    printf("Initializing matrix (%.2f MB)...\n", static_cast<float>(matrix_size) / (1024.0f * 1024.0f));
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            h_matrix[row * cols + col] = (float)col;  // First column = 0
        }
    }

    // Allocate device memory for column (this is GIVEN per problem statement)
    float* d_column;
    CUDA_CHECK(cudaMalloc(&d_column, rows * sizeof(float)));

    // Initialize CUDA resources (events, streams) - can be pre-created
    copy_first_column_init();

    // Warmup: run a few iterations to warm up caches and CUDA runtime
    printf("Warming up...\n");
    for (int i = 0; i < 5; i++) {
        copy_first_column_optimized(h_matrix, d_column, rows, cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Test: Optimized copy (end-to-end including any buffer management)
    printf("\n");
    printf("================================================================================\n");
    printf("Optimized First Column Copy (End-to-End Measurement)\n");
    printf("================================================================================\n");

    float total_time = 0.0f;
    for (int i = 0; i < num_iterations; i++) {
        total_time += copy_first_column_optimized(h_matrix, d_column, rows, cols);
    }
    float avg_time = total_time / num_iterations;
    printf("Average time: %.2f μs\n", avg_time * 1000.0f);
    printf("Status: %s\n", avg_time * 1000.0f < 150.0f ? "✓ PASSED" : "✗ FAILED");

    // Cleanup CUDA resources
    copy_first_column_cleanup();

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
        printf("✓ All %d values correct!\n", rows);
    }

    // Summary
    printf("\n");
    printf("================================================================================\n");
    printf("SUMMARY\n");
    printf("================================================================================\n");
    printf("Target: ~150 μs (end-to-end including buffer allocation/deallocation)\n");
    printf("\n");
    printf("Optimized: %.2f μs %s\n",
           avg_time * 1000.0f,
           avg_time * 1000.0f < 150.0f ? "✓" : "✗");
    printf("\n");
    printf("Overall status: %s\n", avg_time * 1000.0f < 150.0f ? "✓ PASSED" : "✗ FAILED");
    printf("================================================================================\n");

    // Cleanup
    free(h_result);
#endif

    free(h_matrix);
    CUDA_CHECK(cudaFree(d_column));

    printf("\n✓ First column copy tests complete!\n");
}


// ============================================================================
// Main Entry Point
// ============================================================================

int main() {
    printf("CSE 554 Assignment 1 - Section 3: Host-GPU Memory Operations\n");
    printf("================================================================================\n");

    // Run Q1-Q2: Memory Transfer Bandwidth Tests
    run_memory_transfer_tests();

    // Run Q3: First Column Copy Tests
    run_first_column_copy_tests();

    printf("\n================================================================================\n");
    printf("✓ All Section 3 tests complete!\n");
    printf("================================================================================\n");

    return 0;
}
