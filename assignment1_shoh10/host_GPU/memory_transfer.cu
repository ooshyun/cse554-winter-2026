/**
 * CSE 554 Assignment 1 - Section 3 Q1-Q2: Memory Transfer Bandwidth Tests
 * Measure Host-to-GPU and GPU-to-Host bandwidth for various transfer sizes
 * Compare regular (pageable) vs pinned memory
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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


int main() {
    printf("CSE 554 Assignment 1 - Section 3 Q1-Q2: Memory Transfer Bandwidth\n");
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

    printf("================================================================================\n");
    printf("Q1: Regular (Pageable) Memory\n");
    printf("================================================================================\n");

    // Test sizes from 2^0 to 2^28 (256 MB)
    for (int power = 0; power <= 28; power++) {
        size_t size = 1 << power;
        // Reduce iterations for very large sizes to avoid timeout
        int iterations = (power > 24) ? 10 : num_iterations;
        measure_bandwidth(size, iterations, false, csv_file);
    }

    printf("\n");
    printf("================================================================================\n");
    printf("Q2: Pinned (Page-Locked) Memory\n");
    printf("================================================================================\n");

    for (int power = 0; power <= 28; power++) {
        size_t size = 1 << power;
        int iterations = (power > 24) ? 10 : num_iterations;
        measure_bandwidth(size, iterations, true, csv_file);
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

    size_t large_size = 128 << 20;  // 128 MB - large enough to saturate bandwidth
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

    printf("Transfer size: 128 MB\n");
    printf("Peak Host-to-Device (pinned): %.2f GB/s\n", peak_h2d);
    printf("Peak Device-to-Host (pinned): %.2f GB/s\n", peak_d2h);
    printf("Average bidirectional: %.2f GB/s\n", (peak_h2d + peak_d2h) / 2.0f);
    printf("================================================================================\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_data));

    printf("\n✓ Memory transfer tests complete!\n");
    return 0;
}
