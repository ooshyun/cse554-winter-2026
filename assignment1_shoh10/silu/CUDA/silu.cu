/**
 * CSE 554 Assignment 1 - Section 1: SiLU Implementation in CUDA
 * CUDA kernel for Sigmoid Linear Unit (SiLU) activation function
 */

#include "silu.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
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


/**
 * Basic SiLU kernel
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
__global__ void silu_kernel_basic(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = x * sigmoid;
    }
}


/**
 * Optimized SiLU kernel with vectorized loads/stores
 * Uses float4 for coalesced memory access
 */
__global__ void silu_kernel_optimized(const float* input, float* output, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < n) {
        // Vectorized load using float4
        float4 x = *reinterpret_cast<const float4*>(&input[idx]);

        // Compute SiLU for each element
        float4 result;
        result.x = x.x / (1.0f + expf(-x.x));
        result.y = x.y / (1.0f + expf(-x.y));
        result.z = x.z / (1.0f + expf(-x.z));
        result.w = x.w / (1.0f + expf(-x.w));

        // Vectorized store
        *reinterpret_cast<float4*>(&output[idx]) = result;
    }
    else if (idx < n) {
        // Handle remaining elements
        for (int i = idx; i < n && i < idx + 4; i++) {
            float x_val = input[i];
            output[i] = x_val / (1.0f + expf(-x_val));
        }
    }
}


/**
 * Wrapper function for basic SiLU kernel
 */
void silu_cuda_basic(const float* d_input, float* d_output, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    silu_kernel_basic<<<grid_size, block_size>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
}


/**
 * Wrapper function for optimized SiLU kernel
 */
void silu_cuda_optimized(const float* d_input, float* d_output, int n) {
    // Use vectorized kernel
    int block_size = 256;
    int grid_size = ((n / 4) + block_size - 1) / block_size;

    silu_kernel_optimized<<<grid_size, block_size>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
}


/**
 * Fast SiLU - alias for selected best kernel (basic for now)
 */
void silu_cuda_fast(const float* d_input, float* d_output, int n) {
    silu_cuda_basic(d_input, d_output, n);
}


/**
 * Measure kernel execution time
 */
float measure_kernel_time(void (*kernel_func)(const float*, float*, int),
                          const float* d_input, float* d_output, int n,
                          int num_iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel_func(d_input, d_output, n);
    }
    cudaDeviceSynchronize();

    // Measure
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        kernel_func(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / static_cast<float>(num_iterations);
}


/**
 * Get theoretical peak memory bandwidth from device properties (calculated)
 */
float get_peak_bandwidth_calculated() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int memClockRate;
    cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, 0);

    // Peak bandwidth in GB/s (decimal)
    // Formula: 2 × memory_clock_rate (kHz) × bus_width (bytes) / 1e6
    float peak_gb_s = 2.0f * static_cast<float>(memClockRate) * (static_cast<float>(prop.memoryBusWidth) / 8.0f) / 1e6f;
    return peak_gb_s;
}


/**
 * Get theoretical peak memory bandwidth from datasheet
 */
float get_peak_bandwidth_datasheet() {
    // Use constant from gpu_specs.h
    return GPU_PEAK_BANDWIDTH_DATASHEET;
}


/**
 * Calculate bandwidth utilization and performance percentage
 */
float calculate_bandwidth(int n, float time_ms) {
    // Memory accesses: 1 read + 1 write = 2 * n * sizeof(float)
    size_t bytes = 2 * static_cast<size_t>(n) * sizeof(float);
    float time_s = time_ms / 1000.0f;

    // Use decimal GB/s to match official specs (1 GB = 10^9 bytes)
    float bandwidth_gb_s = (static_cast<float>(bytes) / time_s) / 1e9f;
    return bandwidth_gb_s;
}
