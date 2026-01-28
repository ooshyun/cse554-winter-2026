#pragma once

/**
 * CSE 554 Assignment 1 - Section 1: SiLU Implementation
 * Function declarations for CUDA SiLU (Sigmoid Linear Unit) activation
 */

/**
 * Basic SiLU kernel wrapper
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * @param d_input Device input array
 * @param d_output Device output array
 * @param n Number of elements
 */
void silu_cuda_basic(const float* d_input, float* d_output, int n);

/**
 * Optimized SiLU kernel wrapper with vectorized loads/stores (float4)
 * @param d_input Device input array
 * @param d_output Device output array
 * @param n Number of elements
 */
void silu_cuda_optimized(const float* d_input, float* d_output, int n);

/**
 * Fast SiLU - alias for selected best kernel
 * @param d_input Device input array
 * @param d_output Device output array
 * @param n Number of elements
 */
void silu_cuda_fast(const float* d_input, float* d_output, int n);

/**
 * Measure kernel execution time
 * @param kernel_func Kernel function pointer
 * @param d_input Device input array
 * @param d_output Device output array
 * @param n Number of elements
 * @param num_iterations Number of iterations for timing
 * @return Average execution time in milliseconds
 */
float measure_kernel_time(void (*kernel_func)(const float*, float*, int),
                          const float* d_input, float* d_output, int n,
                          int num_iterations);

/**
 * Calculate achieved bandwidth
 * @param n Number of elements
 * @param time_ms Execution time in milliseconds
 * @return Bandwidth in GB/s
 */
float calculate_bandwidth(int n, float time_ms);

/**
 * Get theoretical peak memory bandwidth (calculated from device properties)
 * @return Peak bandwidth in GB/s
 */
float get_peak_bandwidth_calculated();

/**
 * Get theoretical peak memory bandwidth (from datasheet)
 * @return Peak bandwidth in GB/s
 */
float get_peak_bandwidth_datasheet();
