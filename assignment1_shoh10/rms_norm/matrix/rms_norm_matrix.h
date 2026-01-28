#pragma once

/**
 * CSE 554 Assignment 1 - Section 2: RMS Norm Matrix Implementation
 * Function declarations for CUDA RMS Normalization on matrix (8192, 8192)
 */

/**
 * Basic RMS Norm kernel wrapper - each row normalized independently
 * One thread per row
 * @param d_input Device input matrix
 * @param d_output Device output matrix
 * @param rows Number of rows
 * @param cols Number of columns
 */
void rms_norm_matrix_basic(const float* d_input, float* d_output, int rows, int cols);

/**
 * Fast RMS Norm kernel with warp-level primitives and vectorized loads
 * Multiple threads per row with shared memory coordination
 * @param d_input Device input matrix
 * @param d_output Device output matrix
 * @param rows Number of rows
 * @param cols Number of columns
 */
void rms_norm_matrix_fast(const float* d_input, float* d_output, int rows, int cols);

/**
 * W2L3 Hybrid RMS Norm kernel - combined optimizations
 * - Multiple elements per thread (reduction5)
 * - Vectorized loads (float4 for memory bandwidth)
 * - Bank conflict avoidance (transpose_v5)
 * - Unrolled loops for better performance
 * @param d_input Device input matrix
 * @param d_output Device output matrix
 * @param rows Number of rows
 * @param cols Number of columns
 */
void rms_norm_matrix_w2l3_hybrid(const float* d_input, float* d_output, int rows, int cols);

/**
 * Measure RMS Norm kernel execution time
 * @param kernel_func Kernel function pointer
 * @param d_input Device input array
 * @param d_output Device output array
 * @param rows Number of rows
 * @param cols Number of columns
 * @param num_iterations Number of iterations for timing
 * @return Average execution time in milliseconds
 */
float measure_rms_norm_time(void (*kernel_func)(const float*, float*, int, int),
                            const float* d_input, float* d_output,
                            int rows, int cols, int num_iterations);

/**
 * Calculate achieved bandwidth for RMS Norm matrix operation
 * @param rows Number of rows
 * @param cols Number of columns
 * @param time_ms Execution time in milliseconds
 * @return Bandwidth in GB/s
 */
float calculate_rms_norm_bandwidth(int rows, int cols, float time_ms);

/**
 * Get theoretical peak memory bandwidth (calculated from device properties)
 * @return Peak bandwidth in GB/s
 */
float get_peak_bandwidth_rms_norm_calculated();

/**
 * Get theoretical peak memory bandwidth (from datasheet)
 * @return Peak bandwidth in GB/s
 */
float get_peak_bandwidth_rms_norm_datasheet();
