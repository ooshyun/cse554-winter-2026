#pragma once

/**
 * CSE 554 Assignment 1 - Section 2: RMS Norm Vector Implementation
 * Function declarations for CUDA RMS Normalization on vector (1, 1024×1024)
 */

/**
 * Basic two-phase RMS Norm kernel wrapper
 * Phase 1: Compute partial sums AND copy input to output
 * Phase 2: Normalize in-place
 * @param d_input Device input array
 * @param d_output Device output array
 * @param n Number of elements
 */
void rms_norm_vector_basic(const float* d_input, float* d_output, int n);

/**
 * Cooperative groups single-kernel RMS Norm
 * Combines both phases using grid.sync()
 * @param d_input Device input array
 * @param d_output Device output array
 * @param n Number of elements
 */
void rms_norm_vector_coop(const float* d_input, float* d_output, int n);

/**
 * Cleanup function for static buffers
 * Call before program exit or when switching contexts
 */
void rms_norm_vector_cleanup();

/**
 * Measure RMS Norm kernel execution time
 * @param kernel_func Kernel function pointer
 * @param d_input Device input array
 * @param d_output Device output array
 * @param n Number of elements
 * @param num_iterations Number of iterations for timing
 * @return Average execution time in milliseconds
 */
float measure_rms_norm_vector_time(void (*kernel_func)(const float*, float*, int),
                                   const float* d_input, float* d_output, int n,
                                   int num_iterations);

/**
 * Calculate achieved bandwidth for RMS Norm vector operation
 * @param n Number of elements
 * @param time_ms Execution time in milliseconds
 * @return Bandwidth in GB/s
 */
float calculate_rms_norm_vector_bandwidth(int n, float time_ms);

/**
 * Get theoretical peak memory bandwidth (calculated from device properties)
 * @return Peak bandwidth in GB/s
 */
float get_peak_bandwidth_vector_calculated();

/**
 * Get theoretical peak memory bandwidth (from datasheet)
 * @return Peak bandwidth in GB/s
 */
float get_peak_bandwidth_vector_datasheet();
