#pragma once

/**
 * CSE 554 Assignment 1 - Section 3 Q3: First Column Copy
 * Function declarations for efficiently copying first column from host to GPU
 *
 * Problem constraints (per TA comment):
 * - Input matrix is GIVEN in pageable (unpinned) CPU memory
 * - Destination buffer on GPU is GIVEN
 * - Any intermediate buffer allocation/deallocation MUST be included in timing
 */

/**
 * Optimized first column copy with end-to-end timing
 *
 * This function copies the first column of a row-major matrix from
 * pageable host memory to device memory. The timing includes:
 *   - Temporary buffer allocation
 *   - Data extraction from strided source
 *   - GPU transfer
 *   - Temporary buffer deallocation
 *
 * @param h_matrix Host matrix in PAGEABLE memory (row-major)
 * @param d_column Device memory to store first column (GIVEN)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Transfer time in milliseconds
 */
float copy_first_column_optimized(const float* h_matrix, float* d_column,
                                  int rows, int cols);

/**
 * Alternative method: CPU extract + pinned memory copy
 *
 * Extracts first column on CPU into pinned buffer, then copies to GPU.
 * Slower than cudaMemcpy2D due to CPU strided read bottleneck (~480 μs).
 *
 * @param h_matrix Host matrix in PAGEABLE memory (row-major)
 * @param d_column Device memory to store first column (GIVEN)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Transfer time in milliseconds
 */
float copy_first_column_cpu_extract(const float* h_matrix, float* d_column,
                                    int rows, int cols);

/**
 * Method 3: CPU extract + pageable memory copy (baseline/naive)
 * Slowest method - uses pageable memory with internal staging.
 */
float copy_first_column_naive(const float* h_matrix, float* d_column,
                              int rows, int cols);

/**
 * Initialize CUDA resources for copy_first_column functions
 * Call this before using copy_first_column_optimized
 */
void copy_first_column_init();

/**
 * Cleanup CUDA resources
 * Call this when done using copy_first_column functions
 */
void copy_first_column_cleanup();
