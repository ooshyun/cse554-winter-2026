#pragma once

/**
 * CSE 554 Assignment 1 - Section 3 Q3: First Column Copy
 * Function declarations for efficiently copying first column from host to GPU
 */

/**
 * Naive approach: Extract first column on CPU, then copy to GPU
 * @param h_matrix Host matrix pointer (row-major)
 * @param d_column Device memory to store first column
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Transfer time in milliseconds
 */
float copy_first_column_naive(const float* h_matrix, float* d_column, int rows, int cols);

/**
 * Pre-extracted method: Extract first column to pinned memory, then fast contiguous copy
 * @param h_pinned_matrix Host matrix in pinned memory (row-major)
 * @param d_column Device memory to store first column
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Transfer time in milliseconds
 */
float copy_first_column_preextracted(float* h_pinned_matrix, float* d_column, int rows, int cols);

/**
 * Initialize CUDA resources for copy_first_column functions
 * Call this before using copy_first_column_preextracted
 */
void copy_first_column_init();

/**
 * Cleanup CUDA resources
 * Call this when done using copy_first_column functions
 */
void copy_first_column_cleanup();
