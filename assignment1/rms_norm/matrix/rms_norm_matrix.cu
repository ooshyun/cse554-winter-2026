/**
 * CSE 554 Assignment 1 - Section 2: RMS Norm Matrix Implementation
 * CUDA kernel for RMS Normalization on matrix (8192, 8192)
 */

 #include <cuda_runtime.h>
 #include <device_launch_parameters.h>
 #include <stdio.h>
 #include <math.h>
 #include "../../common/gpu_specs.h"
 
 #define CUDA_CHECK(call) \
     do { \
         cudaError_t err = call; \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err)); \
             exit(EXIT_FAILURE); \
         } \
     } while(0)
 
 #define EPSILON 1e-6f
 
 
 /**
  * Basic RMS Norm kernel - each row normalized independently
  * RMSNorm(x_i) = x_i / sqrt((1/n) * sum(x_j^2) + epsilon)
  */
 __global__ void rms_norm_kernel_basic(const float* input, float* output,
                                        int rows, int cols) {
     int row = blockIdx.x * blockDim.x + threadIdx.x;
 
     if (row < rows) {
         // Calculate sum of squares for this row
         float sum_sq = 0.0f;
         for (int col = 0; col < cols; col++) {
             int idx = row * cols + col;
             float val = input[idx];
             sum_sq += val * val;
         }
 
         // Calculate RMS
         float rms = sqrtf(sum_sq / cols + EPSILON);
 
         // Normalize the row
         for (int col = 0; col < cols; col++) {
             int idx = row * cols + col;
             output[idx] = input[idx] / rms;
         }
     }
 }
 
 
 /**
  * Optimized RMS Norm kernel with shared memory reduction
  * Each thread block processes one row
  */
 __global__ void rms_norm_kernel_optimized(const float* input, float* output,
                                            int rows, int cols) {
     extern __shared__ float shared[];
 
     int row = blockIdx.x;
     int tid = threadIdx.x;
     int block_size = blockDim.x;
 
     if (row >= rows) return;
 
     // Each thread computes partial sum of squares
     float partial_sum = 0.0f;
     for (int col = tid; col < cols; col += block_size) {
         int idx = row * cols + col;
         float val = input[idx];
         partial_sum += val * val;
     }
 
     // Store partial sum in shared memory
     shared[tid] = partial_sum;
     __syncthreads();
 
     // Reduce within block using shared memory
     for (int stride = block_size / 2; stride > 0; stride >>= 1) {
         if (tid < stride) {
             shared[tid] += shared[tid + stride];
         }
         __syncthreads();
     }
 
     // Thread 0 has the total sum
     __shared__ float rms;
     if (tid == 0) {
         float sum_sq = shared[0];
         rms = sqrtf(sum_sq / cols + EPSILON);
     }
     __syncthreads();
 
     // Normalize the row
     for (int col = tid; col < cols; col += block_size) {
         int idx = row * cols + col;
         output[idx] = input[idx] / rms;
     }
 }
 
 
 /**
  * Highly optimized RMS Norm kernel
  * - Warp-level primitives for reduction
  * - Vectorized loads/stores where possible
  * - Optimized memory access patterns
  */
 __device__ __forceinline__ float warp_reduce_sum(float val) {
     for (int offset = 16; offset > 0; offset /= 2) {
         val += __shfl_down_sync(0xffffffff, val, offset);
     }
     return val;
 }
 
 
 __global__ void rms_norm_kernel_fast(const float* input, float* output,
                                       int rows, int cols) {
     extern __shared__ float shared[];
 
     int row = blockIdx.x;
     int tid = threadIdx.x;
     int warp_id = tid / 32;
     int lane_id = tid % 32;
     int block_size = blockDim.x;
 
     if (row >= rows) return;
 
     const float* row_input = input + row * cols;
     float* row_output = output + row * cols;
 
     // Phase 1: Compute sum of squares using vectorized loads
     float partial_sum = 0.0f;
 
     // Vectorized processing using float4
     int col = tid * 4;
     if (col + 3 < cols) {
         for (; col + 3 < cols; col += block_size * 4) {
             float4 vals = *reinterpret_cast<const float4*>(&row_input[col]);
             partial_sum += vals.x * vals.x;
             partial_sum += vals.y * vals.y;
             partial_sum += vals.z * vals.z;
             partial_sum += vals.w * vals.w;
         }
     }
 
     // Handle remaining elements
     for (col = tid + (cols / 4) * 4; col < cols; col += block_size) {
         float val = row_input[col];
         partial_sum += val * val;
     }
 
     // Warp-level reduction
     partial_sum = warp_reduce_sum(partial_sum);
 
     // Write warp results to shared memory
     if (lane_id == 0) {
         shared[warp_id] = partial_sum;
     }
     __syncthreads();
 
     // Final reduction across warps
     if (tid == 0) {
         float sum = 0.0f;
         int num_warps = (block_size + 31) / 32;
         for (int i = 0; i < num_warps; i++) {
             sum += shared[i];
         }
         shared[0] = sqrtf(sum / cols + EPSILON);
     }
     __syncthreads();
 
     float rms = shared[0];
 
     // Phase 2: Normalize with vectorized stores
     col = tid * 4;
     if (col + 3 < cols) {
         for (; col + 3 < cols; col += block_size * 4) {
             float4 vals = *reinterpret_cast<const float4*>(&row_input[col]);
             float4 result;
             result.x = vals.x / rms;
             result.y = vals.y / rms;
             result.z = vals.z / rms;
             result.w = vals.w / rms;
             *reinterpret_cast<float4*>(&row_output[col]) = result;
         }
     }
 
     // Handle remaining elements
     for (col = tid + (cols / 4) * 4; col < cols; col += block_size) {
         row_output[col] = row_input[col] / rms;
     }
 }
 
 
 /**
  * Wrapper functions
  */
 void rms_norm_matrix_basic(const float* d_input, float* d_output,
                            int rows, int cols) {
     int block_size = 256;
     int grid_size = (rows + block_size - 1) / block_size;
 
     rms_norm_kernel_basic<<<grid_size, block_size>>>(d_input, d_output,
                                                       rows, cols);
     CUDA_CHECK(cudaGetLastError());
 }
 
 
 void rms_norm_matrix_optimized(const float* d_input, float* d_output,
                                int rows, int cols) {
     int block_size = 256;
     int grid_size = rows;
     int shared_mem = block_size * sizeof(float);
 
     rms_norm_kernel_optimized<<<grid_size, block_size, shared_mem>>>(
         d_input, d_output, rows, cols);
     CUDA_CHECK(cudaGetLastError());
 }
 
 
 void rms_norm_matrix_fast(const float* d_input, float* d_output,
                           int rows, int cols) {
     int block_size = 256;
     int grid_size = rows;
     int shared_mem = (block_size / 32) * sizeof(float);
 
     rms_norm_kernel_fast<<<grid_size, block_size, shared_mem>>>(
         d_input, d_output, rows, cols);
     CUDA_CHECK(cudaGetLastError());
 }
 
 
 /**
  * Measure kernel execution time
  */
 float measure_rms_norm_time(void (*kernel_func)(const float*, float*, int, int),
                             const float* d_input, float* d_output,
                             int rows, int cols, int num_iterations) {
     cudaEvent_t start, stop;
     CUDA_CHECK(cudaEventCreate(&start));
     CUDA_CHECK(cudaEventCreate(&stop));
 
     // Warmup
     for (int i = 0; i < 10; i++) {
         kernel_func(d_input, d_output, rows, cols);
     }
     CUDA_CHECK(cudaDeviceSynchronize());
 
     // Measure
     CUDA_CHECK(cudaEventRecord(start));
     for (int i = 0; i < num_iterations; i++) {
         kernel_func(d_input, d_output, rows, cols);
     }
     CUDA_CHECK(cudaEventRecord(stop));
     CUDA_CHECK(cudaEventSynchronize(stop));
 
     float milliseconds = 0;
     CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
 
     CUDA_CHECK(cudaEventDestroy(start));
     CUDA_CHECK(cudaEventDestroy(stop));
 
     return milliseconds / num_iterations;
 }
 
 
 /**
  * Get theoretical peak memory bandwidth from device properties (calculated)
  */
 float get_peak_bandwidth_rms_norm_calculated() {
     cudaDeviceProp prop;
     CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
 
     // Peak bandwidth in GB/s (decimal)
     float peak_gb_s = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
     return peak_gb_s;
 }
 
 
 /**
  * Get theoretical peak memory bandwidth from datasheet
  * RTX 4070 Ti SUPER: 672 GB/s (official specification)
  */
 float get_peak_bandwidth_rms_norm_datasheet() {
     return GPU_PEAK_BANDWIDTH_DATASHEET;
 }
 
 
 /**
  * Calculate bandwidth utilization and performance percentage
  */
 float calculate_rms_norm_bandwidth(int rows, int cols, float time_ms) {
     // Memory accesses: 1 read + 1 write = 2 * rows * cols * sizeof(float)
     size_t bytes = 2 * (size_t)rows * (size_t)cols * sizeof(float);
     float time_s = time_ms / 1000.0f;
 
     // Use decimal GB/s to match official specs (1 GB = 10^9 bytes)
     float bandwidth_gb_s = (bytes / time_s) / 1e9;
     return bandwidth_gb_s;
 }
 