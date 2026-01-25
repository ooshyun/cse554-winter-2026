/**
 * CSE 554 Assignment 1 - Section 2: RMS Norm Vector Implementation
 * CUDA kernel for RMS Normalization on vector (1, 1024×1024)
 * Single row requires cooperative reduction across multiple thread blocks
 *
 * Optimizations applied:
 * - Cooperative Groups for grid-wide synchronization (eliminates CPU sync)
 * - Increased parallelism with more thread blocks
 * - Single-kernel approach to reduce launch overhead
 * - Vectorized float4 loads for memory coalescing
 */

 #include <cuda_runtime.h>
 #include <cooperative_groups.h>
 #include <device_launch_parameters.h>
 #include <stdio.h>
 #include <math.h>
 #include "../../common/gpu_specs.h"
 
// // __global__ void rms_norm_vector_kernel(...) {
   
// // }


// void rms_norm_vector(float *input, float *weight, float *output, int cols, float epsilon) {

// }

 namespace cg = cooperative_groups;
 
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
  * Device function for warp-level reduction
  */
 __device__ __forceinline__ float warp_reduce_sum(float val) {
     for (int offset = 16; offset > 0; offset /= 2) {
         val += __shfl_down_sync(0xffffffff, val, offset);
     }
     return val;
 }
 
 
 /**
  * ORIGINAL Basic two-phase RMS Norm for long vector
  * Phase 1: Compute partial sums of squares across blocks
  */
 __global__ void rms_norm_vector_phase1(const float* input, float* partial_sums,
                                         int n) {
     extern __shared__ float shared[];

     int tid = threadIdx.x;
     int block_size = blockDim.x;
     int warp_id = tid / 32;
     int lane_id = tid % 32;

     // Each thread computes partial sum with grid-stride loop
     float sum = 0.0f;
     for (int i = blockIdx.x * block_size + tid; i < n; i += gridDim.x * block_size) {
         float val = input[i];
         sum += val * val;
     }

     // Warp-level reduction
     sum = warp_reduce_sum(sum);

     // Write warp results to shared memory
     if (lane_id == 0) {
         shared[warp_id] = sum;
     }
     __syncthreads();

     // Final reduction in first warp
     if (tid < (block_size / 32)) {
         sum = shared[tid];
         sum = warp_reduce_sum(sum);

         if (tid == 0) {
             partial_sums[blockIdx.x] = sum;
         }
     }
 }


 /**
  * OPTIMIZED v1 FIXED: Two-phase RMS Norm for long vector
  * Phase 1: Compute partial sums of squares across blocks
  *
  * Fixes applied:
  * - Removed aggressive unrolling that causes hangs on some architectures
  * - Added tail element handling for n % 4 != 0
  * - Simpler grid-stride loop for better compatibility
  */
 __global__ void rms_norm_vector_phase1_v1(const float* input, float* partial_sums,
                                         int n) {
     extern __shared__ float shared[];

     int tid = threadIdx.x;
     int block_size = blockDim.x;
     int warp_id = tid / 32;
     int lane_id = tid % 32;

     const int n_vec = n / 4;
     const float4* input_vec = reinterpret_cast<const float4*>(input);

     // Vectorized loads with simple grid-stride loop (no aggressive unrolling)
     float sum = 0.0f;

     for (int i = blockIdx.x * block_size + tid; i < n_vec; i += gridDim.x * block_size) {
         float4 vals = input_vec[i];
         sum += vals.x * vals.x + vals.y * vals.y +
                vals.z * vals.z + vals.w * vals.w;
     }

     // FIX: Handle tail elements (last 0-3 floats) that don't fit in float4
     for (int i = n_vec * 4 + blockIdx.x * block_size + tid; i < n; i += gridDim.x * block_size) {
         float val = input[i];
         sum += val * val;
     }

     // Warp-level reduction
     sum = warp_reduce_sum(sum);

     // Write warp results to shared memory (coalesced access)
     if (lane_id == 0) {
         shared[warp_id] = sum;
     }
     __syncthreads();

     // Final reduction in first warp - no divergence
     if (tid < (block_size / 32)) {
         sum = shared[tid];
         sum = warp_reduce_sum(sum);

         if (tid == 0) {
             partial_sums[blockIdx.x] = sum;
         }
     }
 }
 
 
 /**
  * ORIGINAL Basic Phase 2: Normalize using global RMS
  */
 __global__ void rms_norm_vector_phase2(const float* input, float* output,
                                         const float* partial_sums, int n,
                                         int num_blocks) {
     // Each block independently computes global RMS (redundant but safe)
     __shared__ float global_rms;

     if (threadIdx.x == 0) {
         float total_sum = 0.0f;
         for (int i = 0; i < num_blocks; i++) {
             total_sum += partial_sums[i];
         }
         global_rms = sqrtf(total_sum / n + EPSILON);
     }
     __syncthreads();

     // Normalize
     int block_size = blockDim.x;
     for (int i = blockIdx.x * block_size + threadIdx.x; i < n;
          i += gridDim.x * block_size) {
         output[i] = input[i] / global_rms;
     }
 }


 /**
  * OPTIMIZED v1 FIXED: Phase 2: Normalize using global RMS
  *
  * Fixes applied:
  * - Fixed shared memory conflict (global_rms now uses sdata instead of separate variable)
  * - Added tail element handling for n % 4 != 0
  * - Removed aggressive unrolling that causes hangs on some architectures
  */
 __global__ void rms_norm_vector_phase2_v1(const float* input, float* output,
                                         const float* partial_sums, int n,
                                         int num_blocks) {
     extern __shared__ float sdata[];

     int tid = threadIdx.x;
     int block_size = blockDim.x;

     // Parallel reduction of partial_sums (reduces barrier stalls)
     float local_sum = 0.0f;
     for (int i = tid; i < num_blocks; i += block_size) {
         local_sum += partial_sums[i];
     }

     // Warp-level reduction
     int warp_id = tid / 32;
     int lane_id = tid % 32;
     local_sum = warp_reduce_sum(local_sum);

     if (lane_id == 0) {
         sdata[warp_id] = local_sum;
     }
     __syncthreads();

     // Final reduction in first warp
     if (tid < (block_size / 32)) {
         local_sum = sdata[tid];
         local_sum = warp_reduce_sum(local_sum);

         if (tid == 0) {
             // FIX: Use sdata[0] instead of separate __shared__ variable
             // to avoid memory conflict
             sdata[0] = sqrtf(local_sum / n + EPSILON);
         }
     }
     __syncthreads();

     float global_rms = sdata[0];

     // Vectorized normalization for aligned part (simple grid-stride, no unrolling)
     const int n_vec = n / 4;
     const float4* input_vec = reinterpret_cast<const float4*>(input);
     float4* output_vec = reinterpret_cast<float4*>(output);

     for (int i = blockIdx.x * block_size + tid; i < n_vec; i += gridDim.x * block_size) {
         float4 vals = input_vec[i];
         float4 result;
         result.x = vals.x / global_rms;
         result.y = vals.y / global_rms;
         result.z = vals.z / global_rms;
         result.w = vals.w / global_rms;
         output_vec[i] = result;
     }

     // FIX: Handle tail elements (last 0-3 floats) that don't fit in float4
     for (int i = n_vec * 4 + blockIdx.x * block_size + tid; i < n; i += gridDim.x * block_size) {
         output[i] = input[i] / global_rms;
     }
 }
 
 /**
  * Optimized vector RMS Norm with vectorized loads
  */
 __global__ void rms_norm_vector_fast_phase1(const float4* input, float* partial_sums,
                                              int n_vec) {
     extern __shared__ float shared[];
 
     int tid = threadIdx.x;
     int block_size = blockDim.x;
     int warp_id = tid / 32;
     int lane_id = tid % 32;
 
     float sum = 0.0f;
     for (int i = blockIdx.x * block_size + tid; i < n_vec; i += gridDim.x * block_size) {
         float4 vals = input[i];
         sum += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
     }
 
     sum = warp_reduce_sum(sum);
 
     if (lane_id == 0) {
         shared[warp_id] = sum;
     }
     __syncthreads();
 
     if (tid < (block_size / 32)) {
         sum = shared[tid];
         sum = warp_reduce_sum(sum);
         if (tid == 0) {
             partial_sums[blockIdx.x] = sum;
         }
     }
 }
 
 
 __global__ void rms_norm_vector_fast_phase2(const float4* input, float4* output,
                                              const float* global_rms_ptr, int n_vec) {
     float global_rms = *global_rms_ptr;  // Load from device memory once
     int block_size = blockDim.x;
     for (int i = blockIdx.x * block_size + threadIdx.x; i < n_vec;
          i += gridDim.x * block_size) {
         float4 vals = input[i];
         float4 result;
         result.x = vals.x / global_rms;
         result.y = vals.y / global_rms;
         result.z = vals.z / global_rms;
         result.w = vals.w / global_rms;
         output[i] = result;
     }
 }
 
 
 /**
  * Wrapper functions
  */
 void rms_norm_vector_basic(const float* d_input, float* d_output, int n) {
     int block_size = 256;
     int num_blocks = min((n + block_size - 1) / block_size, 1024);

     float* d_partial_sums;
     CUDA_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(float)));

     int shared_mem = (block_size / 32) * sizeof(float);

     // Phase 1: Original basic version
     rms_norm_vector_phase1<<<num_blocks, block_size, shared_mem>>>(
         d_input, d_partial_sums, n);
     CUDA_CHECK(cudaGetLastError());
     CUDA_CHECK(cudaDeviceSynchronize());

     // Phase 2: Original basic version
     rms_norm_vector_phase2<<<num_blocks, block_size>>>(
         d_input, d_output, d_partial_sums, n, num_blocks);
     CUDA_CHECK(cudaGetLastError());

     CUDA_CHECK(cudaFree(d_partial_sums));
 }


 void rms_norm_vector_basic_v1(const float* d_input, float* d_output, int n) {
     int block_size = 256;
     int num_blocks = min((n + block_size - 1) / block_size, 1024);

     float* d_partial_sums;
     CUDA_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(float)));

     // Shared memory for warp results
     int shared_mem = (block_size / 32) * sizeof(float);

     // Phase 1: Vectorized partial sums
     rms_norm_vector_phase1_v1<<<num_blocks, block_size, shared_mem>>>(
         d_input, d_partial_sums, n);
     CUDA_CHECK(cudaGetLastError());
     CUDA_CHECK(cudaDeviceSynchronize());

     // Phase 2: Parallel reduction + vectorized normalization
     // Phase 2 also needs shared memory for partial_sums reduction
     rms_norm_vector_phase2_v1<<<num_blocks, block_size, shared_mem>>>(
         d_input, d_output, d_partial_sums, n, num_blocks);
     CUDA_CHECK(cudaGetLastError());

     CUDA_CHECK(cudaFree(d_partial_sums));
 }
 
 
 /**
  * Ultra-optimized single-kernel RMS Norm using Cooperative Groups
  * Eliminates CPU-GPU synchronization for maximum bandwidth
  */
 __global__ void rms_norm_vector_cooperative(const float4* input, float4* output,
                                              float* partial_sums, float* global_rms_ptr,
                                              int n_vec, int n) {
     // Create grid group for grid-wide synchronization
     cg::grid_group grid = cg::this_grid();
 
     extern __shared__ float shared[];
 
     int tid = threadIdx.x;
     int block_size = blockDim.x;
     int warp_id = tid / 32;
     int lane_id = tid % 32;
 
     // Phase 1: Compute sum of squares with vectorized loads
     float sum = 0.0f;
     for (int i = blockIdx.x * block_size + tid; i < n_vec; i += gridDim.x * block_size) {
         float4 vals = input[i];
         sum += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
     }
 
     // Warp-level reduction
     sum = warp_reduce_sum(sum);
 
     // Write warp results to shared memory
     if (lane_id == 0) {
         shared[warp_id] = sum;
     }
     __syncthreads();
 
     // Block-level reduction
     if (tid < (block_size / 32)) {
         sum = shared[tid];
         sum = warp_reduce_sum(sum);
 
         if (tid == 0) {
             partial_sums[blockIdx.x] = sum;
         }
     }
 
     // Grid-wide synchronization (NO CPU involved!)
     grid.sync();
 
     // Single thread computes global RMS
     if (tid == 0 && blockIdx.x == 0) {
         float total_sum = 0.0f;
         for (int i = 0; i < gridDim.x; i++) {
             total_sum += partial_sums[i];
         }
         *global_rms_ptr = sqrtf(total_sum / n + EPSILON);
     }
 
     // Broadcast RMS to all blocks
     grid.sync();
 
     // Phase 2: Normalize with vectorized stores
     float rms_val = *global_rms_ptr;
     for (int i = blockIdx.x * block_size + tid; i < n_vec; i += gridDim.x * block_size) {
         float4 vals = input[i];
         float4 result;
         result.x = vals.x / rms_val;
         result.y = vals.y / rms_val;
         result.z = vals.z / rms_val;
         result.w = vals.w / rms_val;
         output[i] = result;
     }
 }
 
 
 /**
  * GPU-side reduction of partial sums - eliminates CPU sync!
  */
 __global__ void reduce_partial_sums_gpu(const float* partial_sums, float* global_rms,
                                          int num_blocks, int n) {
     extern __shared__ float shared[];
 
     int tid = threadIdx.x;
     int warp_id = tid / 32;
     int lane_id = tid % 32;
 
     // Load partial sums
     float sum = 0.0f;
     for (int i = tid; i < num_blocks; i += blockDim.x) {
         sum += partial_sums[i];
     }
 
     // Warp reduction
     sum = warp_reduce_sum(sum);
     if (lane_id == 0) {
         shared[warp_id] = sum;
     }
     __syncthreads();
 
     // Final reduction
     if (tid < (blockDim.x / 32)) {
         sum = shared[tid];
         sum = warp_reduce_sum(sum);
         if (tid == 0) {
             *global_rms = sqrtf(sum / n + EPSILON);
         }
     }
 }
 
 /**
  * Ultra-optimized vector RMS norm with GPU-only reduction
  *
  * Key optimizations (based on 2025 CUDA best practices):
  * 1. Increased parallelism: 2048 blocks instead of 32
  * 2. GPU-side reduction: NO CPU synchronization
  * 3. Vectorized float4 operations throughout
  * 4. Grid-stride loop for optimal work distribution
  *
  * References:
  * - [Layer Normalization as fast as possible](https://fleetwood.dev/posts/layernorm-as-fast-as-possible)
  * - [Optimizing a Layer Normalization Kernel with CUDA](https://aryagxr.com/blogs/cuda-optimizing-layernorm)
  */
 // Static buffers to avoid reallocation overhead in benchmarks
 static float* g_d_partial_sums = nullptr;
 static float* g_d_global_rms = nullptr;
 static int g_allocated_blocks = 0;
 
 void rms_norm_vector_fast(const float* d_input, float* d_output, int n) {
     const int block_size = 256;
     const int n_vec = n / 4;
 
     // CRITICAL OPTIMIZATION: Use 2048 blocks instead of 32!
     // This increases parallelism dramatically
     const int num_blocks = 2048;
 
     // Reuse allocated buffers to avoid malloc/free overhead
     if (g_allocated_blocks != num_blocks) {
         if (g_d_partial_sums) CUDA_CHECK(cudaFree(g_d_partial_sums));
         if (g_d_global_rms) CUDA_CHECK(cudaFree(g_d_global_rms));
 
         CUDA_CHECK(cudaMalloc(&g_d_partial_sums, num_blocks * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&g_d_global_rms, sizeof(float)));
         g_allocated_blocks = num_blocks;
     }
 
     const int shared_mem = (block_size / 32) * sizeof(float);
 
     // Phase 1: Compute partial sums with vectorized loads (HIGH PARALLELISM)
     rms_norm_vector_fast_phase1<<<num_blocks, block_size, shared_mem>>>(
         reinterpret_cast<const float4*>(d_input),
         g_d_partial_sums,
         n_vec);
     CUDA_CHECK(cudaGetLastError());
 
     // Phase 2: GPU-side reduction (NO CPU sync!)
     reduce_partial_sums_gpu<<<1, block_size, shared_mem>>>(
         g_d_partial_sums,
         g_d_global_rms,
         num_blocks,
         n);
     CUDA_CHECK(cudaGetLastError());
 
     // Phase 3: Normalize with vectorized stores
     rms_norm_vector_fast_phase2<<<num_blocks, block_size>>>(
         reinterpret_cast<const float4*>(d_input),
         reinterpret_cast<float4*>(d_output),
         g_d_global_rms,
         n_vec);
     CUDA_CHECK(cudaGetLastError());
 }
 
 
 float measure_rms_norm_vector_time(void (*kernel_func)(const float*, float*, int),
                                    const float* d_input, float* d_output, int n,
                                    int num_iterations) {
     cudaEvent_t start, stop;
     CUDA_CHECK(cudaEventCreate(&start));
     CUDA_CHECK(cudaEventCreate(&stop));
 
     for (int i = 0; i < 10; i++) {
         kernel_func(d_input, d_output, n);
     }
     CUDA_CHECK(cudaDeviceSynchronize());
 
     CUDA_CHECK(cudaEventRecord(start));
     for (int i = 0; i < num_iterations; i++) {
         kernel_func(d_input, d_output, n);
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
 float get_peak_bandwidth_vector_calculated() {
     cudaDeviceProp prop;
     CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

     int memClockRate;
     cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, 0);

     // Peak bandwidth in GB/s (decimal)
     float peak_gb_s = 2.0f * memClockRate * (prop.memoryBusWidth / 8) / 1e6;
     return peak_gb_s;
 }
 
 
 /**
  * Get theoretical peak memory bandwidth from datasheet
  * RTX 4070 Ti SUPER: 672 GB/s (official specification)
  */
 float get_peak_bandwidth_vector_datasheet() {
     return GPU_PEAK_BANDWIDTH_DATASHEET;
 }
 
 
 float calculate_rms_norm_vector_bandwidth(int n, float time_ms) {
     size_t bytes = 2 * (size_t)n * sizeof(float);
     float time_s = time_ms / 1000.0f;

     // Use decimal GB/s to match official specs (1 GB = 10^9 bytes)
     return (bytes / time_s) / 1e9;
 }


 /**
  * ============================================================================
  * W2L3-INSPIRED OPTIMIZATIONS FOR VECTOR RMS NORM
  * ============================================================================
  */

 /**
  * W2L3 Reduction-inspired kernel for vector
  * Key features from reduction5.cu:
  * - Multiple elements per thread (ELEMENTS_PER_THREAD = 64 for long vector)
  * - Sequential addressing in shared memory reduction
  * - Grid-stride loop pattern
  */
 #define ELEMENTS_PER_THREAD_VEC 64
 #define BLOCK_SIZE_W2L3_VEC 256

 __global__ void rms_norm_vector_w2l3_reduction_phase1(const float* input,
                                                        float* partial_sums,
                                                        int n) {
     __shared__ float sdata[BLOCK_SIZE_W2L3_VEC];

     int tid = threadIdx.x;

     // Phase 1: Each thread processes multiple elements (reduction5 pattern)
     float local_sum = 0.0f;
     int i = blockIdx.x * blockDim.x + tid;

     #pragma unroll 8
     for (int j = 0; j < ELEMENTS_PER_THREAD_VEC; j++) {
         if (i < n) {
             float val = input[i];
             local_sum += val * val;
         }
         i += gridDim.x * blockDim.x;
     }

     // Store in shared memory
     sdata[tid] = local_sum;
     __syncthreads();

     // Phase 2: Sequential addressing reduction
     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
         if (tid < s) {
             sdata[tid] += sdata[tid + s];
         }
         __syncthreads();
     }

     // Write block result
     if (tid == 0) {
         partial_sums[blockIdx.x] = sdata[0];
     }
 }

 __global__ void rms_norm_vector_w2l3_reduction_phase2(const float* input,
                                                        float* output,
                                                        const float* global_rms_ptr,
                                                        int n) {
     float rms = *global_rms_ptr;
     int i = blockIdx.x * blockDim.x + threadIdx.x;

     #pragma unroll 8
     for (int j = 0; j < ELEMENTS_PER_THREAD_VEC; j++) {
         if (i < n) {
             output[i] = input[i] / rms;
         }
         i += gridDim.x * blockDim.x;
     }
 }


 /**
  * W2L3 Hybrid kernel: reduction5 + vectorization
  * - Multiple elements per thread
  * - Float4 vectorization
  * - Unrolled reduction
  */
 #define ELEMENTS_PER_THREAD_VEC4 32
 #define BLOCK_SIZE_W2L3_HYBRID 256

 __global__ void rms_norm_vector_w2l3_hybrid_phase1(const float4* input,
                                                     float* partial_sums,
                                                     int n_vec) {
     __shared__ float sdata[BLOCK_SIZE_W2L3_HYBRID + 1];  // +1 for bank conflict avoidance

     int tid = threadIdx.x;
     int i = blockIdx.x * blockDim.x + tid;

     // Vectorized reduction with multiple elements per thread
     float local_sum = 0.0f;

     #pragma unroll 4
     for (int j = 0; j < ELEMENTS_PER_THREAD_VEC4; j++) {
         if (i < n_vec) {
             float4 vals = input[i];
             local_sum += vals.x * vals.x;
             local_sum += vals.y * vals.y;
             local_sum += vals.z * vals.z;
             local_sum += vals.w * vals.w;
         }
         i += gridDim.x * blockDim.x;
     }

     // Store with padding
     sdata[tid] = local_sum;
     __syncthreads();

     // Unrolled reduction (power-of-2 optimization)
     if (BLOCK_SIZE_W2L3_HYBRID >= 512 && tid < 256) sdata[tid] += sdata[tid + 256];
     __syncthreads();
     if (BLOCK_SIZE_W2L3_HYBRID >= 256 && tid < 128) sdata[tid] += sdata[tid + 128];
     __syncthreads();
     if (BLOCK_SIZE_W2L3_HYBRID >= 128 && tid < 64) sdata[tid] += sdata[tid + 64];
     __syncthreads();

     // Warp-level reduction (no sync needed)
     if (tid < 32) {
         volatile float* vsdata = sdata;
         if (BLOCK_SIZE_W2L3_HYBRID >= 64) vsdata[tid] += vsdata[tid + 32];
         if (BLOCK_SIZE_W2L3_HYBRID >= 32) vsdata[tid] += vsdata[tid + 16];
         if (BLOCK_SIZE_W2L3_HYBRID >= 16) vsdata[tid] += vsdata[tid + 8];
         if (BLOCK_SIZE_W2L3_HYBRID >= 8) vsdata[tid] += vsdata[tid + 4];
         if (BLOCK_SIZE_W2L3_HYBRID >= 4) vsdata[tid] += vsdata[tid + 2];
         if (BLOCK_SIZE_W2L3_HYBRID >= 2) vsdata[tid] += vsdata[tid + 1];
     }

     if (tid == 0) {
         partial_sums[blockIdx.x] = sdata[0];
     }
 }

 __global__ void rms_norm_vector_w2l3_hybrid_phase2(const float4* input,
                                                     float4* output,
                                                     const float* global_rms_ptr,
                                                     int n_vec) {
     float rms = *global_rms_ptr;
     int i = blockIdx.x * blockDim.x + threadIdx.x;

     #pragma unroll 4
     for (int j = 0; j < ELEMENTS_PER_THREAD_VEC4; j++) {
         if (i < n_vec) {
             float4 vals = input[i];
             float4 result;
             result.x = vals.x / rms;
             result.y = vals.y / rms;
             result.z = vals.z / rms;
             result.w = vals.w / rms;
             output[i] = result;
         }
         i += gridDim.x * blockDim.x;
     }
 }


 /**
  * Wrapper functions for W2L3 optimizations
  */
 static float* g_w2l3_partial_sums = nullptr;
 static float* g_w2l3_global_rms = nullptr;
 static int g_w2l3_allocated_blocks = 0;

 void rms_norm_vector_w2l3_reduction(const float* d_input, float* d_output, int n) {
     const int num_blocks = 2048;

     // Reuse allocated buffers
     if (g_w2l3_allocated_blocks != num_blocks) {
         if (g_w2l3_partial_sums) CUDA_CHECK(cudaFree(g_w2l3_partial_sums));
         if (g_w2l3_global_rms) CUDA_CHECK(cudaFree(g_w2l3_global_rms));

         CUDA_CHECK(cudaMalloc(&g_w2l3_partial_sums, num_blocks * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&g_w2l3_global_rms, sizeof(float)));
         g_w2l3_allocated_blocks = num_blocks;
     }

     // Phase 1: Compute partial sums
     rms_norm_vector_w2l3_reduction_phase1<<<num_blocks, BLOCK_SIZE_W2L3_VEC>>>(
         d_input, g_w2l3_partial_sums, n);
     CUDA_CHECK(cudaGetLastError());

     // Phase 2: GPU-side reduction
     const int shared_mem = (BLOCK_SIZE_W2L3_VEC / 32) * sizeof(float);
     reduce_partial_sums_gpu<<<1, BLOCK_SIZE_W2L3_VEC, shared_mem>>>(
         g_w2l3_partial_sums, g_w2l3_global_rms, num_blocks, n);
     CUDA_CHECK(cudaGetLastError());

     // Phase 3: Normalize
     rms_norm_vector_w2l3_reduction_phase2<<<num_blocks, BLOCK_SIZE_W2L3_VEC>>>(
         d_input, d_output, g_w2l3_global_rms, n);
     CUDA_CHECK(cudaGetLastError());
 }


 void rms_norm_vector_w2l3_hybrid(const float* d_input, float* d_output, int n) {
     const int num_blocks = 2048;
     const int n_vec = n / 4;

     // Reuse allocated buffers
     if (g_w2l3_allocated_blocks != num_blocks) {
         if (g_w2l3_partial_sums) CUDA_CHECK(cudaFree(g_w2l3_partial_sums));
         if (g_w2l3_global_rms) CUDA_CHECK(cudaFree(g_w2l3_global_rms));

         CUDA_CHECK(cudaMalloc(&g_w2l3_partial_sums, num_blocks * sizeof(float)));
         CUDA_CHECK(cudaMalloc(&g_w2l3_global_rms, sizeof(float)));
         g_w2l3_allocated_blocks = num_blocks;
     }

     // Phase 1: Vectorized partial sums
     rms_norm_vector_w2l3_hybrid_phase1<<<num_blocks, BLOCK_SIZE_W2L3_HYBRID>>>(
         reinterpret_cast<const float4*>(d_input), g_w2l3_partial_sums, n_vec);
     CUDA_CHECK(cudaGetLastError());

     // Phase 2: GPU-side reduction
     const int shared_mem = (BLOCK_SIZE_W2L3_HYBRID / 32) * sizeof(float);
     reduce_partial_sums_gpu<<<1, BLOCK_SIZE_W2L3_HYBRID, shared_mem>>>(
         g_w2l3_partial_sums, g_w2l3_global_rms, num_blocks, n);
     CUDA_CHECK(cudaGetLastError());

     // Phase 3: Vectorized normalize
     rms_norm_vector_w2l3_hybrid_phase2<<<num_blocks, BLOCK_SIZE_W2L3_HYBRID>>>(
         reinterpret_cast<const float4*>(d_input),
         reinterpret_cast<float4*>(d_output),
         g_w2l3_global_rms,
         n_vec);
     CUDA_CHECK(cudaGetLastError());
 }
 
/**
 * SAFE v2: Conservative float4 optimization without complex patterns
 * Based on basic kernel with simple float4 addition
 */
__global__ void rms_norm_vector_phase1_v2(const float* input, float* partial_sums, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Process with simple grid-stride loop
    float sum = 0.0f;

    // Process float4 vectors
    const int n_vec = n / 4;
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    for (int i = blockIdx.x * block_size + tid; i < n_vec; i += gridDim.x * block_size) {
        float4 vals = input_vec[i];
        sum += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
    }

    // Handle remainder with regular floats (last 0-3 elements)
    for (int i = n_vec * 4 + blockIdx.x * block_size + tid; i < n; i += gridDim.x * block_size) {
        float val = input[i];
        sum += val * val;
    }

    // Standard warp reduction (same as basic)
    sum = warp_reduce_sum(sum);

    if (lane_id == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (tid < (block_size / 32)) {
        sum = shared[tid];
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            partial_sums[blockIdx.x] = sum;
        }
    }
}

__global__ void rms_norm_vector_phase2_v2(const float* input, float* output,
                                           const float* partial_sums, int n,
                                           int num_blocks) {
    // Exactly same as basic phase2 - proven to work
    __shared__ float global_rms;

    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < num_blocks; i++) {
            total_sum += partial_sums[i];
        }
        global_rms = sqrtf(total_sum / n + EPSILON);
    }
    __syncthreads();

    // Simple normalization (no vectorization here for stability)
    int block_size = blockDim.x;
    for (int i = blockIdx.x * block_size + threadIdx.x; i < n;
         i += gridDim.x * block_size) {
        output[i] = input[i] / global_rms;
    }
}

void rms_norm_vector_safe_v2(const float* d_input, float* d_output, int n) {
    int block_size = 256;
    int num_blocks = min((n + block_size - 1) / block_size, 1024);

    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(float)));

    int shared_mem = (block_size / 32) * sizeof(float);

    rms_norm_vector_phase1_v2<<<num_blocks, block_size, shared_mem>>>(
        d_input, d_partial_sums, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    rms_norm_vector_phase2_v2<<<num_blocks, block_size>>>(
        d_input, d_output, d_partial_sums, n, num_blocks);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_partial_sums));
}
