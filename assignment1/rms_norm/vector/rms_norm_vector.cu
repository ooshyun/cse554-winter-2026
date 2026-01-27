/**
 * CSE 554 Assignment 1 - Section 2: RMS Norm Vector Implementation
 * CUDA kernel for RMS Normalization on vector (1, 1024×1024)
 * Single row requires cooperative reduction across multiple thread blocks
 *
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
  * ORIGINAL Basic Phase 2: Normalize using global RMS
  */
__global__ void rms_norm_vector_phase2(const float* input, float* output,
                                        const float* partial_sums, int n,
                                        int num_blocks) {
    // // Each block independently computes global RMS (redundant but safe)
    // __shared__ float global_rms;

    // if (threadIdx.x == 0) {
    //     float total_sum = 0.0f;
    //     for (int i = 0; i < num_blocks; i++) {
    //         total_sum += partial_sums[i];
    //     }
    //     global_rms = sqrtf(total_sum / n + EPSILON);
    // }
    // __syncthreads();

    // // Normalize
    // int block_size = blockDim.x;
    // for (int i = blockIdx.x * block_size + threadIdx.x; i < n;
    //     i += gridDim.x * block_size) {
    //     output[i] = input[i] / global_rms;
    // }
    __shared__ float sdata[256];  // blockDim.x 크기
    
    // ========== 1. 모든 스레드가 partial_sums를 나눠서 로드 ==========
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        local_sum += partial_sums[i];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();  // 이제 workload가 균등함
    
    // ========== 2. Shared memory에서 parallel reduction ==========
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // ========== 3. 마지막 warp는 warp-level reduction (더 빠름) ==========
    if (threadIdx.x < 32) {
        volatile float* vsdata = sdata;
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 32];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 16];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 8];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 4];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 2];
        vsdata[threadIdx.x] += vsdata[threadIdx.x + 1];
    }
    
    __shared__ float global_rms;
    if (threadIdx.x == 0) {
        global_rms = sqrtf(sdata[0] / n + EPSILON);
    }
    __syncthreads();
    
    // ========== 4. Normalize ==========
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        output[i] = input[i] / global_rms;
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

     size_t shared_mem = static_cast<size_t>(block_size / 32) * sizeof(float);

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
 
     return milliseconds / static_cast<float>(num_iterations);
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
     float peak_gb_s = 2.0f * static_cast<float>(memClockRate) * (static_cast<float>(prop.memoryBusWidth) / 8.0f) / 1e6f;
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
     return (static_cast<float>(bytes) / time_s) / 1e9f;
 }
