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
  * Optimized Phase 1: Compute partial sums AND copy input to output
  * This avoids reading input twice (once here, once in phase2)
  */
 __global__ void rms_norm_vector_phase1(const float* input, float* partial_sums,
                                         float* output, int n) {
     extern __shared__ float shared[];

     int tid = threadIdx.x;
     int block_size = blockDim.x;
     int warp_id = tid / 32;
     int lane_id = tid % 32;

     // Each thread computes partial sum with grid-stride loop
     // AND copies input to output for phase2
     float sum = 0.0f;
     for (int i = blockIdx.x * block_size + tid; i < n; i += gridDim.x * block_size) {
         float val = input[i];
         output[i] = val;  // Copy to output (will be normalized in-place by phase2)
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
  * Optimized Phase 2: Normalize in-place (output already contains input values)
  */
__global__ void rms_norm_vector_phase2(float* output,
                                        const float* partial_sums, int n,
                                        int num_blocks) {
    __shared__ float sdata[256];

    // ========== 1. Load partial_sums in parallel ==========
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        local_sum += partial_sums[i];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // ========== 2. Parallel reduction in shared memory ==========
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // ========== 3. Final warp-level reduction ==========
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

    // ========== 4. Normalize in-place (output already has input values) ==========
    float rms_val = global_rms;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        output[i] = output[i] / rms_val;
    }
}

/**
 * Single-kernel RMS Norm using cooperative groups grid synchronization
 * Combines both phases into one kernel using grid.sync()
 */
__global__ void rms_norm_vector_single_kernel(const float* input, float* output,
                                               float* partial_sums, int n) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = block_size / 32;

    // ========== Phase 1: Compute partial sums of squares ==========
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

    // Final reduction within block - first warp reduces warp sums
    float block_sum = 0.0f;
    if (tid < num_warps) {
        block_sum = shared[tid];
    }
    if (tid < 32) {
        block_sum = warp_reduce_sum(block_sum);
        if (tid == 0) {
            partial_sums[blockIdx.x] = block_sum;
        }
    }
    __syncthreads();  // Ensure partial_sums write is visible

    // ========== Grid-wide synchronization ==========
    grid.sync();

    // ========== Phase 2: Compute global RMS and normalize ==========
    // All threads participate in reducing partial_sums
    float local_sum = 0.0f;
    for (int i = tid; i < gridDim.x; i += block_size) {
        local_sum += partial_sums[i];
    }
    shared[tid] = local_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = block_size / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Final warp reduction
    float total_sum = 0.0f;
    if (tid < 32) {
        total_sum = shared[tid];
        if (block_size >= 64) total_sum += shared[tid + 32];
        total_sum = warp_reduce_sum(total_sum);
    }

    __shared__ float global_rms;
    if (tid == 0) {
        global_rms = sqrtf(total_sum / n + EPSILON);
    }
    __syncthreads();

    // Normalize output
    float rms_val = global_rms;
    for (int i = blockIdx.x * block_size + tid; i < n; i += gridDim.x * block_size) {
        output[i] = input[i] / rms_val;
    }
}

/**
 * Wrapper function for single-kernel version using cooperative launch
 * Uses static allocation to avoid repeated malloc/free overhead
 */
void rms_norm_vector_coop(const float* d_input, float* d_output, int n) {
    static float* d_partial_sums = nullptr;
    static int allocated_blocks = 0;
    static int cached_num_blocks = 0;

    int block_size = 256;

    // Only query occupancy once
    if (cached_num_blocks == 0) {
        int num_blocks_per_sm;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, rms_norm_vector_single_kernel, block_size,
            block_size * sizeof(float)));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        int max_blocks = num_blocks_per_sm * prop.multiProcessorCount;
        int cooperative_limit = min(max_blocks, 256);
        cached_num_blocks = cooperative_limit;
    }

    int num_blocks = min((n + block_size - 1) / block_size, cached_num_blocks);

    // Allocate/reallocate partial_sums only if needed
    if (d_partial_sums == nullptr || num_blocks > allocated_blocks) {
        if (d_partial_sums != nullptr) {
            CUDA_CHECK(cudaFree(d_partial_sums));
        }
        CUDA_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(float)));
        allocated_blocks = num_blocks;
    }

    size_t shared_mem = block_size * sizeof(float);

    // Cooperative launch arguments
    void* kernel_args[] = {
        (void*)&d_input,
        (void*)&d_output,
        (void*)&d_partial_sums,
        (void*)&n
    };

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)rms_norm_vector_single_kernel,
        dim3(num_blocks),
        dim3(block_size),
        kernel_args,
        shared_mem));

    CUDA_CHECK(cudaGetLastError());
}

 // Static buffers for kernel wrappers (avoid per-call malloc overhead)
 static float* g_basic_partial_sums = nullptr;
 static int g_basic_allocated_blocks = 0;

 /**
  * Cleanup function for static buffers
  * Call before program exit or when switching contexts
  */
 void rms_norm_vector_cleanup() {
    if (g_basic_partial_sums != nullptr) {
        cudaFree(g_basic_partial_sums);
        g_basic_partial_sums = nullptr;
        g_basic_allocated_blocks = 0;
    }
 }

 /**
  * Wrapper functions
  * Uses static allocation to avoid cudaMalloc/cudaFree overhead per call
  */
 void rms_norm_vector_basic(const float* d_input, float* d_output, int n) {
    int block_size = 256;
    int num_blocks = min((n + block_size - 1) / block_size, 1024);

    // Allocate only once (or reallocate if more blocks needed)
    if (g_basic_partial_sums == nullptr || num_blocks > g_basic_allocated_blocks) {
        if (g_basic_partial_sums != nullptr) {
            CUDA_CHECK(cudaFree(g_basic_partial_sums));
        }
        CUDA_CHECK(cudaMalloc(&g_basic_partial_sums, num_blocks * sizeof(float)));
        g_basic_allocated_blocks = num_blocks;
    }

    size_t shared_mem = static_cast<size_t>(block_size / 32) * sizeof(float);

    // Phase 1: Compute partial sums AND copy input to output
    rms_norm_vector_phase1<<<num_blocks, block_size, shared_mem>>>(
        d_input, g_basic_partial_sums, d_output, n);

    // Phase 2: Normalize in-place (output already contains input values)
    rms_norm_vector_phase2<<<num_blocks, block_size>>>(
        d_output, g_basic_partial_sums, n, num_blocks);
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
     // Effective memory: input read (n) + output write (n) = 2n
     // Note: Implementation reads input twice (phase1 + phase2), but we measure
     // effective bandwidth (algorithm's minimum required data movement)
     size_t bytes = 2 * (size_t)n * sizeof(float);
     float time_s = time_ms / 1000.0f;

     // Use decimal GB/s to match official specs (1 GB = 10^9 bytes)
     return (static_cast<float>(bytes) / time_s) / 1e9f;
 }
