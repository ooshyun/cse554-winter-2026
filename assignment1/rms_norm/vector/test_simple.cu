#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void test_rms_cooperative(const float* input, float* output,
                                      float* partial_sums, float* global_rms,
                                      int n) {
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Phase 1
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        sum += input[i] * input[i];
    }

    sum = warp_reduce_sum(sum);
    if (lane_id == 0) shared[warp_id] = sum;
    __syncthreads();

    if (tid < (blockDim.x / 32)) {
        sum = shared[tid];
        sum = warp_reduce_sum(sum);
        if (tid == 0) partial_sums[blockIdx.x] = sum;
    }

    grid.sync();

    if (tid == 0 && blockIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < gridDim.x; i++) total += partial_sums[i];
        *global_rms = sqrtf(total / n + 1e-6f);
    }

    grid.sync();

    float rms_val = *global_rms;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        output[i] = input[i] / rms_val;
    }
}

int main() {
    int n = 1000;
    float *d_in, *d_out, *d_partial, *d_rms;

    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rms, sizeof(float)));

    float h_in[1000];
    for (int i = 0; i < n; i++) h_in[i] = (float)i;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    printf("Launching cooperative kernel...\n");

    void* args[] = {(void*)&d_in, (void*)&d_out, (void*)&d_partial, (void*)&d_rms, (void*)&n};
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)test_rms_cooperative,
                                           4, 256, args, 8 * sizeof(float), 0));

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Kernel completed!\n");

    float h_out[1000];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    printf("First result: %f\n", h_out[0]);

    return 0;
}
