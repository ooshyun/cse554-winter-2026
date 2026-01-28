#include <cuda_runtime.h>
#include <iostream>

#define BLOCKSIZE 256


// Kernel function to add two vectors
__global__ void add(int *a, int *b, int *c, size_t num) {
    int block_start = blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    int index = block_start + thread_id;
    if (index < num) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    size_t num = 1000000000;
    float num_iterations = 100;

    int * host_a = new int[num];
    int * host_b = new int[num];
    int * host_c = new int[num];

    // Initialize host arrays
    for (int i = 0; i < num; i++) {
        host_a[i] = i;
        host_b[i] = i;
    }
    

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, num * sizeof(int));
    cudaMalloc((void**)&d_b, num * sizeof(int));
    cudaMalloc((void**)&d_c, num * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, host_a, num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, host_b, num * sizeof(int), cudaMemcpyHostToDevice);

    dim3 num_block((num + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 num_threads(BLOCKSIZE);
    /*
    The below code corresponds to the cudaLaunchKernel signature

    __host__​cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )

    num_block corresponds to gridDim: how many blocks you want to launch.
    num_threads corresponds to blockDim: how many threads per block you want.

    */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup: Run kernel a few times to stabilize GPU state (clock scaling, cache warmup)
    for (int i = 0; i < 10; i++) {
        add<<<num_block, num_threads>>>(d_a, d_b, d_c, num);
    }
    cudaDeviceSynchronize();

    float time_kernel = 0;

    // Now measure the actual kernel execution time
    // Record start event and ensure it's recorded before kernel launch
    // cudaEventRecord(start);
    // cudaEventSynchronize(start);  // Ensure start event is recorded

    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(start);
        add<<<num_block, num_threads>>>(d_a, d_b, d_c, num);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float lap_time;
        cudaEventElapsedTime(&lap_time, start, stop);
        time_kernel += lap_time;
        // add<<<num_block, num_threads>>>(d_a, d_b, d_c, num);
    }

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);  // Wait for stop event to be recorded
    // float time_kernel = 0;
    // cudaEventElapsedTime(&time_kernel, start, stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(host_c, d_c, num * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num; i++) {
        if (host_c[i] != host_a[i] + host_b[i]) {
            std::cerr << "Error at index " << i << ": " << host_c[i] << std::endl;
            break;
        }
    }


    float time_kernel_ms = time_kernel / num_iterations;  // Average over num_iterations iterations
    // Read 2 + Write 1 = 3 
    float bandwidth_kernel = 3 * num * sizeof(int) / (time_kernel_ms / 1000) / 1e9;
    float efficiency_kernel = bandwidth_kernel / 672.0f * 100.0f;

    std::cout << "Result: " << host_c[0] << std::endl;
    std::cout << "Kernel execution time: " << time_kernel_ms << " ms" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_kernel << " GB/s" << std::endl;
    std::cout << "Efficiency: " << efficiency_kernel << "%" << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] host_a;
    delete[] host_b;
    delete[] host_c;

    return 0;
}