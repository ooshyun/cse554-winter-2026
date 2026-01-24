#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    printf("Testing CUDA initialization...\n");
    
    // Try to reset device first
    cudaError_t err = cudaDeviceReset();
    printf("cudaDeviceReset: %s\n", cudaGetErrorString(err));
    
    // Get device count
    int count;
    err = cudaGetDeviceCount(&count);
    printf("cudaGetDeviceCount: %s, count=%d\n", cudaGetErrorString(err), count);
    
    if (err != cudaSuccess) {
        printf("Failed at GetDeviceCount\n");
        return 1;
    }
    
    // Set device
    err = cudaSetDevice(0);
    printf("cudaSetDevice(0): %s\n", cudaGetErrorString(err));
    
    if (err != cudaSuccess) {
        printf("Failed at SetDevice\n");
        return 1;
    }
    
    // Now try GetDevice
    int device;
    err = cudaGetDevice(&device);
    printf("cudaGetDevice: %s, device=%d\n", cudaGetErrorString(err), device);
    
    if (err == cudaSuccess) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, device);
        printf("cudaGetDeviceProperties: %s\n", cudaGetErrorString(err));
        
        if (err == cudaSuccess) {
            printf("GPU: %s\n", prop.name);
            printf("Compute: %d.%d\n", prop.major, prop.minor);
        }
    }
    
    return 0;
}
