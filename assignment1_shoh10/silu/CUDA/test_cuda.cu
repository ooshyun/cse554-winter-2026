#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    printf("cudaGetDevice returned: %s\n", cudaGetErrorString(err));
    
    if (err == cudaSuccess) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, device);
        printf("cudaGetDeviceProperties returned: %s\n", cudaGetErrorString(err));
        
        if (err == cudaSuccess) {
            printf("GPU: %s\n", prop.name);
        }
    }
    
    return 0;
}
