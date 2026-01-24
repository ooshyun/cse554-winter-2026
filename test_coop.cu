#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

__global__ void test_coop_kernel() {
    cg::grid_group grid = cg::this_grid();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Grid size: %d\n", grid.size());
    }
    grid.sync();
}

int main() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("Device: %s\n", prop.name);
    printf("Supports cooperative launch: %d\n", prop.cooperativeLaunch);
    
    if (!prop.cooperativeLaunch) {
        printf("ERROR: Device does not support cooperative launch!\n");
        return 1;
    }
    
    void* args[] = {};
    cudaLaunchCooperativeKernel((void*)test_coop_kernel, 2, 256, args, 0, 0);
    cudaDeviceSynchronize();
    
    printf("Cooperative kernel succeeded!\n");
    return 0;
}
