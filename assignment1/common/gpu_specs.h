/**
 * GPU Specifications Constants
 * Multi-GPU Support: RTX 4070 Ti SUPER and Quadro RTX 6000
 *
 * Sources:
 * - NVIDIA Ada Architecture Whitepaper v2.1
 * - NVIDIA Quadro RTX 6000 Datasheet
 */

 #ifndef GPU_SPECS_H
 #define GPU_SPECS_H
 
 // Auto-detect GPU at runtime
 #ifdef __CUDA_ARCH__
     // Device code - compile-time known
     #define GPU_DETECTED_AT_COMPILE_TIME 1
 #else
     // Host code - runtime detection needed
     #define GPU_DETECTED_AT_COMPILE_TIME 0
 #endif
 
 /**
  * RTX 4070 Ti SUPER Specifications (Ada Lovelace Architecture)
  * Compute Capability: 8.9
  * Memory: 16 GB GDDR6X
  */
 #define RTX_4070_TI_SUPER_BANDWIDTH 672.0f   // GB/s (256-bit bus, 21 Gbps)
 #define RTX_4070_TI_SUPER_COMPUTE_CAP_MAJOR 8
 #define RTX_4070_TI_SUPER_COMPUTE_CAP_MINOR 9
 #define RTX_4070_TI_SUPER_SM_COUNT 66
 #define RTX_4070_TI_SUPER_TFLOPS 44.1f       // FP32 TFLOPS
 
 /**
  * Quadro RTX 6000 Specifications (Turing Architecture)
  * Compute Capability: 7.5
  * Memory: 24 GB GDDR6
  */
 #define QUADRO_RTX_6000_BANDWIDTH 672.0f     // GB/s (384-bit bus, 14 Gbps)
 #define QUADRO_RTX_6000_COMPUTE_CAP_MAJOR 7
 #define QUADRO_RTX_6000_COMPUTE_CAP_MINOR 5
 #define QUADRO_RTX_6000_SM_COUNT 72
 #define QUADRO_RTX_6000_TFLOPS 16.3f         // FP32 TFLOPS
 
 /**
  * Default values (RTX 4070 Ti SUPER for backward compatibility)
  */
 #define GPU_PEAK_BANDWIDTH_DATASHEET RTX_4070_TI_SUPER_BANDWIDTH
 #define GPU_PEAK_TFLOPS_SPARSITY 780.0f      // FP8 with sparsity (Ada only)
 #define GPU_MEMORY_BUS_WIDTH 256             // bits (RTX 4070 Ti SUPER)
 #define GPU_MEMORY_CLOCK_GBPS 21             // Gbps (RTX 4070 Ti SUPER)
 
 /**
  * Runtime GPU detection function (to be used in host code)
  * Returns peak bandwidth based on detected GPU
  */
 static inline float get_gpu_peak_bandwidth() {
     cudaDeviceProp prop;
     cudaGetDeviceProperties(&prop, 0);
 
     // Check compute capability to identify GPU
     if (prop.major == 8 && prop.minor == 9) {
         // RTX 4070 Ti SUPER (Ada Lovelace)
         return RTX_4070_TI_SUPER_BANDWIDTH;
     } else if (prop.major == 7 && prop.minor == 5) {
         // Quadro RTX 6000 (Turing)
         return QUADRO_RTX_6000_BANDWIDTH;
     } else {
         // Fallback: calculate from device properties
         return 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
     }
 }
 
 /**
  * Get SM count for current GPU
  */
 static inline int get_gpu_sm_count() {
     cudaDeviceProp prop;
     cudaGetDeviceProperties(&prop, 0);
     return prop.multiProcessorCount;
 }
 
 /**
  * Get optimal block size multiplier for current GPU
  */
 static inline int get_optimal_blocks_per_sm() {
     cudaDeviceProp prop;
     cudaGetDeviceProperties(&prop, 0);
 
     if (prop.major == 8 && prop.minor == 9) {
         // Ada Lovelace: 4 blocks per SM
         return 4;
     } else if (prop.major == 7 && prop.minor == 5) {
         // Turing: 4 blocks per SM
         return 4;
     } else {
         return 4;  // Conservative default
     }
 }
 
 #endif // GPU_SPECS_H
 