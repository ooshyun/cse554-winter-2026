# W2L3 Optimization Techniques Applied to RMS Norm Matrix

## Overview

이 문서는 W2L3 수업에서 배운 `reduction5.cu`와 `transpose_v5.cu`의 최적화 기법을 RMS Norm Matrix 커널에 적용한 결과를 설명합니다.

## Implemented Kernels

### 1. W2L3_REDUCTION Kernel
**Inspiration**: `reduction5.cu` (W2L3/reduction5.cu)

**Key Optimizations**:
- **Multiple Elements Per Thread**: 각 스레드가 32개의 원소를 처리 (ELEMENTS_PER_THREAD = 32)
- **Strided Access Pattern**: `i = tid; i += blockDim.x` 패턴으로 coalesced memory access
- **Sequential Addressing**: Shared memory reduction에서 bank conflict 최소화
- **Loop Unrolling**: `#pragma unroll 8`로 루프 오버헤드 감소

**Code Structure** (Lines 200-265 in rms_norm_matrix.cu):
```cuda
#define ELEMENTS_PER_THREAD 32
#define BLOCK_SIZE_W2L3 256

__global__ void rms_norm_kernel_w2l3_reduction(...) {
    __shared__ float sdata[BLOCK_SIZE_W2L3];

    // Phase 1: Each thread processes multiple elements
    float local_sum = 0.0f;
    int i = tid;
    #pragma unroll 8
    for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
        if (i < cols) {
            float val = row_input[i];
            local_sum += val * val;
        }
        i += blockDim.x;
    }

    // Phase 2: Sequential addressing reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Phase 3: Normalize
    ...
}
```

**Performance**: 282.58 GB/s (42.0% of peak, 4.11x speedup vs basic)

### 2. W2L3_TILE Kernel
**Inspiration**: `transpose_v5.cu` (W2L3/transpose_v5.cu)

**Key Optimizations**:
- **Large Tile Size**: 128 elements per tile (TILE_SIZE = 128)
- **Bank Conflict Avoidance**: Padding (TILE_SIZE_PAD = 129) 사용
- **Tile-Based Processing**: 8192 columns을 128 element tiles로 분할 처리

**Code Structure** (Lines 268-336 in rms_norm_matrix.cu):
```cuda
#define TILE_SIZE 128
#define TILE_SIZE_PAD 129  // +1 for bank conflict avoidance

__global__ void rms_norm_kernel_w2l3_tile(...) {
    __shared__ float sdata[TILE_SIZE];

    // Phase 1: Process tiles
    float sum_sq = 0.0f;
    for (int base = 0; base < cols; base += TILE_SIZE) {
        // Load tile with padding
        if (base + tid < cols) {
            float val = row_input[base + tid];
            tile[tid] = val;
            sum_sq += val * val;
        }
        __syncthreads();
    }

    // Phase 2: Reduction
    ...

    // Phase 3: Normalize in tiles
    for (int base = 0; base < cols; base += TILE_SIZE) {
        if (base + tid < cols) {
            row_output[base + tid] = row_input[base + tid] / rms;
        }
    }
}
```

**Performance**: 226.05 GB/s (33.6% of peak, 3.29x speedup vs basic)

**Note**: TILE 커널이 다른 커널보다 느린 이유:
- Tile loop overhead가 8192/128 = 64회 발생
- 각 tile마다 __syncthreads() 호출 필요
- RMS Norm의 경우 한 번의 전체 row scan이 더 효율적

### 3. W2L3_HYBRID Kernel
**Inspiration**: reduction5 + transpose_v5 combined

**Key Optimizations**:
- **Vectorized Memory Access**: float4를 사용한 4-way vectorization
- **Multiple Elements + Vectorization**: 16 iterations × 4 floats = 64 elements per thread
- **Bank Conflict Avoidance**: sdata[BLOCK_SIZE_HYBRID + 1] padding
- **Unrolled Reduction**: Power-of-2 unrolling for reduction phases
- **Warp-Level Optimization**: Volatile pointer로 warp 내 동기화 제거

**Code Structure** (Lines 339-441 in rms_norm_matrix.cu):
```cuda
#define BLOCK_SIZE_HYBRID 256
#define ELEMENTS_PER_THREAD_VEC 16

__global__ void rms_norm_kernel_w2l3_hybrid(...) {
    __shared__ float sdata[BLOCK_SIZE_HYBRID + 1];  // +1 for bank conflict avoidance

    // Phase 1: Vectorized reduction
    int vec_cols = cols / 4;
    #pragma unroll 4
    for (int j = 0; j < ELEMENTS_PER_THREAD_VEC; j++) {
        if (i < vec_cols) {
            float4 vals = reinterpret_cast<const float4*>(row_input)[i];
            local_sum += vals.x * vals.x;
            local_sum += vals.y * vals.y;
            local_sum += vals.z * vals.z;
            local_sum += vals.w * vals.w;
        }
        i += blockDim.x;
    }

    // Phase 2: Unrolled reduction
    if (BLOCK_SIZE_HYBRID >= 256 && tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (BLOCK_SIZE_HYBRID >= 128 && tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    // Warp-level reduction (no sync)
    if (tid < 32) {
        volatile float* vsdata = sdata;
        if (BLOCK_SIZE_HYBRID >= 64) vsdata[tid] += vsdata[tid + 32];
        if (BLOCK_SIZE_HYBRID >= 32) vsdata[tid] += vsdata[tid + 16];
        ...
    }

    // Phase 3: Vectorized normalization
    ...
}
```

**Performance**: 281.73 GB/s (41.9% of peak, 4.10x speedup vs basic)

## Performance Comparison

### Benchmark Results (RTX 4070 Ti SUPER, 8192×8192 matrix, 100 iterations)

| Kernel | Time (ms) | Bandwidth (GB/s) | % of Peak | Speedup vs Basic |
|--------|-----------|------------------|-----------|------------------|
| FAST (original) | 1.8960 | 283.16 | 42.1% | 4.12x |
| OPTIMIZED | 1.8968 | 283.05 | 42.1% | 4.12x |
| **W2L3_REDUCTION** | 1.8999 | 282.58 | 42.0% | 4.11x |
| **W2L3_HYBRID** | 1.9057 | 281.73 | 41.9% | 4.10x |
| **W2L3_TILE** | 2.3750 | 226.05 | 33.6% | 3.29x |
| BASIC | 7.8154 | 68.69 | 10.2% | 1.00x |

### Key Findings

1. **W2L3_REDUCTION이 가장 우수**:
   - reduction5.cu의 핵심 기법 (multiple elements per thread, sequential addressing)이 RMS Norm에 매우 효과적
   - 원래 FAST 커널과 거의 동등한 성능 (283.16 vs 282.58 GB/s)

2. **W2L3_HYBRID는 2위**:
   - Float4 vectorization과 unrolled reduction 결합
   - Reduction 커널 대비 약간 느린 이유: float4 alignment overhead

3. **W2L3_TILE이 상대적으로 느린 이유**:
   - Transpose 작업은 2D → 2D 변환에 최적화
   - RMS Norm은 1D row 처리 문제 → tile overhead가 이점보다 큼
   - 64번의 tile loop + syncthreads overhead

## Optimization Lessons Learned

### From reduction5.cu ✅
- **Multiple elements per thread**: Memory latency hiding에 매우 효과적
- **Sequential addressing**: Bank conflict 완전 제거
- **Stride pattern**: Coalesced memory access 보장

### From transpose_v5.cu ⚠️
- **Large tiles**: 2D transpose에는 효과적, 하지만 1D row 처리에는 overhead
- **Bank conflict padding**: 항상 유익하지만 RMS Norm에서는 marginal gain
- **Shared memory tiling**: Access pattern이 단순한 경우 불필요한 복잡도

### Best Practices
1. **문제 특성 이해**: 2D 기법을 1D 문제에 적용할 때 주의
2. **Simple First**: W2L3_REDUCTION처럼 단순하면서 효과적인 방법 우선
3. **Measure Everything**: 직관이 아닌 실제 측정으로 판단
4. **Combine Wisely**: 여러 최적화 기법을 무분별하게 결합하지 말것

## Code Files

- **rms_norm_matrix.cu**: 커널 구현 (Lines 197-441: W2L3 kernels)
- **test_w2l3.cu**: 벤치마크 테스트 프로그램
- **main.cu**: 정확성 및 성능 테스트

## How to Run

### Compile and Run
```bash
cd assignment1/rms_norm/matrix
make clean && make
./rms_norm_matrix_test  # Basic correctness test

# Run W2L3-specific benchmark
nvcc -O3 -arch=sm_89 -std=c++17 --use_fast_math test_w2l3.cu rms_norm_matrix.cu -o test_w2l3
./test_w2l3
```

### Expected Output
```
W2L3 Optimization Test
========================================

Benchmarking kernels on 8192 x 8192 matrix (100 iterations)...

1. BASIC kernel:
   Time: 7.8154 ms | Bandwidth: 68.69 GB/s (10.2% of peak)

...

Best W2L3 kernel: W2L3_REDUCTION (282.58 GB/s)
========================================
```

## Profiling with Nsight Compute

```bash
# Profile W2L3_REDUCTION kernel
ncu -o profiling_results/w2l3_reduction --set full ./test_w2l3

# Compare all kernels
ncu -o profiling_results/all_kernels --set full --kernel-id ::rms_norm_kernel_w2l3 ./test_w2l3
```

## Conclusion

W2L3의 reduction5.cu에서 배운 최적화 기법이 RMS Norm Matrix 문제에 매우 효과적으로 적용되었습니다:

- ✅ **W2L3_REDUCTION**: 282.58 GB/s (42.0% of peak) - **Best W2L3 kernel**
- ✅ **W2L3_HYBRID**: 281.73 GB/s (41.9% of peak) - 복합 최적화
- ⚠️ **W2L3_TILE**: 226.05 GB/s (33.6% of peak) - 문제 특성 불일치

핵심 교훈: **문제의 특성을 이해하고 적절한 최적화 기법을 선택하는 것이 중요**합니다.
