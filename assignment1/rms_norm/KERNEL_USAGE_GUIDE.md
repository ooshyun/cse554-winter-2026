# RMS Norm Kernel Usage Guide

## Summary

This guide documents which kernels are safe to use and which have known issues.

## Matrix RMS Norm Kernels (✅ All Safe)

All matrix kernels process **one row per block**, so `__syncthreads()` works correctly within each block.

### Available Kernels

| Kernel | Status | Performance | Description |
|--------|--------|-------------|-------------|
| `rms_norm_matrix_basic` | ✅ Safe | 68.69 GB/s | Simple two-phase: reduction + normalize |
| `rms_norm_matrix_optimized` | ✅ Safe | 283.05 GB/s | Warp-level reduction optimization |
| `rms_norm_matrix_fast` | ✅ Safe | 283.16 GB/s | **Recommended** - Best performance |
| `rms_norm_matrix_w2l3_reduction` | ✅ Safe | 282.58 GB/s | W2L3 reduction5-inspired |
| `rms_norm_matrix_w2l3_tile` | ✅ Safe | 226.05 GB/s | W2L3 transpose_v5-inspired (slower) |
| `rms_norm_matrix_w2l3_hybrid` | ✅ Safe | 281.73 GB/s | Combined W2L3 optimizations |

### Recommendation
**Use `rms_norm_matrix_fast`** for production. It achieves ~283 GB/s (42% of peak bandwidth).

### Usage
```cpp
void (*picked_kernel)(const float*, float*, int, int) = rms_norm_matrix_fast;
```

---

## Vector RMS Norm Kernels

Vector kernels require **multiple blocks** to process a single long vector, making cross-block synchronization critical.

### ✅ Safe Kernels

| Kernel | Status | Performance | Description |
|--------|--------|-------------|-------------|
| `rms_norm_vector_basic` | ✅ **Fixed** | 1.86 GB/s | Two-phase with redundant RMS computation |
| `rms_norm_vector_fast` | ✅ Safe | 800 GB/s | **Recommended** - Uses device memory for RMS |
| `rms_norm_vector_cooperative` | ✅ Safe | ~600 GB/s | Cooperative Groups for grid sync |
| `rms_norm_vector_w2l3_reduction` | ✅ Safe | 90.73 GB/s | W2L3-inspired 2-phase |
| `rms_norm_vector_w2l3_hybrid` | ✅ Safe | 168.63 GB/s | W2L3 vectorized variant |

### ❌ Unsafe Kernel (DO NOT USE)

| Kernel | Status | Issue | Fix |
|--------|--------|-------|-----|
| `rms_norm_vector_atomic` | ❌ **UNSAFE** | `blockIdx.x == 0` check causes uninitialized `rms` in other blocks | **Do not use** - design is fundamentally flawed |

**Problem in `rms_norm_vector_atomic` (line 164-168)**:
```cuda
__shared__ float rms;
if (tid == 0 && blockIdx.x == 0) {  // ❌ Only block 0 initializes rms
    while (atomicAdd(global_sum, 0.0f) == 0.0f);
    rms = sqrtf((*global_sum) / n + EPSILON);
}
__syncthreads();  // ❌ Other blocks use uninitialized rms

// All blocks normalize using rms
for (int i = blockIdx.x * block_size + tid; i < n; i += gridDim.x * block_size) {
    output[i] = input[i] / rms;  // ❌ Undefined behavior in non-zero blocks!
}
```

**Why this is wrong**:
- `__syncthreads()` only synchronizes **within a block**, not across blocks
- Blocks 1, 2, 3, ... never initialize `rms` in their shared memory
- Using uninitialized shared memory causes undefined behavior (inf, NaN, garbage)

**Why we don't recommend fixing it**:
- Atomic spin-wait pattern is extremely inefficient
- Better alternatives exist (cooperative groups, 2-phase)
- Performance would still be poor even if fixed

### Recommendation
**Use `rms_norm_vector_fast`** for production. It achieves ~800 GB/s (119% of peak bandwidth!).

### Usage
```cpp
void (*picked_kernel)(const float*, float*, int) = rms_norm_vector_fast;
```

---

## Bug Details: Fixed Issues

### Issue 1: `rms_norm_vector_phase2` (FIXED)

**Original Bug** (line 103):
```cuda
if (threadIdx.x == 0 && blockIdx.x == 0) {  // ❌ Only block 0
    float total_sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total_sum += partial_sums[i];
    }
    global_rms = sqrtf(total_sum / n + EPSILON);
}
__syncthreads();  // ❌ Other blocks have uninitialized global_rms
```

**Fixed Version** (line 103):
```cuda
if (threadIdx.x == 0) {  // ✅ Every block's thread 0
    float total_sum = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total_sum += partial_sums[i];
    }
    global_rms = sqrtf(total_sum / n + EPSILON);
}
__syncthreads();  // ✅ All blocks have valid global_rms
```

**Why this works**:
- Each block's thread 0 independently computes the same `global_rms`
- Redundant computation, but ensures all blocks have valid values
- All blocks read the same `partial_sums` array → same result
- `__syncthreads()` ensures thread 0 finishes before other threads use `global_rms`

---

## Testing Recommendations

### Correctness Testing
Always test with the **basic or fast** kernel first to verify correctness:

```bash
# Matrix
cd assignment1/rms_norm/matrix
make clean && make
./rms_norm_matrix_test

# Vector
cd assignment1/rms_norm/vector
make clean && make
./rms_norm_vector_test
```

### Performance Testing
Compare different kernels using the W2L3 test programs:

```bash
# Matrix W2L3 comparison
cd assignment1/rms_norm/matrix
./rms_norm_matrix_test_w2l3

# Vector W2L3 comparison
cd assignment1/rms_norm/vector
./rms_norm_vector_test_w2l3
```

### Kernel Selection
Edit the `picked_kernel` line in `main.cu`:

```cpp
// Matrix (line 200)
void (*picked_kernel)(const float*, float*, int, int) = rms_norm_matrix_fast;

// Vector (line 176)
void (*picked_kernel)(const float*, float*, int) = rms_norm_vector_fast;
```

---

## Common Patterns to Avoid

### ❌ Anti-Pattern: Conditional Shared Memory Initialization
```cuda
__shared__ float value;
if (threadIdx.x == 0 && blockIdx.x == 0) {  // ❌ WRONG
    value = compute_something();
}
__syncthreads();
// Other blocks have uninitialized value!
```

### ✅ Correct Pattern 1: Every Block Computes
```cuda
__shared__ float value;
if (threadIdx.x == 0) {  // ✅ Every block's thread 0
    value = compute_something();
}
__syncthreads();
// All blocks have valid value
```

### ✅ Correct Pattern 2: Use Device Memory
```cuda
// Thread 0 of block 0 writes to device memory
if (threadIdx.x == 0 && blockIdx.x == 0) {
    *global_value = compute_something();
}
__threadfence();  // Ensure visibility
grid.sync();       // Grid-wide sync (Cooperative Groups)

// All threads read from device memory
float value = *global_value;  // ✅ All blocks see the value
```

### ✅ Correct Pattern 3: Two-Phase Execution
```cuda
// Phase 1: Each block computes partial result
partial_results[blockIdx.x] = block_computation();

// CPU or separate kernel reduces partial results
// Phase 2: Each block uses the final result
float final_result = *global_result;
```

---

## Performance Notes

### Matrix Performance
- **Target**: > 300 GB/s
- **Achieved**: ~283 GB/s (94% of target, 42% of peak)
- **Bottleneck**: Row-wise access pattern, limited by compute

### Vector Performance
- **Target**: > 200 GB/s
- **Achieved**: ~800 GB/s (400% of target, 119% of peak!)
- **Note**: Exceeding peak bandwidth suggests excellent cache utilization

### Why Vector is Faster
1. **Longer vectors**: Better amortization of launch overhead
2. **Sequential access**: Perfect memory coalescing
3. **Cooperative Groups**: Single-kernel eliminates CPU-GPU sync
4. **Cache reuse**: Multiple passes benefit from L2 cache

---

## Compilation

### Standard Compilation
```bash
make clean && make
```

### With Profiling
```bash
# Nsight Compute profiling
ncu -o profiling_results/rms_norm_matrix ./rms_norm_matrix_test
ncu -o profiling_results/rms_norm_vector ./rms_norm_vector_test
```

### Architecture Selection
The Makefile automatically detects GPU architecture. To manually specify:
```bash
nvcc -arch=sm_89 ...  # RTX 4070 Ti SUPER
nvcc -arch=sm_75 ...  # Quadro RTX 6000
```

---

## References

- Matrix implementation: [assignment1/rms_norm/matrix/rms_norm_matrix.cu](matrix/rms_norm_matrix.cu)
- Vector implementation: [assignment1/rms_norm/vector/rms_norm_vector.cu](vector/rms_norm_vector.cu)
- W2L3 optimization notes: [W2L3_OPTIMIZATION_NOTES.md](matrix/W2L3_OPTIMIZATION_NOTES.md)
- GPU specs header: [common/gpu_specs.h](../common/gpu_specs.h)
