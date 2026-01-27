import torch
import triton
import triton.language as tl


"""
CSE 554 Assignment 1 - Section 1: SiLU Implementation in Triton
Custom Triton kernel for SiLU activation function
"""

import torch
import triton
import triton.language as tl

@triton.jit
def silu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for SiLU with vectorized operations
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input with vectorization hint
    # eviction_policy="evict_last": eviction_policy="evict_last": 
    # Marks this data as low-priority for the L2 cache. Since it's unlikely to be reused, 
    # it will be evicted first when the cache is full, 
    # preventing "cache pollution" and keeping more important data available.
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_last")

    # Compute SiLU in one fused operation
    # Using the mathematical identity: x / (1 + exp(-x))
    output = x / (1.0 + tl.exp(-x))

    # Store with vectorization hint
    tl.store(output_ptr + offsets, output, mask=mask)


def silu_triton(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    Apply SiLU activation using optimized Triton kernel

    Args:
        x: Input tensor
        block_size: Block size for Triton kernel (default: 1024)

    Returns:
        Output tensor with SiLU activation applied
    """
    output = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    silu_kernel[grid](
        x,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
    )

    return output


def benchmark_triton_silu(shape=(8192, 8192), num_iterations=1000):
    """
    Benchmark the Triton SiLU kernel

    Args:
        shape: Input tensor shape
        num_iterations: Number of iterations for benchmarking

    Returns:
        Dictionary with timing and bandwidth metrics
    """
    device = torch.device("cuda")
    x = torch.randn(shape, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = silu_triton(x)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(num_iterations):
        result = silu_triton(x)

    end.record()
    torch.cuda.synchronize()
    
    elapsed_time_ms = start.elapsed_time(end) / num_iterations

    # Calculate bandwidth
    num_elements = shape[0] * shape[1]
    bytes_per_element = 4
    min_memory_accesses = 2 * num_elements * bytes_per_element  # 1 read + 1 write

    # Use decimal GB/s (1 GB = 10^9 bytes) to match official GPU specifications
    elapsed_time_s = elapsed_time_ms / 1000.0
    bandwidth_gb_s = (min_memory_accesses / elapsed_time_s) / (1024**3)

    # Peak bandwidth from datasheet (RTX 4070 Ti SUPER)
    peak_bandwidth_datasheet = 672.0  # GB/s
    bandwidth_percentage = (bandwidth_gb_s / peak_bandwidth_datasheet) * 100.0

    return {
            "execution_time_ms": elapsed_time_ms,
            "bandwidth_gb_s": bandwidth_gb_s,
            "bandwidth_percentage": bandwidth_percentage,
            "peak_bandwidth_datasheet": peak_bandwidth_datasheet,
            "min_memory_accesses_mb": min_memory_accesses / (1024**2)
        }


if __name__ == "__main__":
    print("CSE 554 Assignment 1 - Section 1: SiLU (Triton Kernel)")
    print("="*80)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Triton requires CUDA.")
        exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # Test correctness
    print("\nTesting correctness...")
    x_test = torch.tensor([[1.0, -1.0], [0.0, 2.0]], device="cuda")
    result = silu_triton(x_test)
    expected = torch.nn.functional.silu(x_test)

    print(f"Input: {x_test.cpu().numpy()}")
    print(f"Triton Output: {result.cpu().numpy()}")
    print(f"Expected: {expected.cpu().numpy()}")
    print(f"Max difference: {torch.max(torch.abs(result - expected)).item():.2e}")

    # Benchmark
    print("\nBenchmarking Triton kernel...")
    metrics = benchmark_triton_silu()

    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"Average execution time: {metrics['execution_time_ms']:.4f} ms")
    print(f"Minimum memory accesses: {metrics['min_memory_accesses_mb']:.2f} MB")
    print(f"Bandwidth utilization: {metrics['bandwidth_gb_s']:.2f} GB/s")
    print("="*80)

    print("\n✓ Triton kernel test complete!")
