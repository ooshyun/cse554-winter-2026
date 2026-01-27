"""
CSE 554 Assignment 1 - Section 1: SiLU Implementation in PyTorch
Implementation of Sigmoid Linear Unit (SiLU) using basic PyTorch tensor operations
without using nn.SiLU or nn.Sigmoid
"""

import torch
import json
from torch.profiler import profile, ProfilerActivity


def silu_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Compute SiLU activation: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Args:
        x: Input tensor of shape (8192, 8192)

    Returns:
        Output tensor with SiLU activation applied
    """
    # Implement sigmoid manually: sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid = 1.0 / (1.0 + torch.exp(-x))

    # SiLU(x) = x * sigmoid(x)
    return x * sigmoid


def profile_silu():
    """
    Profile the SiLU implementation using Torch Profiler and measure bandwidth
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create input matrix (8192, 8192)
    shape = (8192, 8192)
    x = torch.randn(shape, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = silu_torch(x)

    torch.cuda.synchronize()

    # Profile using Torch Profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(100):
            result = silu_torch(x)
        torch.cuda.synchronize()

    # Export profiler results
    prof.export_chrome_trace("silu/profiling_results/torch_silu.json")
    print("\nTorch Profiler results saved to: silu/profiling_results/torch_silu.json")

    # Print profiler summary
    print("\n" + "="*80)
    print("TORCH PROFILER SUMMARY")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Measure execution time for bandwidth calculation using CUDA Events
    num_iterations = 1000

    # Create CUDA events for accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    for _ in range(num_iterations):
        result = silu_torch(x)

    end_event.record()
    torch.cuda.synchronize()

    # Get elapsed time in milliseconds
    elapsed_time_ms = start_event.elapsed_time(end_event) / num_iterations

    # Calculate bandwidth
    # Memory accesses: 1 read (input) + 1 write (output) = 2 accesses
    # Each element is 4 bytes (float32)
    num_elements = shape[0] * shape[1]
    bytes_per_element = 4
    min_memory_accesses = 2 * num_elements * bytes_per_element

    # Use decimal GB/s (1 GB = 10^9 bytes) to match official GPU specifications
    elapsed_time_s = elapsed_time_ms / 1000.0
    bandwidth_gb_s = (min_memory_accesses / elapsed_time_s) / (1024**3)

    # Peak bandwidth from datasheet (RTX 4070 Ti SUPER)
    peak_bandwidth_datasheet = 672.0  # GB/s
    bandwidth_percentage = (bandwidth_gb_s / peak_bandwidth_datasheet) * 100.0

    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"Matrix shape: {shape}")
    print(f"Average execution time: {elapsed_time_ms:.4f} ms")
    print(f"Minimum memory accesses: {min_memory_accesses / (1024**2):.2f} MB")
    print(f"Bandwidth utilization: {bandwidth_gb_s:.2f} GB/s ({bandwidth_percentage:.1f}% of peak)")
    print(f"Peak bandwidth (datasheet): {peak_bandwidth_datasheet:.2f} GB/s")
    print("="*80)

    # Verify correctness
    x_test = torch.tensor([[1.0, -1.0], [0.0, 2.0]], device=device)
    result_test = silu_torch(x_test)
    print("\nCorrectness check:")
    print(f"Input: {x_test.cpu().numpy()}")
    print(f"Output: {result_test.cpu().numpy()}")

    # Compare with PyTorch's built-in SiLU for validation
    expected = torch.nn.functional.silu(x_test)
    print(f"Expected (nn.functional.silu): {expected.cpu().numpy()}")
    print(f"Max difference: {torch.max(torch.abs(result_test - expected)).item():.2e}")

    return {
        "execution_time_ms": elapsed_time_ms,
        "bandwidth_gb_s": bandwidth_gb_s,
        "bandwidth_percentage": bandwidth_percentage,
        "peak_bandwidth_datasheet": peak_bandwidth_datasheet,
        "min_memory_accesses_mb": min_memory_accesses / (1024**2)
    }


def analyze_kernels():
    """
    Analyze what kernels are being launched
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (8192, 8192)
    x = torch.randn(shape, device=device, dtype=torch.float32)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _ = silu_torch(x)
        torch.cuda.synchronize()

    print("\n" + "="*80)
    print("KERNEL ANALYSIS")
    print("="*80)
    print("Kernels launched during SiLU execution:")

    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            # Use device_time instead of deprecated cuda_time
            if hasattr(evt, 'device_time'):
                print(f"  - {evt.key}: {evt.device_time / 1000:.4f} ms")
            elif hasattr(evt, 'self_cuda_time_total'):
                print(f"  - {evt.key}: {evt.self_cuda_time_total / 1000:.4f} ms")
            elif hasattr(evt, 'cuda_time'):
                # Fallback for older PyTorch versions
                print(f"  - {evt.key}: {evt.cuda_time / 1000:.4f} ms")

    print("="*80)


if __name__ == "__main__":
    print("CSE 554 Assignment 1 - Section 1: SiLU (PyTorch Implementation)")
    print("="*80)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU.")
    else:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # Run profiling
    metrics = profile_silu()

    # Analyze kernels
    analyze_kernels()

    print("\n✓ Profiling complete!")
    print("  - Torch Profiler output: profiling_results/torch_silu.json")
    print("  - Run Nsight Systems: nsys profile -o profiling_results/torch_silu python3 srcs/python/silu/silu_torch.py")
