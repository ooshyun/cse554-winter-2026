"""
CSE 554 Assignment 1 - Section 1: SiLU Triton Kernel Testing
Test suite for Triton SiLU implementation
"""

import sys

import torch
from silu_triton_kernel import silu_triton, benchmark_triton_silu


def test_silu_correctness():
    """Test SiLU kernel correctness against PyTorch reference"""
    device = torch.device("cuda")

    # Test various input patterns
    test_cases = [
        torch.randn(100, device=device),
        torch.randn(8192, 8192, device=device),
        torch.tensor([0.0, 1.0, -1.0, 10.0, -10.0], device=device),
        torch.linspace(-10, 10, 1000, device=device),
    ]

    for i, x in enumerate(test_cases):
        result = silu_triton(x)
        expected = torch.nn.functional.silu(x)

        max_diff = torch.max(torch.abs(result - expected)).item()
        print(f"Test case {i+1}: max difference = {max_diff:.2e}")

        assert max_diff < 1e-5, f"Test case {i+1} failed: max difference = {max_diff}"

    print("✓ All correctness tests passed!")


def test_silu_shapes():
    """Test SiLU kernel with various tensor shapes"""
    device = torch.device("cuda")

    shapes = [
        (100,),
        (1000,),
        (1024, 1024),
        (8192, 8192),
        (100, 200, 300),
    ]

    for shape in shapes:
        x = torch.randn(shape, device=device)
        result = silu_triton(x)
        expected = torch.nn.functional.silu(x)

        assert result.shape == expected.shape, f"Shape mismatch for {shape}"
        max_diff = torch.max(torch.abs(result - expected)).item()
        assert max_diff < 1e-5, f"Failed for shape {shape}: max difference = {max_diff}"

        print(f"✓ Shape {shape}: passed")

    print("✓ All shape tests passed!")


def test_silu_edge_cases():
    """Test SiLU kernel with edge cases"""
    device = torch.device("cuda")

    # Test edge cases
    edge_cases = {
        "zeros": torch.zeros(100, device=device),
        "ones": torch.ones(100, device=device),
        "mixed": torch.tensor([float('inf'), float('-inf'), 0.0, 1.0, -1.0], device=device),
    }

    for name, x in edge_cases.items():
        result = silu_triton(x)
        expected = torch.nn.functional.silu(x)

        # For inf values, check if both are inf
        if name == "mixed":
            # Just check finite values
            finite_mask = torch.isfinite(expected)
            if finite_mask.any():
                max_diff = torch.max(torch.abs(result[finite_mask] - expected[finite_mask])).item()
                print(f"✓ Edge case '{name}': max difference (finite) = {max_diff:.2e}")
        else:
            max_diff = torch.max(torch.abs(result - expected)).item()
            print(f"✓ Edge case '{name}': max difference = {max_diff:.2e}")


def test_silu_block_sizes():
    """Test SiLU kernel with different block sizes"""
    device = torch.device("cuda")
    x = torch.randn(8192, 8192, device=device)
    expected = torch.nn.functional.silu(x)

    block_sizes = [128, 256, 512, 1024]

    for block_size in block_sizes:
        result = silu_triton(x, block_size=block_size)
        max_diff = torch.max(torch.abs(result - expected)).item()

        print(f"Block size {block_size}: max difference = {max_diff:.2e}")
        assert max_diff < 1e-5, f"Failed for block size {block_size}"

    print("✓ All block size tests passed!")


def test_silu_performance():
    """Benchmark SiLU kernel performance"""
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARKING")
    print("="*80)

    shapes = [
        (8192, 8192),
    ]

    for shape in shapes:
        print(f"\nShape: {shape}")
        metrics = benchmark_triton_silu(shape=shape, num_iterations=100)
        print(f"  Execution time: {metrics['execution_time_ms']:.4f} ms")
        print(f"  Bandwidth: {metrics['bandwidth_gb_s']:.2f} GB/s")
        print(f"  Bandwidth percentage: {metrics['bandwidth_percentage']:.1f}%")


if __name__ == "__main__":
    print("CSE 554 Assignment 1 - SiLU Triton Kernel Test Suite")
    print("="*80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Tests require CUDA.")
        exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")

    # Run all tests
    try:
        test_silu_correctness()
        print()
        test_silu_shapes()
        print()
        test_silu_edge_cases()
        print()
        test_silu_block_sizes()
        print()
        test_silu_performance()

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        exit(1)
