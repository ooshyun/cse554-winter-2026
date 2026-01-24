#!/bin/bash
# CSE 554 Assignment 1 - Section 1: Run Python/Triton Tests
# This script runs PyTorch and Triton implementations for SiLU

set -e  # Exit on error

echo "========================================================================"
echo "CSE 554 Assignment 1 - Section 1: SiLU (Python/Triton)"
echo "========================================================================"
echo ""

# Create profiling results directory if it doesn't exist
mkdir -p profiling_results

# Check if CUDA is available
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
    echo "ERROR: CUDA is not available. Please check your PyTorch installation."
    exit 1
}

echo "✓ CUDA is available"
echo ""

# Run PyTorch implementation
echo "========================================================================"
echo "Running PyTorch SiLU Implementation..."
echo "========================================================================"
cd srcs/python/silu
uv run python silu_torch.py
cd ../../..
echo ""

# Run Triton implementation
echo "========================================================================"
echo "Running Triton SiLU Kernel..."
echo "========================================================================"
cd srcs/python/silu
uv run python silu_triton_kernel.py
cd ../../..
echo ""

# Run Triton tests
echo "========================================================================"
echo "Running Triton SiLU Tests..."
echo "========================================================================"
cd srcs/python/silu
uv run python silu_triton_test.py
cd ../../..
echo ""

# Profile with Nsight Systems (optional - commented out by default)
# Uncomment to run Nsight Systems profiling
# echo "========================================================================"
# echo "Profiling with Nsight Systems..."
# echo "========================================================================"
# nsys profile -o profiling_results/torch_silu python3 srcs/python/silu/silu_torch.py
# echo ""

echo "========================================================================"
echo "Section 1 (Python/Triton) Complete!"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - profiling_results/torch_silu.json (Torch Profiler)"
echo ""
echo "To profile with Nsight Systems, run:"
echo "  nsys profile -o profiling_results/torch_silu uv run python srcs/python/silu/silu_torch.py"
echo ""
