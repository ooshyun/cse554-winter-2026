#!/bin/bash
# CSE 554 Assignment 1 - Section 1: Run CUDA Tests
# This script compiles and runs CUDA SiLU implementation

set -e  # Exit on error

echo "========================================================================"
echo "CSE 554 Assignment 1 - Section 1: SiLU (CUDA)"
echo "========================================================================"
echo ""

# Create profiling results directory
mkdir -p profiling_results

# Navigate to CUDA SiLU directory
cd srcs/cuda/silu

echo "Compiling CUDA SiLU kernel..."
make clean
make
echo "✓ Compilation successful"
echo ""

echo "========================================================================"
echo "Running CUDA SiLU Tests..."
echo "========================================================================"
./silu_test
echo ""

echo "========================================================================"
echo "Profiling with Nsight Compute..."
echo "========================================================================"
make profile
echo "✓ Profiling complete"
echo ""

# Return to root directory
cd ../../..

echo "========================================================================"
echo "Section 1 (CUDA) Complete!"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - profiling_results/cuda_silu.ncu-rep (Nsight Compute)"
echo ""
echo "To view Nsight Compute results, run:"
echo "  ncu-ui profiling_results/cuda_silu.ncu-rep"
echo ""
echo "To profile with Nsight Systems, run:"
echo "  cd srcs/cuda/silu && make profile-nsys && cd ../../.."
echo ""
