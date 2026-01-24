#!/bin/bash
# CSE 554 Assignment 1 - Section 2: Run RMS Norm Tests
# This script compiles and runs both matrix and vector RMS Norm implementations

set -e  # Exit on error

echo "========================================================================"
echo "CSE 554 Assignment 1 - Section 2: RMS Norm"
echo "========================================================================"
echo ""

# Create profiling results directory
mkdir -p profiling_results

# Section 2 Q2: Matrix RMS Norm (8192, 8192)
echo "========================================================================"
echo "Q2: Matrix RMS Norm (8192, 8192)"
echo "========================================================================"
cd srcs/cuda/rms_norm/matrix

echo "Compiling matrix RMS Norm..."
make clean
make
echo "✓ Compilation successful"
echo ""

echo "Running matrix RMS Norm tests..."
./rms_norm_matrix_test
echo ""

echo "Profiling with Nsight Compute..."
make profile
echo "✓ Profiling complete"
echo ""

cd ../../../..

# Section 2 Q3: Vector RMS Norm (1, 1024×1024)
echo "========================================================================"
echo "Q3: Vector RMS Norm (1, 1024×1024)"
echo "========================================================================"
cd srcs/cuda/rms_norm/vector

echo "Compiling vector RMS Norm..."
make clean
make
echo "✓ Compilation successful"
echo ""

echo "Running vector RMS Norm tests..."
./rms_norm_vector_test
echo ""

echo "Profiling with Nsight Compute..."
make profile
echo "✓ Profiling complete"
echo ""

cd ../../../..

# Summary
echo "========================================================================"
echo "Section 2 Complete!"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - profiling_results/rms_norm_matrix.ncu-rep"
echo "  - profiling_results/rms_norm_vector.ncu-rep"
echo ""
echo "Performance Targets:"
echo "  - Matrix (8192, 8192):      > 300 GB/s"
echo "  - Vector (1, 1024×1024):    > 200 GB/s"
echo ""
echo "To view results:"
echo "  ncu-ui profiling_results/rms_norm_matrix.ncu-rep"
echo "  ncu-ui profiling_results/rms_norm_vector.ncu-rep"
echo ""
