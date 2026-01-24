#!/bin/bash
# CSE 554 Assignment 1 - Section 3: Run Host-GPU Memory Tests
# This script compiles and runs host-GPU memory copy implementations

set -e  # Exit on error

echo "========================================================================"
echo "CSE 554 Assignment 1 - Section 3: Host-GPU Memory Copy"
echo "========================================================================"
echo ""

# Create profiling results directory
mkdir -p profiling_results

cd srcs/cuda/host_gpu

echo "Compiling Host-GPU memory transfer tests..."
make clean
make
echo "✓ Compilation successful"
echo ""

# Q1-Q2: Memory Transfer Bandwidth Tests
echo "========================================================================"
echo "Q1-Q2: Memory Transfer Bandwidth Tests"
echo "========================================================================"
echo "Measuring Host-to-GPU and GPU-to-Host bandwidth..."
echo "Transfer sizes: 2^0 to 2^20 bytes"
echo ""

./memory_transfer_test
echo ""

# Q3: First Column Copy
echo "========================================================================"
echo "Q3: First Column Copy (8192, 65536)"
echo "========================================================================"
echo "Copying first column of each row to GPU..."
echo "Target: < 100 μs"
echo ""

./copy_first_column_test
echo ""

echo "Profiling with Nsight Systems..."
make profile-copy
echo "✓ Profiling complete"
echo ""

cd ../../..

# Summary
echo "========================================================================"
echo "Section 3 Complete!"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - profiling_results/copy.nsys-rep"
echo "  - bandwidth_plot_data.csv (if generated)"
echo ""
echo "Tasks completed:"
echo "  ✓ Q1: Measured Host-to-GPU and GPU-to-Host bandwidth"
echo "  ✓ Q2: Tested pinned memory performance"
echo "  ✓ Q3: Optimized first column copy (target: <100 μs)"
echo ""
echo "Next steps:"
echo "  1. Plot bandwidth vs transfer size curve"
echo "  2. Include pinned memory curve in the plot"
echo "  3. Add plots and screenshots to answer sheet"
echo ""
echo "To view Nsight Systems results:"
echo "  nsys-ui profiling_results/copy.nsys-rep"
echo ""
