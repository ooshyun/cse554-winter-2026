#!/bin/bash
# CSE 554 Assignment 1 - Section 3: Host-GPU Memory Copy

set -e  # Exit on error

echo "========================================================================"
echo "CSE 554 Assignment 1 - Section 3: Host-GPU Memory Copy"
echo "========================================================================"
echo ""

CURRENT_DIR=$(pwd)

if [ ! -d "host_GPU/profiling_results" ]; then
    mkdir -p host_GPU/profiling_results
fi


cd host_GPU

echo "Compiling Host-GPU memory transfer tests..."
make clean
make
echo "✓ Compilation successful"
echo ""

echo "Measuring Host-to-GPU and GPU-to-Host bandwidth..."
echo "Transfer sizes: 2^0 to 2^20 bytes"
echo ""

make run
echo ""

make profile
echo ""

# Clean up
make clean
echo "✓ Cleanup complete"
echo ""

# Return to root directory
cd $CURRENT_DIR
