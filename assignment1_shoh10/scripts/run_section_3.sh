#!/bin/bash
# CSE 554 Assignment 1 - Section 3: Host-GPU Memory Copy

set -e  # Exit on error

echo "========================================================================"
echo "CSE 554 Assignment 1 - Section 3: Host-GPU Memory Copy"
echo "========================================================================"
echo ""

# Set run question number
# Default to 2, Options 2, 3, 9 is all
if [ -z "$1" ]; then
    RUN_QUESTION_NUMBER=2
fi

RUN_QUESTION_NUMBER=$1
CURRENT_DIR=$(pwd)

if [ ! -d "host_GPU/profiling_results" ]; then
    mkdir -p host_GPU/profiling_results
fi

if [ "$RUN_QUESTION_NUMBER" -eq 2 ] || [ "$RUN_QUESTION_NUMBER" -eq 9 ]; then
    # Run Section 3-Q2: Memory Transfer Bandwidth Tests
    cd host_GPU

    echo "Compiling Host-GPU memory transfer tests..."
    make clean
    make
    echo "✓ Compilation successful"
    echo ""

    echo "Measuring Host-to-GPU and GPU-to-Host bandwidth..."
    echo "Transfer sizes: 2^0 to 2^20 bytes"
    echo ""

    make run-memory
    echo ""

    # Clean up
    make clean
    echo "✓ Cleanup complete"
    echo ""

    # Return to root directory
    cd $CURRENT_DIR
else
    echo "Skipping Section 3-Q2."
fi

if [ "$RUN_QUESTION_NUMBER" -eq 3 ] || [ "$RUN_QUESTION_NUMBER" -eq 9 ]; then
    # Q3: First Column Copy
    echo "========================================================================"
    echo "Q3: First Column Copy (8192, 65536)"
    echo "========================================================================"
    echo "Copying first column of each row to GPU..."
    echo "Target: < 100 μs"
    echo ""

    cd host_GPU
    make clean
    make
    echo "✓ Compilation successful"
    echo ""

    make run-copy
    echo ""

    # Profile
    echo "Profiling with Nsight Systems..."
    make profile-copy
    echo "✓ Profiling complete"
    echo ""

    # Clean up
    make clean
    echo "✓ Cleanup complete"
    echo ""

    # Return to root directory
    cd $CURRENT_DIR
else
    echo "Skipping Section 3-Q3."
fi
