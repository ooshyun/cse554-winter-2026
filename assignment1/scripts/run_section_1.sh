#!/bin/bash
# CSE 554 Assignment 1 - Section 1: SiLU (Python/Triton)

set -e  # Exit on error

echo "========================================================================"
echo "CSE 554 Assignment 1 - Section 1: SiLU (Python/Triton)"
echo "========================================================================"
echo ""

# Set run question number
# Default to 2, Options 2, 3, 4, 9 is all
if [ -z "$1" ]; then
    RUN_QUESTION_NUMBER=2
fi

RUN_QUESTION_NUMBER=$1
CURRENT_DIR=$(pwd)

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
    uv pip install -r requirements.txt
fi

if [ "$RUN_QUESTION_NUMBER" -eq 2 ] || [ "$RUN_QUESTION_NUMBER" -eq 9 ]; then
    # Run Section 1-Q2: SiLU (PyTorch Implementation)
    if [ ! -d "silu/profiling_results" ]; then
        mkdir -p silu/profiling_results
    fi

    PROFILE_NAME="torch_silu"
    uv run python silu/silu_torch.py --profile --profile_name $PROFILE_NAME
    nsys profile -o silu/profiling_results/$PROFILE_NAME --stats=true uv run python silu/silu_torch.py

    if [ -f "silu/profiling_results/$PROFILE_NAME.sqlite" ]; then
        rm silu/profiling_results/$PROFILE_NAME.sqlite
    fi
    nsys stats --report cuda_gpu_kern_sum silu/profiling_results/$PROFILE_NAME.nsys-rep
else
    echo "Skipping Section 1-Q2."
fi

if [ "$RUN_QUESTION_NUMBER" -eq 3 ] || [ "$RUN_QUESTION_NUMBER" -eq 9 ]; then
    # Run Section 1-Q3: SiLU (Triton Kernel)
    uv run python silu/silu_triton_test.py
else
    echo "Skipping Section 1-Q3."
fi

if [ "$RUN_QUESTION_NUMBER" -eq 4 ] || [ "$RUN_QUESTION_NUMBER" -eq 9 ]; then
    # Run Section 1-Q4: SiLU (CUDA Implementation)
    cd silu/CUDA

    echo "Compiling CUDA SiLU kernel..."
    make clean
    make
    echo "✓ Compilation successful"
    echo ""

    echo "========================================================================"
    echo "Running CUDA SiLU Tests..."
    echo "========================================================================"
    if [ ! -f "silu_test" ]; then
        echo "Error: silu_test not found"
        exit 1
    fi
    ./silu_test
    echo ""

    echo "========================================================================"
    echo "Profiling with Nsight Compute..."
    echo "========================================================================"
    sudo make profile-ncu
    echo "✓ Profiling complete"
    echo ""

    # Clean up
    make clean
    echo "✓ Cleanup complete"
    echo ""

    # Return to root directory
    cd $CURRENT_DIR
else 
    echo "Skipping Section 1-Q4."
fi