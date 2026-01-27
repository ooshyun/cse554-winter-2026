#! /bin/bash


echo "========================================================================"
echo "CSE 554 Assignment 1 - Section 2: RMS Norm"
echo "========================================================================"
echo ""

# Set run question number
# Default to 2, Options 2, 3, 9 is all
if [ -z "$1" ]; then
    RUN_QUESTION_NUMBER=2
else
    RUN_QUESTION_NUMBER=$1
fi
CURRENT_DIR=$(pwd)

if [ "$RUN_QUESTION_NUMBER" -eq 2 ] || [ "$RUN_QUESTION_NUMBER" -eq 9 ]; then
    # Run Section 2-Q2: RMS Norm (Matrix)
    cd rms_norm/matrix

    echo "Compiling RMS Norm Matrix..."
    make clean
    make
    echo "✓ Compilation successful"
    echo ""

    echo "Running RMS Norm Matrix Tests..."
    make run
    echo ""

    echo "Profiling with Nsight Compute..."
    make profile
    echo "✓ Profiling complete"
    echo ""

    # Clean up
    make clean
    echo "✓ Cleanup complete"
    echo ""

    # Return to root directory
    cd $CURRENT_DIR
else
    echo "Skipping Section 2-Q2."
fi

if [ "$RUN_QUESTION_NUMBER" -eq 3 ] || [ "$RUN_QUESTION_NUMBER" -eq 9 ]; then
    # Run Section 2-Q3: RMS Norm (Vector)
    cd rms_norm/vector

    echo "Compiling RMS Norm Vector..."
    make clean
    make
    echo "✓ Compilation successful"
    echo ""

    echo "Running RMS Norm Vector Tests..."
    make run
    echo ""

    echo "Profiling with Nsight Compute..."
    make profile
    echo "✓ Profiling complete"
    echo ""

    # Clean up
    make clean
    echo "✓ Cleanup complete"
    echo ""

    # Return to root directory
    cd $CURRENT_DIR
else
    echo "Skipping Section 2-Q3."
fi
