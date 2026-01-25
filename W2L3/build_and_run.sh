# #! /bin/bash

# nvcc -o branch branch.cu
# echo "✅ Branch built."
# nvcc -o reduction1 reduction1.cu
# echo "✅ Reduction1 built."
# nvcc -o reduction2 reduction2.cu
# echo "✅ Reduction2 built."
# nvcc -o reduction3 reduction3.cu
# echo "✅ Reduction3 built."
# nvcc -o reduction4 reduction4.cu
# echo "✅ Reduction4 built."
# nvcc -o reduction5 reduction5.cu
# echo "✅ Reduction5 built."

# nvcc -o transpose_v1 transpose_v1.cu
# echo "✅ Transpose_v1 built."
# nvcc -o transpose_v2 transpose_v2.cu
# echo "✅ Transpose_v2 built."
# nvcc -o transpose_v3 transpose_v3.cu
# echo "✅ Transpose_v3 built."
# nvcc -o transpose_v4 transpose_v4.cu
# echo "✅ Transpose_v4 built."
# nvcc -o transpose_v5 transpose_v5.cu
# echo "✅ Transpose_v5 built."

# echo "✅ Running branch..."
# ./branch
# echo "✅ Branch run complete."
# ./reduction1
# echo "✅ Reduction1 run complete."
# ./reduction2
# echo "✅ Reduction2 run complete."
# ./reduction3
# echo "✅ Reduction3 run complete."
# ./reduction4
# echo "✅ Reduction4 run complete."
# ./reduction5
# echo "✅ Reduction5 run complete."
# ./transpose_v1
# echo "✅ Transpose_v1 run complete."
# ./transpose_v2
# echo "✅ Transpose_v2 run complete."
# ./transpose_v3
# echo "✅ Transpose_v3 run complete."
# ./transpose_v4
# echo "✅ Transpose_v4 run complete."
# ./transpose_v5
# echo "✅ Transpose_v5 run complete."

echo "✅ Profiling..."
timestamp=$(date +%Y%m%d_%H%M%S)
echo "✅ Creating profiling results directory..."
mkdir -p profiling_results
ncu -o profiling_results/branch_$timestamp --set full ./branch
echo "✅ Profiling branch complete."
ncu -o profiling_results/reduction1_$timestamp --set full ./reduction1
echo "✅ Profiling reduction1 complete."
ncu -o profiling_results/reduction2_$timestamp --set full ./reduction2
echo "✅ Profiling reduction2 complete."
ncu -o profiling_results/reduction3_$timestamp --set full ./reduction3
echo "✅ Profiling reduction3 complete."
ncu -o profiling_results/reduction4_$timestamp --set full ./reduction4
echo "✅ Profiling reduction4 complete."
ncu -o profiling_results/reduction5_$timestamp --set full ./reduction5
echo "✅ Profiling reduction5 complete."
ncu -o profiling_results/transpose_v1_$timestamp --set full ./transpose_v1
echo "✅ Profiling transpose_v1 complete."
ncu -o profiling_results/transpose_v2_$timestamp --set full ./transpose_v2
echo "✅ Profiling transpose_v2 complete."
ncu -o profiling_results/transpose_v3_$timestamp --set full ./transpose_v3
echo "✅ Profiling transpose_v3 complete."
ncu -o profiling_results/transpose_v4_$timestamp --set full ./transpose_v4
echo "✅ Profiling transpose_v4 complete."
ncu -o profiling_results/transpose_v5_$timestamp --set full ./transpose_v5
echo "✅ Profiling transpose_v5 complete."
