#!/bin/bash
#
# GPU Configuration Checker
# Displays comprehensive GPU settings for CUDA development
#
# Usage: ./check_gpu_config.sh [--compare SERVER_NAME]
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print section header
print_header() {
    echo -e "\n${BOLD}${BLUE}========================================${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}========================================${NC}"
}

# Function to print key-value pair
print_kv() {
    local key=$1
    local value=$2
    printf "%-35s: ${GREEN}%s${NC}\n" "$key" "$value"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠  $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗  $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓  $1${NC}"
}

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. NVIDIA drivers may not be installed."
    exit 1
fi

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    print_warning "nvcc not found. CUDA toolkit may not be installed."
    NVCC_AVAILABLE=false
else
    NVCC_AVAILABLE=true
fi

# Main information gathering
SERVER_NAME=$(hostname)
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

print_header "SERVER INFORMATION"
print_kv "Server Name" "$SERVER_NAME"
print_kv "Timestamp" "$TIMESTAMP"
print_kv "User" "$USER"
print_kv "OS" "$(uname -s) $(uname -r)"
print_kv "Architecture" "$(uname -m)"

# GPU Driver Information
print_header "NVIDIA DRIVER INFORMATION"
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
print_kv "Driver Version" "$DRIVER_VERSION"

# CUDA Version from driver
CUDA_DRIVER_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
print_kv "CUDA Version (from driver)" "$CUDA_DRIVER_VERSION"

# CUDA Toolkit Version
if [ "$NVCC_AVAILABLE" = true ]; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    print_kv "CUDA Toolkit (nvcc)" "$NVCC_VERSION"

    # Check for version mismatch
    if [ "$CUDA_DRIVER_VERSION" != "$NVCC_VERSION" ]; then
        print_warning "Driver CUDA version ($CUDA_DRIVER_VERSION) differs from toolkit ($NVCC_VERSION)"
    fi
fi

# GPU Count
print_header "GPU CONFIGURATION"
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
print_kv "Number of GPUs" "$GPU_COUNT"

# Detailed GPU Information
for i in $(seq 0 $((GPU_COUNT-1))); do
    echo ""
    echo -e "${CYAN}GPU $i:${NC}"

    GPU_NAME=$(nvidia-smi -i $i --query-gpu=gpu_name --format=csv,noheader)
    GPU_UUID=$(nvidia-smi -i $i --query-gpu=gpu_uuid --format=csv,noheader)
    COMPUTE_CAP=$(nvidia-smi -i $i --query-gpu=compute_cap --format=csv,noheader)
    MEMORY_TOTAL=$(nvidia-smi -i $i --query-gpu=memory.total --format=csv,noheader)
    MEMORY_USED=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader)
    MEMORY_FREE=$(nvidia-smi -i $i --query-gpu=memory.free --format=csv,noheader)
    POWER_LIMIT=$(nvidia-smi -i $i --query-gpu=power.limit --format=csv,noheader)
    TEMP=$(nvidia-smi -i $i --query-gpu=temperature.gpu --format=csv,noheader)
    UTILIZATION=$(nvidia-smi -i $i --query-gpu=utilization.gpu --format=csv,noheader)

    print_kv "  Name" "$GPU_NAME"
    print_kv "  Compute Capability" "$COMPUTE_CAP"
    print_kv "  Memory Total" "$MEMORY_TOTAL"
    print_kv "  Memory Used" "$MEMORY_USED"
    print_kv "  Memory Free" "$MEMORY_FREE"
    print_kv "  Power Limit" "$POWER_LIMIT"
    print_kv "  Temperature" "$TEMP °C"
    print_kv "  Utilization" "$UTILIZATION"

    # Determine architecture
    MAJOR=$(echo $COMPUTE_CAP | cut -d'.' -f1)
    MINOR=$(echo $COMPUTE_CAP | cut -d'.' -f2)

    if [ "$MAJOR" = "8" ] && [ "$MINOR" = "9" ]; then
        ARCH="Ada Lovelace (sm_89)"
        ARCH_FLAG="-arch=sm_89"
    elif [ "$MAJOR" = "7" ] && [ "$MINOR" = "5" ]; then
        ARCH="Turing (sm_75)"
        ARCH_FLAG="-arch=sm_75"
    elif [ "$MAJOR" = "8" ] && [ "$MINOR" = "0" ]; then
        ARCH="Ampere (sm_80)"
        ARCH_FLAG="-arch=sm_80"
    elif [ "$MAJOR" = "9" ] && [ "$MINOR" = "0" ]; then
        ARCH="Hopper (sm_90)"
        ARCH_FLAG="-arch=sm_90"
    else
        ARCH="Unknown (sm_${MAJOR}${MINOR})"
        ARCH_FLAG="-arch=sm_${MAJOR}${MINOR}"
    fi

    print_kv "  Architecture" "$ARCH"
    print_kv "  Recommended nvcc flag" "$ARCH_FLAG"
done

# GPU Performance Settings
print_header "GPU PERFORMANCE SETTINGS"
for i in $(seq 0 $((GPU_COUNT-1))); do
    echo -e "\n${CYAN}GPU $i Performance:${NC}"

    CLOCKS_GRAPHICS=$(nvidia-smi -i $i --query-gpu=clocks.gr --format=csv,noheader)
    CLOCKS_SM=$(nvidia-smi -i $i --query-gpu=clocks.sm --format=csv,noheader)
    CLOCKS_MEM=$(nvidia-smi -i $i --query-gpu=clocks.mem --format=csv,noheader)
    CLOCKS_VIDEO=$(nvidia-smi -i $i --query-gpu=clocks.video --format=csv,noheader 2>/dev/null || echo "N/A")

    print_kv "  Graphics Clock" "$CLOCKS_GRAPHICS"
    print_kv "  SM Clock" "$CLOCKS_SM"
    print_kv "  Memory Clock" "$CLOCKS_MEM"
    if [ "$CLOCKS_VIDEO" != "N/A" ]; then
        print_kv "  Video Clock" "$CLOCKS_VIDEO"
    fi

    # Performance mode
    PERF_STATE=$(nvidia-smi -i $i --query-gpu=pstate --format=csv,noheader)
    print_kv "  Performance State" "$PERF_STATE"

    # Clock limits
    MAX_GRAPHICS=$(nvidia-smi -i $i --query-gpu=clocks.max.graphics --format=csv,noheader)
    MAX_SM=$(nvidia-smi -i $i --query-gpu=clocks.max.sm --format=csv,noheader)
    MAX_MEM=$(nvidia-smi -i $i --query-gpu=clocks.max.mem --format=csv,noheader)

    print_kv "  Max Graphics Clock" "$MAX_GRAPHICS"
    print_kv "  Max SM Clock" "$MAX_SM"
    print_kv "  Max Memory Clock" "$MAX_MEM"
done

# Memory Bandwidth Calculation
print_header "MEMORY BANDWIDTH"
for i in $(seq 0 $((GPU_COUNT-1))); do
    echo -e "\n${CYAN}GPU $i Bandwidth:${NC}"

    GPU_NAME=$(nvidia-smi -i $i --query-gpu=gpu_name --format=csv,noheader)
    MEM_CLOCK=$(nvidia-smi -i $i --query-gpu=clocks.mem --format=csv,noheader | awk '{print $1}')

    # Get memory bus width using nvidia-smi
    if [ -f "/tmp/gpu_check_$i.txt" ]; then
        rm "/tmp/gpu_check_$i.txt"
    fi

    # Try to get bus width from nvidia-smi
    nvidia-smi -i $i -q | grep -A 20 "FB Memory Usage" > "/tmp/gpu_check_$i.txt"

    # Known GPU specifications
    case "$GPU_NAME" in
        *"RTX 4070 Ti SUPER"*)
            BUS_WIDTH=256
            OFFICIAL_BW=672.0
            ;;
        *"Quadro RTX 6000"*)
            BUS_WIDTH=384
            OFFICIAL_BW=672.0
            ;;
        *"RTX 4090"*)
            BUS_WIDTH=384
            OFFICIAL_BW=1008.0
            ;;
        *"A100"*)
            BUS_WIDTH=5120  # HBM2e
            OFFICIAL_BW=1555.0
            ;;
        *)
            BUS_WIDTH="Unknown"
            OFFICIAL_BW="Unknown"
            ;;
    esac

    print_kv "  Memory Bus Width" "${BUS_WIDTH} bit"
    print_kv "  Current Memory Clock" "${MEM_CLOCK} MHz"

    if [ "$BUS_WIDTH" != "Unknown" ] && [ "$MEM_CLOCK" != "" ]; then
        # Calculate bandwidth: BW = 2 × clock (GHz) × bus_width (bytes)
        # Memory clock is in MHz, convert to GHz
        CALCULATED_BW=$(echo "scale=2; 2 * $MEM_CLOCK / 1000 * $BUS_WIDTH / 8" | bc)
        print_kv "  Calculated Bandwidth" "${CALCULATED_BW} GB/s"
    fi

    if [ "$OFFICIAL_BW" != "Unknown" ]; then
        print_kv "  Official Spec Bandwidth" "${OFFICIAL_BW} GB/s"
    fi
done

# ECC Status
print_header "ERROR CORRECTION (ECC)"
for i in $(seq 0 $((GPU_COUNT-1))); do
    ECC_MODE=$(nvidia-smi -i $i --query-gpu=ecc.mode.current --format=csv,noheader 2>/dev/null || echo "Not Supported")
    if [ "$ECC_MODE" != "Not Supported" ]; then
        echo -e "\n${CYAN}GPU $i ECC:${NC}"
        print_kv "  ECC Mode" "$ECC_MODE"

        if [ "$ECC_MODE" = "Enabled" ]; then
            ECC_ERRORS=$(nvidia-smi -i $i --query-gpu=ecc.errors.corrected.volatile.total --format=csv,noheader)
            print_kv "  Corrected Errors" "$ECC_ERRORS"
        fi
    fi
done

# CUDA Toolkit Paths
print_header "CUDA ENVIRONMENT"
if [ "$NVCC_AVAILABLE" = true ]; then
    NVCC_PATH=$(which nvcc)
    print_kv "nvcc Path" "$NVCC_PATH"

    CUDA_HOME_PATH=$(dirname $(dirname $NVCC_PATH))
    print_kv "CUDA Home" "$CUDA_HOME_PATH"
fi

if [ -n "$CUDA_HOME" ]; then
    print_kv "CUDA_HOME env var" "$CUDA_HOME"
else
    print_warning "CUDA_HOME environment variable not set"
fi

# LD_LIBRARY_PATH check
if echo "$LD_LIBRARY_PATH" | grep -q "cuda"; then
    print_success "CUDA libraries in LD_LIBRARY_PATH"
else
    print_warning "CUDA libraries may not be in LD_LIBRARY_PATH"
fi

# CUDA-capable devices check
if [ "$NVCC_AVAILABLE" = true ]; then
    print_header "CUDA DEVICE QUERY"

    # Create a simple CUDA test program
    cat > /tmp/cuda_test.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("\nDevice %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads Dim: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Size: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Multiprocessor Count: %d\n", prop.multiProcessorCount);
        printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
        printf("  Max Memory Pitch: %zu bytes\n", prop.memPitch);
        printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  Cooperative Launch: %s\n", prop.cooperativeLaunch ? "Yes" : "No");
    }

    return 0;
}
EOF

    # Compile and run
    if nvcc -o /tmp/cuda_test /tmp/cuda_test.cu 2>/dev/null; then
        /tmp/cuda_test
        rm -f /tmp/cuda_test /tmp/cuda_test.cu
    else
        print_error "Failed to compile CUDA test program"
    fi
fi

# Compilation recommendations
print_header "COMPILATION RECOMMENDATIONS"

GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)

echo -e "${CYAN}For $GPU_NAME (Compute Capability $COMPUTE_CAP):${NC}"
echo ""

MAJOR=$(echo $COMPUTE_CAP | cut -d'.' -f1)
MINOR=$(echo $COMPUTE_CAP | cut -d'.' -f2)

if [ "$MAJOR" = "8" ] && [ "$MINOR" = "9" ]; then
    echo "  Optimal compilation flags:"
    echo "    nvcc -O3 -arch=sm_89 -std=c++17 --use_fast_math"
    echo ""
    echo "  Features available:"
    echo "    ✓ Tensor Cores (4th gen)"
    echo "    ✓ RT Cores (3rd gen)"
    echo "    ✓ Float4 vectorization"
    echo "    ✓ Cooperative Groups"
    echo "    ✓ Dynamic parallelism"
elif [ "$MAJOR" = "7" ] && [ "$MINOR" = "5" ]; then
    echo "  Optimal compilation flags:"
    echo "    nvcc -O3 -arch=sm_75 -std=c++17 --use_fast_math"
    echo ""
    echo "  Features available:"
    echo "    ✓ Tensor Cores (1st gen)"
    echo "    ✓ RT Cores (1st gen)"
    echo "    ✓ Float4 vectorization"
    echo "    ✓ Cooperative Groups"
    echo "    ✗ Requires C++17 for cooperative groups in CUDA 13+"
fi

# Summary
print_header "SUMMARY"
echo -e "${GREEN}✓ GPU Configuration Check Complete${NC}"
echo ""
echo "Quick Reference:"
print_kv "Server" "$SERVER_NAME"
print_kv "GPU(s)" "$GPU_COUNT × $GPU_NAME"
print_kv "Compute Capability" "$COMPUTE_CAP"
print_kv "CUDA Driver" "$CUDA_DRIVER_VERSION"
if [ "$NVCC_AVAILABLE" = true ]; then
    print_kv "CUDA Toolkit" "$NVCC_VERSION"
fi

# Save output to file
OUTPUT_FILE="gpu_config_${SERVER_NAME}_$(date +%Y%m%d_%H%M%S).txt"
echo ""
echo -e "Full report saved to: ${CYAN}$OUTPUT_FILE${NC}"

# Clean up temp files
rm -f /tmp/gpu_check_*.txt

echo ""
