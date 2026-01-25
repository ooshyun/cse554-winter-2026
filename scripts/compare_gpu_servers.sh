#!/bin/bash
#
# GPU Server Comparison Script
# Compares GPU configurations between two servers
#
# Usage:
#   1. Run on Server 1: ./compare_gpu_servers.sh --save server1
#   2. Run on Server 2: ./compare_gpu_servers.sh --save server2
#   3. Compare: ./compare_gpu_servers.sh --compare server1 server2
#

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

CONFIG_DIR="$HOME/.gpu_configs"
mkdir -p "$CONFIG_DIR"

# Function to save GPU configuration
save_config() {
    local name=$1
    local output_file="$CONFIG_DIR/${name}.conf"

    echo "Saving GPU configuration to: $output_file"

    # Get GPU information
    {
        echo "SERVER_NAME=$(hostname)"
        echo "TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"
        echo "GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -1)"
        echo "COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
        echo "DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        echo "CUDA_DRIVER_VERSION=$(nvidia-smi | grep 'CUDA Version' | awk '{print $9}')"

        if command -v nvcc &> /dev/null; then
            echo "NVCC_VERSION=$(nvcc --version | grep 'release' | awk '{print $5}' | cut -d',' -f1)"
        else
            echo "NVCC_VERSION=NOT_INSTALLED"
        fi

        echo "MEMORY_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
        echo "MEMORY_CLOCK=$(nvidia-smi --query-gpu=clocks.mem --format=csv,noheader | head -1)"
        echo "GRAPHICS_CLOCK=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader | head -1)"
        echo "SM_CLOCK=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader | head -1)"
        echo "MAX_MEM_CLOCK=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader | head -1)"
        echo "MAX_GRAPHICS_CLOCK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader | head -1)"
        echo "POWER_LIMIT=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader | head -1)"
        echo "PERF_STATE=$(nvidia-smi --query-gpu=pstate --format=csv,noheader | head -1)"

        # CUDA device properties
        if command -v nvcc &> /dev/null; then
            cat > /tmp/gpu_props.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("MULTIPROCESSOR_COUNT=%d\n", prop.multiProcessorCount);
    printf("MEMORY_BUS_WIDTH=%d\n", prop.memoryBusWidth);
    printf("L2_CACHE_SIZE=%d\n", prop.l2CacheSize);
    printf("SHARED_MEM_PER_BLOCK=%zu\n", prop.sharedMemPerBlock);
    printf("REGISTERS_PER_BLOCK=%d\n", prop.regsPerBlock);
    printf("MAX_THREADS_PER_BLOCK=%d\n", prop.maxThreadsPerBlock);
    printf("WARP_SIZE=%d\n", prop.warpSize);
    printf("CONCURRENT_KERNELS=%d\n", prop.concurrentKernels);
    printf("COOPERATIVE_LAUNCH=%d\n", prop.cooperativeLaunch);

    return 0;
}
EOF
            if nvcc -o /tmp/gpu_props /tmp/gpu_props.cu 2>/dev/null; then
                /tmp/gpu_props
                rm -f /tmp/gpu_props /tmp/gpu_props.cu
            fi
        fi

    } > "$output_file"

    echo "Configuration saved successfully!"
    echo ""
}

# Function to compare two configurations
compare_configs() {
    local name1=$1
    local name2=$2
    local file1="$CONFIG_DIR/${name1}.conf"
    local file2="$CONFIG_DIR/${name2}.conf"

    if [ ! -f "$file1" ]; then
        echo -e "${RED}Error: Configuration file not found: $file1${NC}"
        echo "Run: $0 --save $name1"
        exit 1
    fi

    if [ ! -f "$file2" ]; then
        echo -e "${RED}Error: Configuration file not found: $file2${NC}"
        echo "Run: $0 --save $name2"
        exit 1
    fi

    # Load configurations
    source "$file1"
    SERVER1_NAME=$SERVER_NAME
    GPU1_NAME=$GPU_NAME
    GPU1_COUNT=$GPU_COUNT
    GPU1_COMPUTE_CAP=$COMPUTE_CAP
    GPU1_DRIVER=$DRIVER_VERSION
    GPU1_CUDA_DRIVER=$CUDA_DRIVER_VERSION
    GPU1_NVCC=$NVCC_VERSION
    GPU1_MEMORY=$MEMORY_TOTAL
    GPU1_MEM_CLOCK=$MEMORY_CLOCK
    GPU1_GRAPHICS_CLOCK=$GRAPHICS_CLOCK
    GPU1_SM_CLOCK=$SM_CLOCK
    GPU1_MAX_MEM=$MAX_MEM_CLOCK
    GPU1_MAX_GRAPHICS=$MAX_GRAPHICS_CLOCK
    GPU1_POWER=$POWER_LIMIT
    GPU1_PERF=$PERF_STATE
    GPU1_SM_COUNT=$MULTIPROCESSOR_COUNT
    GPU1_BUS_WIDTH=$MEMORY_BUS_WIDTH
    GPU1_L2=$L2_CACHE_SIZE
    GPU1_SHARED_MEM=$SHARED_MEM_PER_BLOCK
    GPU1_MAX_THREADS=$MAX_THREADS_PER_BLOCK
    GPU1_COOP=$COOPERATIVE_LAUNCH

    source "$file2"
    SERVER2_NAME=$SERVER_NAME
    GPU2_NAME=$GPU_NAME
    GPU2_COUNT=$GPU_COUNT
    GPU2_COMPUTE_CAP=$COMPUTE_CAP
    GPU2_DRIVER=$DRIVER_VERSION
    GPU2_CUDA_DRIVER=$CUDA_DRIVER_VERSION
    GPU2_NVCC=$NVCC_VERSION
    GPU2_MEMORY=$MEMORY_TOTAL
    GPU2_MEM_CLOCK=$MEMORY_CLOCK
    GPU2_GRAPHICS_CLOCK=$GRAPHICS_CLOCK
    GPU2_SM_CLOCK=$SM_CLOCK
    GPU2_MAX_MEM=$MAX_MEM_CLOCK
    GPU2_MAX_GRAPHICS=$MAX_GRAPHICS_CLOCK
    GPU2_POWER=$POWER_LIMIT
    GPU2_PERF=$PERF_STATE
    GPU2_SM_COUNT=$MULTIPROCESSOR_COUNT
    GPU2_BUS_WIDTH=$MEMORY_BUS_WIDTH
    GPU2_L2=$L2_CACHE_SIZE
    GPU2_SHARED_MEM=$SHARED_MEM_PER_BLOCK
    GPU2_MAX_THREADS=$MAX_THREADS_PER_BLOCK
    GPU2_COOP=$COOPERATIVE_LAUNCH

    # Print comparison
    echo -e "\n${BOLD}${BLUE}========================================${NC}"
    echo -e "${BOLD}${BLUE}GPU SERVER COMPARISON${NC}"
    echo -e "${BOLD}${BLUE}========================================${NC}\n"

    printf "%-35s | %-40s | %-40s\n" "Property" "$name1" "$name2"
    printf "%-35s-+-%-40s-+-%-40s\n" "-----------------------------------" "----------------------------------------" "----------------------------------------"

    # Helper function to compare and print
    compare_line() {
        local prop=$1
        local val1=$2
        local val2=$3

        local color1=$NC
        local color2=$NC
        local marker1=""
        local marker2=""

        if [ "$val1" != "$val2" ]; then
            color1=$YELLOW
            color2=$YELLOW
            marker1="⚠"
            marker2="⚠"
        fi

        printf "%-35s | ${color1}%-40s${NC} | ${color2}%-40s${NC}\n" "$prop" "$marker1 $val1" "$marker2 $val2"
    }

    # Compare each property
    compare_line "Server Name" "$SERVER1_NAME" "$SERVER2_NAME"
    compare_line "GPU Model" "$GPU1_NAME" "$GPU2_NAME"
    compare_line "Number of GPUs" "$GPU1_COUNT" "$GPU2_COUNT"
    compare_line "Compute Capability" "$GPU1_COMPUTE_CAP" "$GPU2_COMPUTE_CAP"

    echo ""
    echo -e "${BOLD}Software Versions:${NC}"
    compare_line "NVIDIA Driver" "$GPU1_DRIVER" "$GPU2_DRIVER"
    compare_line "CUDA Driver Version" "$GPU1_CUDA_DRIVER" "$GPU2_CUDA_DRIVER"
    compare_line "CUDA Toolkit (nvcc)" "$GPU1_NVCC" "$GPU2_NVCC"

    echo ""
    echo -e "${BOLD}Hardware Specifications:${NC}"
    compare_line "Multiprocessor (SM) Count" "$GPU1_SM_COUNT" "$GPU2_SM_COUNT"
    compare_line "Total Memory" "$GPU1_MEMORY" "$GPU2_MEMORY"
    compare_line "Memory Bus Width" "$GPU1_BUS_WIDTH bits" "$GPU2_BUS_WIDTH bits"
    compare_line "L2 Cache Size" "$(($GPU1_L2 / 1024)) KB" "$(($GPU2_L2 / 1024)) KB"
    compare_line "Shared Mem per Block" "$(($GPU1_SHARED_MEM / 1024)) KB" "$(($GPU2_SHARED_MEM / 1024)) KB"
    compare_line "Max Threads per Block" "$GPU1_MAX_THREADS" "$GPU2_MAX_THREADS"

    echo ""
    echo -e "${BOLD}Performance Settings:${NC}"
    compare_line "Current Memory Clock" "$GPU1_MEM_CLOCK" "$GPU2_MEM_CLOCK"
    compare_line "Current Graphics Clock" "$GPU1_GRAPHICS_CLOCK" "$GPU2_GRAPHICS_CLOCK"
    compare_line "Current SM Clock" "$GPU1_SM_CLOCK" "$GPU2_SM_CLOCK"
    compare_line "Max Memory Clock" "$GPU1_MAX_MEM" "$GPU2_MAX_MEM"
    compare_line "Max Graphics Clock" "$GPU1_MAX_GRAPHICS" "$GPU2_MAX_GRAPHICS"
    compare_line "Power Limit" "$GPU1_POWER" "$GPU2_POWER"
    compare_line "Performance State" "$GPU1_PERF" "$GPU2_PERF"

    echo ""
    echo -e "${BOLD}Features:${NC}"
    compare_line "Cooperative Launch Support" "$([ $GPU1_COOP -eq 1 ] && echo 'Yes' || echo 'No')" "$([ $GPU2_COOP -eq 1 ] && echo 'Yes' || echo 'No')"

    # Calculate memory bandwidth
    echo ""
    echo -e "${BOLD}${BLUE}========================================${NC}"
    echo -e "${BOLD}${BLUE}MEMORY BANDWIDTH COMPARISON${NC}"
    echo -e "${BOLD}${BLUE}========================================${NC}\n"

    # Extract numeric values
    MEM_CLOCK1=$(echo $GPU1_MEM_CLOCK | awk '{print $1}')
    MEM_CLOCK2=$(echo $GPU2_MEM_CLOCK | awk '{print $1}')

    if [ -n "$MEM_CLOCK1" ] && [ -n "$GPU1_BUS_WIDTH" ]; then
        BW1=$(echo "scale=2; 2 * $MEM_CLOCK1 / 1000 * $GPU1_BUS_WIDTH / 8" | bc)
        echo -e "${CYAN}$name1:${NC}"
        echo "  Memory Clock: $GPU1_MEM_CLOCK"
        echo "  Bus Width: $GPU1_BUS_WIDTH bits"
        echo "  Calculated Bandwidth: ${GREEN}$BW1 GB/s${NC}"
    fi

    echo ""

    if [ -n "$MEM_CLOCK2" ] && [ -n "$GPU2_BUS_WIDTH" ]; then
        BW2=$(echo "scale=2; 2 * $MEM_CLOCK2 / 1000 * $GPU2_BUS_WIDTH / 8" | bc)
        echo -e "${CYAN}$name2:${NC}"
        echo "  Memory Clock: $GPU2_MEM_CLOCK"
        echo "  Bus Width: $GPU2_BUS_WIDTH bits"
        echo "  Calculated Bandwidth: ${GREEN}$BW2 GB/s${NC}"
    fi

    # Compilation compatibility
    echo ""
    echo -e "${BOLD}${BLUE}========================================${NC}"
    echo -e "${BOLD}${BLUE}COMPILATION COMPATIBILITY${NC}"
    echo -e "${BOLD}${BLUE}========================================${NC}\n"

    MAJOR1=$(echo $GPU1_COMPUTE_CAP | cut -d'.' -f1)
    MINOR1=$(echo $GPU1_COMPUTE_CAP | cut -d'.' -f2)
    MAJOR2=$(echo $GPU2_COMPUTE_CAP | cut -d'.' -f1)
    MINOR2=$(echo $GPU2_COMPUTE_CAP | cut -d'.' -f2)

    if [ "$GPU1_COMPUTE_CAP" = "$GPU2_COMPUTE_CAP" ]; then
        echo -e "${GREEN}✓ Both GPUs have same compute capability ($GPU1_COMPUTE_CAP)${NC}"
        echo -e "  Use: ${CYAN}-arch=sm_${MAJOR1}${MINOR1}${NC}"
    else
        echo -e "${YELLOW}⚠ Different compute capabilities:${NC}"
        echo -e "  $name1: $GPU1_COMPUTE_CAP → ${CYAN}-arch=sm_${MAJOR1}${MINOR1}${NC}"
        echo -e "  $name2: $GPU2_COMPUTE_CAP → ${CYAN}-arch=sm_${MAJOR2}${MINOR2}${NC}"
        echo ""
        echo "  For compatibility, use lower architecture or compile separately"
    fi

    if [ "$GPU1_NVCC" != "$GPU2_NVCC" ]; then
        echo ""
        echo -e "${YELLOW}⚠ Different CUDA Toolkit versions:${NC}"
        echo "  $name1: $GPU1_NVCC"
        echo "  $name2: $GPU2_NVCC"
        echo "  This may cause compilation differences!"
    fi

    echo ""
    echo -e "${BOLD}${BLUE}========================================${NC}"
    echo -e "${BOLD}${BLUE}RECOMMENDATIONS${NC}"
    echo -e "${BOLD}${BLUE}========================================${NC}\n"

    echo "For cross-server compatibility:"
    echo ""
    echo "1. Use common.mk for automatic architecture detection:"
    echo "   include common.mk"
    echo ""
    echo "2. Test on both servers:"
    echo "   Server 1 ($name1): make clean && make && ./test"
    echo "   Server 2 ($name2): make clean && make && ./test"
    echo ""
    echo "3. Profile on both to compare performance:"
    echo "   ncu --set full ./program"
    echo ""
}

# Main script
if [ "$1" = "--save" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --save <config_name>"
        exit 1
    fi
    save_config "$2"

elif [ "$1" = "--compare" ]; then
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "Usage: $0 --compare <config1> <config2>"
        exit 1
    fi
    compare_configs "$2" "$3"

elif [ "$1" = "--list" ]; then
    echo "Saved configurations:"
    ls -1 "$CONFIG_DIR"/*.conf 2>/dev/null | sed 's/.*\///' | sed 's/\.conf$//' || echo "No configurations saved yet"

else
    echo "GPU Server Comparison Tool"
    echo ""
    echo "Usage:"
    echo "  $0 --save <name>          Save current GPU configuration"
    echo "  $0 --compare <n1> <n2>    Compare two saved configurations"
    echo "  $0 --list                 List saved configurations"
    echo ""
    echo "Example workflow:"
    echo "  # On RTX 4070 Ti SUPER server:"
    echo "  $0 --save rtx4070"
    echo ""
    echo "  # On Quadro RTX 6000 server:"
    echo "  $0 --save quadro6000"
    echo ""
    echo "  # Compare (can run on either server):"
    echo "  $0 --compare rtx4070 quadro6000"
    exit 1
fi
