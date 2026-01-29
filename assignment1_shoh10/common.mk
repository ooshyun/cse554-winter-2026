# Common Makefile Configuration for Multi-GPU Support
# Auto-detects GPU architecture and sets appropriate compilation flags

NVCC = nvcc

# GPU selection: Use GPU 6 if 8+ GPUs detected (RTX 6000 server), otherwise GPU 0
NUM_GPUS := $(shell nvidia-smi -L 2>/dev/null | wc -l)
GPU_ID ?= $(shell [ $(NUM_GPUS) -ge 8 ] && echo 6 || echo 0)
export CUDA_VISIBLE_DEVICES=$(GPU_ID)

# Detect GPU compute capability
CUDA_DEVICE_QUERY := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
COMPUTE_CAP_MAJOR := $(shell echo $(CUDA_DEVICE_QUERY) | cut -d'.' -f1)
COMPUTE_CAP_MINOR := $(shell echo $(CUDA_DEVICE_QUERY) | cut -d'.' -f2)

# Determine SM architecture
ifeq ($(COMPUTE_CAP_MAJOR),8)
    ifeq ($(COMPUTE_CAP_MINOR),9)
        # RTX 4070 Ti SUPER (Ada Lovelace)
        SM_ARCH = sm_89
        GPU_NAME = RTX_4070_Ti_SUPER
    else
        SM_ARCH = sm_80
        GPU_NAME = Ampere
    endif
else ifeq ($(COMPUTE_CAP_MAJOR),7)
    ifeq ($(COMPUTE_CAP_MINOR),5)
        # Quadro RTX 6000 (Turing)
        SM_ARCH = sm_75
        GPU_NAME = Quadro_RTX_6000
    else
        SM_ARCH = sm_70
        GPU_NAME = Volta
    endif
else ifeq ($(COMPUTE_CAP_MAJOR),9)
    SM_ARCH = sm_90
    GPU_NAME = Hopper
else
    # Fallback to sm_75 for compatibility
    SM_ARCH = sm_75
    GPU_NAME = Unknown
endif

# Standard NVCC flags
# Note: sm_89 requires CUDA 11.8+, C++17 requires CUDA 11+
# For older CUDA (10.x), fall back to sm_75 and C++14
CUDA_VERSION := $(shell nvcc --version | grep "release" | sed 's/.*release \([0-9]*\).*/\1/')
ifeq ($(shell [ $(CUDA_VERSION) -ge 12 ] && echo yes),yes)
    # CUDA 12+: support both sm_75 and sm_89, C++17
    NVCC_FLAGS = -arch=$(SM_ARCH) -std=c++17 \
                 -gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_89,code=sm_89 \
                 -O3 -DNDEBUG \
                 -Xcompiler=-fPIE \
                 -Xcompiler=-Wconversion \
                 -Xcompiler=-fno-strict-aliasing
else
    # Older CUDA (10.x): sm_75 only, C++14 (compatible with Turing and newer via JIT)
    NVCC_FLAGS = -arch=sm_75 -std=c++14 \
                 -gencode arch=compute_75,code=sm_75 \
                 -O3 -DNDEBUG \
                 -Xcompiler=-fPIE \
                 -Xcompiler=-Wconversion \
                 -Xcompiler=-fno-strict-aliasing
endif

LINK_FLAGS = -cudart static

# Print detected configuration
$(info ========================================)
$(info Detected GPUs: $(NUM_GPUS), Using GPU: $(GPU_ID))
$(info Compute Capability: $(COMPUTE_CAP_MAJOR).$(COMPUTE_CAP_MINOR))
$(info Compiling for: $(SM_ARCH) ($(GPU_NAME)))
$(info ========================================)
