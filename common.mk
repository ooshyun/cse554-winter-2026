# Common Makefile Configuration for Multi-GPU Support
# Auto-detects GPU architecture and sets appropriate compilation flags

NVCC = nvcc

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

# Standard NVCC flags with detected architecture
# Note: C++17 required for CUDA 13.0 cooperative groups
# NVCC_FLAGS = -O3 -arch=$(SM_ARCH) -std=c++17 --use_fast_math -Xptxas -v
NVCC_FLAGS = -arch=$(SM_ARCH) -std=c++17
LINK_FLAGS = -lcudart

# Print detected configuration
$(info ========================================)
$(info Detected GPU: Compute Capability $(COMPUTE_CAP_MAJOR).$(COMPUTE_CAP_MINOR))
$(info Compiling for: $(SM_ARCH) ($(GPU_NAME)))
$(info ========================================)
