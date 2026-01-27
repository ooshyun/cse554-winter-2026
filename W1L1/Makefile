# Makefile for CUDA Addition Programs
# W1L1: addition.cu and addition2.cu

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -std=c++11 -Xptxas -v

# Nsight Compute path (try default path first, fallback to system ncu)
NCU = $(shell which ncu 2>/dev/null || echo /usr/local/cuda-12.4/bin/ncu)

# GPU selection: Use GPU 6 if 8+ GPUs detected (RTX 6000 server), otherwise GPU 0
NUM_GPUS := $(shell nvidia-smi -L 2>/dev/null | wc -l)
GPU_ID ?= $(shell [ $(NUM_GPUS) -ge 8 ] && echo 6 || echo 0)

# Targets
TARGET1 = addition
TARGET2 = addition2

# Default target: build both
all: $(TARGET1) $(TARGET2)

# Compile addition.cu
$(TARGET1): addition.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Compile addition2.cu
$(TARGET2): addition2.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Run addition
run-addition: $(TARGET1)
	CUDA_VISIBLE_DEVICES=$(GPU_ID) ./$(TARGET1)

# Run addition2
run-addition2: $(TARGET2)
	CUDA_VISIBLE_DEVICES=$(GPU_ID) ./$(TARGET2)

# Run both
run: run-addition run-addition2

# Profile addition with Nsight Compute
profile-addition: $(TARGET1)
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	CUDA_VISIBLE_DEVICES=$(GPU_ID) $(NCU) -o addition_$$timestamp --set full ./$(TARGET1)

# Profile addition2 with Nsight Compute
profile-addition2: $(TARGET2)
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	CUDA_VISIBLE_DEVICES=$(GPU_ID) $(NCU) -o addition2_$$timestamp --set full ./$(TARGET2)

# Profile both
profile: profile-addition profile-addition2

# Clean build files
clean:
	rm -f $(TARGET1) $(TARGET2) *.o

# Clean all including profiling results
cleanall: clean
	rm -f addition_*.ncu-rep addition2_*.ncu-rep

.PHONY: all run run-addition run-addition2 profile profile-addition profile-addition2 clean cleanall
