# Makefile for CUDA SiLU Implementation
# CSE 554 Assignment 1 - Section 1

# Include common GPU detection and flags
include ../../../common.mk

# Target executable
TARGET = silu_test

# Source files
SOURCES = main.cu silu.cu
HEADERS = silu.h

# Object files
OBJECTS = $(SOURCES:.cu=.o)

# FLAG for Nsight Compute profiling
PROFILE_FLAGS = -DPROFILE_NCUS

# Default target
all: $(TARGET)

# Link object files
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LINK_FLAGS)

# Compile source files
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile source files with profiling flags
%.profile.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(PROFILE_FLAGS) -c $< -o $@

# Profile object files
PROFILE_OBJECTS = $(SOURCES:.cu=.profile.o)

# Profile target executable
PROFILE_TARGET = $(TARGET)_profile

# Run the test
run: $(TARGET)
	CUDA_VISIBLE_DEVICES=$(GPU_ID) ./$(TARGET)

# Profile
profile: $(PROFILE_TARGET)
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	CUDA_VISIBLE_DEVICES=$(GPU_ID) ncu -o ../profiling_results/cuda_silu_$$timestamp --set full ./$(PROFILE_TARGET)

# Build profile target
$(PROFILE_TARGET): $(PROFILE_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(PROFILE_FLAGS) -o $@ $^ $(LINK_FLAGS)

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET) $(PROFILE_OBJECTS) $(PROFILE_TARGET)

# Clean all generated files
cleanall: clean
	rm -f ../profiling_results/cuda_silu.*

.PHONY: all run profile profile-ncu profile-nsys clean cleanall
