# Makefile for Host-GPU Memory Transfer Tests
# CSE 554 Assignment 1 - Section 3

# Include common GPU detection and flags
include ../../common.mk

# Targets
HOST_GPU = host_GPU_test

# FLAG for Nsight Compute profiling
PROFILE_FLAGS = -DPROFILE_NCUS

all: $(HOST_GPU)

# Header
HEADERS = copy_first_column.h

# Source files
SOURCES = main.cu copy_first_column.cu
OBJECTS = $(SOURCES:.cu=.o)

# Memory transfer benchmark
$(HOST_GPU): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJECTS) $(LINK_FLAGS)

# Compile source files
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile source files with profiling flags
%.profile.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(PROFILE_FLAGS) -c $< -o $@

# Profile object files
PROFILE_OBJECTS = $(SOURCES:.cu=.profile.o)

# Profile target executables
PROFILE_HOST_GPU = $(HOST_GPU)_profile

# Build profile targets
$(PROFILE_HOST_GPU): $(PROFILE_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(PROFILE_FLAGS) -o $@ $(PROFILE_OBJECTS) $(LINK_FLAGS)

run: $(HOST_GPU)
	CUDA_VISIBLE_DEVICES=$(GPU_ID) ./$(HOST_GPU)

# Profile first column copy with Nsight Systems
profile: $(HOST_GPU)
	mkdir -p profiling_results
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	CUDA_VISIBLE_DEVICES=$(GPU_ID) nsys profile -o profiling_results/$$timestamp ./$(HOST_GPU)

clean:
	rm -f $(OBJECTS) $(HOST_GPU) $(PROFILE_OBJECTS) $(PROFILE_HOST_GPU)

cleanall: clean
	rm -f profiling_results/copy.*

.PHONY: all run-memory run-copy profile-copy clean cleanall
