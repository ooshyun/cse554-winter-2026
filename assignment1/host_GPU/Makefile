# Makefile for Host-GPU Memory Transfer Tests
# CSE 554 Assignment 1 - Section 3

# Include common GPU detection and flags
include ../../common.mk

# Targets
MEMORY_TRANSFER = memory_transfer_test
FIRST_COLUMN = copy_first_column_test

# FLAG for Nsight Compute profiling
PROFILE_FLAGS = -DPROFILE_NCUS

all: $(MEMORY_TRANSFER) $(FIRST_COLUMN)

# Memory transfer benchmark
$(MEMORY_TRANSFER): memory_transfer.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LINK_FLAGS)

# First column copy
$(FIRST_COLUMN): copy_first_column.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LINK_FLAGS)

# Profile target executables
PROFILE_MEMORY_TRANSFER = $(MEMORY_TRANSFER)_profile
PROFILE_FIRST_COLUMN = $(FIRST_COLUMN)_profile

# Build profile targets
$(PROFILE_MEMORY_TRANSFER): memory_transfer.cu
	$(NVCC) $(NVCC_FLAGS) $(PROFILE_FLAGS) -o $@ $< $(LINK_FLAGS)

$(PROFILE_FIRST_COLUMN): copy_first_column.cu
	$(NVCC) $(NVCC_FLAGS) $(PROFILE_FLAGS) -o $@ $< $(LINK_FLAGS)

run-memory: $(MEMORY_TRANSFER)
	./$(MEMORY_TRANSFER)

run-copy: $(FIRST_COLUMN)
	./$(FIRST_COLUMN)

# Profile first column copy with Nsight Systems
profile-copy: $(PROFILE_FIRST_COLUMN)
	mkdir -p profiling_results
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	nsys profile -o profiling_results/$$timestamp ./$(PROFILE_FIRST_COLUMN)
	# @timestamp=$$(date +%Y%m%d_%H%M%S); \
	# ncu -o profiling_results/$$timestamp --set full ./$(PROFILE_FIRST_COLUMN)

profile-memory: $(PROFILE_MEMORY_TRANSFER)
	mkdir -p profiling_results
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	nsys profile -o profiling_results/$$timestamp ./$(PROFILE_MEMORY_TRANSFER)
	# @timestamp=$$(date +%Y%m%d_%H%M%S); \
	# ncu -o profiling_results/$$timestamp --set full ./$(PROFILE_MEMORY_TRANSFER)

clean:
	rm -f $(MEMORY_TRANSFER) $(FIRST_COLUMN) $(PROFILE_MEMORY_TRANSFER) $(PROFILE_FIRST_COLUMN)
	rm -f bandwidth_data.csv

cleanall: clean
	rm -f profiling_results/copy.*

.PHONY: all run-memory run-copy profile-copy clean cleanall
