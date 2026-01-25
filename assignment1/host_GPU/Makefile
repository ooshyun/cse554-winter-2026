# Makefile for Host-GPU Memory Transfer Tests
# CSE 554 Assignment 1 - Section 3

# Include common GPU detection and flags
include ../../common.mk

# Targets
MEMORY_TRANSFER = memory_transfer_test
FIRST_COLUMN = copy_first_column_test

all: $(MEMORY_TRANSFER) $(FIRST_COLUMN)

# Memory transfer benchmark
$(MEMORY_TRANSFER): memory_transfer.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# First column copy
$(FIRST_COLUMN): copy_first_column.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run-memory: $(MEMORY_TRANSFER)
	./$(MEMORY_TRANSFER)

run-copy: $(FIRST_COLUMN)
	./$(FIRST_COLUMN)

# Profile first column copy with Nsight Systems
profile-copy: $(FIRST_COLUMN)
	mkdir -p profiling_results
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	nsys profile -o profiling_results/copy_$$timestamp ./$(FIRST_COLUMN)
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	ncu -o profiling_results/copy_$$timestamp --set full ./$(FIRST_COLUMN)

clean:
	rm -f $(MEMORY_TRANSFER) $(FIRST_COLUMN)
	rm -f bandwidth_data.csv

cleanall: clean
	rm -f profiling_results/copy.*

.PHONY: all run-memory run-copy profile-copy clean cleanall
