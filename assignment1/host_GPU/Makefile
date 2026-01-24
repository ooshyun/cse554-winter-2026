# Makefile for Host-GPU Memory Transfer Tests
# CSE 554 Assignment 1 - Section 3

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_89 -std=c++11

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
	nsys profile -o ../../../profiling_results/copy ./$(FIRST_COLUMN)

clean:
	rm -f $(MEMORY_TRANSFER) $(FIRST_COLUMN)
	rm -f bandwidth_data.csv

cleanall: clean
	rm -f ../../../profiling_results/copy.*

.PHONY: all run-memory run-copy profile-copy clean cleanall
