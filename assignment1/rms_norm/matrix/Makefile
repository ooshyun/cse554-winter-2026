# Makefile for RMS Norm Matrix Implementation
# CSE 554 Assignment 1 - Section 2

# Include common GPU detection and flags
include ../../../common.mk

TARGET = rms_norm_matrix_test
SOURCES = main.cu rms_norm_matrix.cu
OBJECTS = $(SOURCES:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	mkdir -p profiling_results
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	nsys profile -o profiling_results/rms_norm_matrix_$$timestamp ./$(TARGET)
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	ncu -o profiling_results/rms_norm_matrix_$$timestamp --set full ./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

cleanall: clean
	rm -f profiling_results/rms_norm_matrix.*

.PHONY: all run profile profile-nsys clean cleanall
