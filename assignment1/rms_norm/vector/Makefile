# Makefile for RMS Norm Vector Implementation
# CSE 554 Assignment 1 - Section 2

NVCC = nvcc
# Cooperative groups require compute capability 6.0+, using sm_75 for RTX 4070 Ti SUPER (sm_89)
# Note: -rdc=true (relocatable device code) required for cooperative kernel launch
NVCC_FLAGS = -O3 -arch=sm_89 -std=c++11 -Xptxas -v -rdc=true

TARGET = rms_norm_vector_test
SOURCES = main.cu rms_norm_vector.cu
OBJECTS = $(SOURCES:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	@echo "⚠️  Nsight Compute profiling requires GPU perf counter permissions (root access)"
	@echo "   ncu can interfere with CUDA initialization causing runtime errors"
	@echo ""
	@echo "✓ Using Nsight Systems profile instead: profiling_results/rms_norm_vector.nsys-rep"
	@echo "   View with: nsys-ui ../../../../profiling_results/rms_norm_vector.nsys-rep"
	@echo ""
	@echo "ℹ️  To profile with ncu manually (requires sudo):"
	@echo "   sudo ncu -o ../../../../profiling_results/rms_norm_vector --set full ./$(TARGET)"

profile-nsys: $(TARGET)
	nsys profile -o ../../../../profiling_results/rms_norm_vector ./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

cleanall: clean
	rm -f ../../../../profiling_results/rms_norm_vector.*

.PHONY: all run profile profile-nsys clean cleanall
