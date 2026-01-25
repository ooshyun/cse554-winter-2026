# Makefile for RMS Norm Vector Implementation
# CSE 554 Assignment 1 - Section 2

# Include common GPU detection and flags
include ../../../common.mk

# Cooperative groups require compute capability 6.0+
# Note: -rdc=true (relocatable device code) required for cooperative kernel launch
NVCC_FLAGS += -rdc=true

TARGET = rms_norm_vector_test
SOURCES = main.cu rms_norm_vector.cu
OBJECTS = $(SOURCES:.cu=.o)

# FLAG for Nsight Compute profiling
PROFILE_FLAGS = -DPROFILE_NCUS

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LINK_FLAGS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile source files with profiling flags
%.profile.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(PROFILE_FLAGS) -c $< -o $@

# Profile object files
PROFILE_OBJECTS = $(SOURCES:.cu=.profile.o)

# Profile target executable
PROFILE_TARGET = $(TARGET)_profile

run: $(TARGET)
	./$(TARGET)

profile: $(PROFILE_TARGET)
	mkdir -p profiling_results
# @timestamp=$$(date +%Y%m%d_%H%M%S); \
# nsys profile -o profiling_results/rms_norm_vector_$$timestamp ./$(PROFILE_TARGET)
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	ncu -o profiling_results/rms_norm_vector_$$timestamp --set full ./$(PROFILE_TARGET)

# Build profile target
$(PROFILE_TARGET): $(PROFILE_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(PROFILE_FLAGS) -o $@ $^ $(LINK_FLAGS)

clean:
	rm -f $(OBJECTS) $(TARGET) $(PROFILE_OBJECTS) $(PROFILE_TARGET)

cleanall: clean
	rm -f profiling_results/rms_norm_vector.*

.PHONY: all run profile profile-nsys clean cleanall
