cmake_minimum_required(VERSION 3.10)

project(CSE554 LANGUAGES CXX CUDA)

# Language standards and compiler settings
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# default to RTX 6000 (Turing = 75)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)
endif()

# Base flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG -Xcompiler=-fPIE -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing")

# Add SM90a codegen only when targeting Hopper "a" variant.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "(^|;)(90a|90)($|;)")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code=arch=compute_90a,code=[compute_90a,sm_90a]")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(
  rms_norm_matrix
  rms_norm_matrix.cu
  main.cu
)


