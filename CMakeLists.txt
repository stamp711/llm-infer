cmake_minimum_required(VERSION 3.18)
project(llm_infer LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures
# Common architectures: 70 (V100), 75 (RTX 20xx), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Collect source files
file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

# Create executable
add_executable(llm_infer ${SOURCES})

# Link CUDA libraries
target_link_libraries(llm_infer
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(llm_infer PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    )
endif()

# CUDA compiler flags
target_compile_options(llm_infer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Set output directory
set_target_properties(llm_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
