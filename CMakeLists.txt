cmake_minimum_required(VERSION 3.18)

# Set vcpkg toolchain if VCPKG_ROOT is set and toolchain not already specified
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

project(llm_infer LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CLI11 REQUIRED)
find_package(mio REQUIRED)
find_package(magic_enum CONFIG REQUIRED)

# Set CUDA architectures
# Common architectures: 75 (RTX 20xx), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# Add libraries
add_subdirectory(safetensors)
add_subdirectory(tensor)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Collect source files
file(
    GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

# Create executable
add_executable(llm_infer ${SOURCES})

set_target_properties(llm_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(
    llm_infer
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    OpenMP::OpenMP_CXX
    CLI11::CLI11
    mio::mio
    magic_enum::magic_enum
    safetensors
    tensor
)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        llm_infer
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-g -Wall -Wextra -O3>
    )
endif()

# CUDA compiler flags
target_compile_options(
    llm_infer
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Set output directory
set_target_properties(
    llm_infer
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
