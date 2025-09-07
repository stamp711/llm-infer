cmake_minimum_required(VERSION 3.18)

# Set vcpkg toolchain if VCPKG_ROOT is set and toolchain not already specified
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING
        "Vcpkg toolchain file"
    )
endif()

project(llm_infer LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CLI11 REQUIRED)
find_package(mio REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

# Set CUDA architectures
# Common architectures: 75 (RTX 20xx), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Add libraries
add_subdirectory(tensor)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR}/vcpkg_installed/x64-linux/include")

# set_target_properties(your_target_name PROPERTIES
#     CUDA_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_BINARY_DIR}/vcpkg_installed/x64-linux/include"
# )

# Create the main library with all source files
file(
    GLOB_RECURSE LIB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_library(llm_infer_lib STATIC ${LIB_SOURCES})

set_target_properties(llm_infer_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries to llm_infer_lib
target_link_libraries(
    llm_infer_lib
    PUBLIC
        CUDA::cudart
        OpenMP::OpenMP_CXX
        mio::mio
        magic_enum::magic_enum
        nlohmann_json::nlohmann_json
        fmt::fmt
        tensor
)

# Compiler flags for library
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        llm_infer_lib
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-g
            -Wall
            -Wextra
            -O3
            -funroll-loops
            -march=native>
    )
endif()

# CUDA compiler flags for library
target_compile_options(
    llm_infer_lib
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Function to create and configure a tool
function(add_tool tool_name source_file)
    add_executable(${tool_name} ${source_file})

    target_link_libraries(${tool_name} llm_infer_lib CLI11::CLI11)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${tool_name} PRIVATE -g -Wall -Wextra -O3)
    endif()

    set_target_properties(
        ${tool_name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

# Create tools
add_tool(infer tools/infer.cpp)
add_tool(gguf_info tools/gguf_info.cpp)
add_tool(hf_info tools/hf_info.cpp)
