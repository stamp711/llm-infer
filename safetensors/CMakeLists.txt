cmake_minimum_required(VERSION 3.18)
project(safetensors)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(mio REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(safetensors STATIC
    src/safetensors.cpp
)

target_include_directories(safetensors PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(safetensors PUBLIC
    mio::mio
    nlohmann_json::nlohmann_json
)