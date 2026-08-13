# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.15)

# Define the project name and programming language
project(gpt2-inference LANGUAGES CXX)

# Declare an executable target and associate its source files
add_executable(gpt2-inference
    src/loader.cpp
    src/tensor.cpp
    src/ops.cpp
)

# Enforce the C++26 dialect directly
target_compile_options(gpt2-inference PRIVATE -std=c++26)

# Include paths
target_include_directories(gpt2-inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)