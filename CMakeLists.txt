# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.15)

# Define the project name and programming language
project(gpt2-inference LANGUAGES CXX)

# Specify the C++ standard to enforce
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Declare an executable target and associate its source files
add_executable(gpt2-inference
    src/loader.cpp
    src/tensor.cpp
)

# Include paths
target_include_directories(gpt2-inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)