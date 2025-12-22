cmake_minimum_required(VERSION 3.10)
project(NanoNet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add ONNX subdirectory (this handles protobuf generation and header paths)
add_subdirectory(onnx)

# Project headers
include_directories(
    BEFORE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(test_mnist
    tests/test_mnist.cpp
    src/tensor.cpp
    src/ops.cpp
    src/matmul.cpp
    src/onnx_loader.cpp
)

# Link against ONNX (this also links Protobuf)
target_link_libraries(test_mnist PRIVATE onnx)
