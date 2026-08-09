// ONNX Runtime C++ Integration for VulkanVM
// Build with: VVM_BUILD_ONNX=ON

cmake_minimum_required(VERSION 3.20)
project(vulkanvm_onnx LANGUAGES CXX)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Find ONNX Runtime
# ONNX Runtime provides CMake config if installed via vcpkg/conan or built from source
find_package(onnxruntime REQUIRED)

# Find VulkanVM
find_package(VulkanVM REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Source files
set(ONNX_SRCS
    onnx_binding.cpp
    onnx_execution_provider.cpp
)

# Create the extension module
pybind11_add_module(vulkanvm_onnx ${ONNX_SRCS})

# Link libraries
target_link_libraries(vulkanvm_onnx PRIVATE
    vulkan_vm
    onnxruntime
    pybind11::module
)

# Compile options
target_compile_options(vulkanvm_onnx PRIVATE -Wall -Wextra)

# Set output name
set_target_properties(vulkanvm_onnx PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Install
install(TARGETS vulkanvm_onnx
    LIBRARY DESTINATION ${Python3_SITEARCH}/vulkanvm
    RUNTIME DESTINATION ${Python3_SITEARCH}/vulkanvm
)