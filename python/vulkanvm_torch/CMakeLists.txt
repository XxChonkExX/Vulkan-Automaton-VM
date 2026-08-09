# PyTorch C++ Extension for VulkanVM
# Build with: VVM_BUILD_PYTORCH=ON

cmake_minimum_required(VERSION 3.20)
project(vulkanvm_torch LANGUAGES CXX)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Find PyTorch
find_package(Torch REQUIRED)
# OR use: set(CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}")

message(STATUS "PyTorch found: ${TORCH_VERSION}")
message(STATUS "Python found: ${Python3_VERSION}")

# Find VulkanVM
find_package(VulkanVM REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Source files
set(TORCH_SRCS
    pytorch_binding.cpp
    pytorch_tensor.cpp
    pytorch_pool.cpp
)

# Create the extension module
pybind11_add_module(vulkanvm_torch ${TORCH_SRCS})

# Link libraries
target_link_libraries(vulkanvm_torch PRIVATE
    vulkan_vm
    ${TORCH_LIBRARIES}
    pybind11::module
)

# Compile options
target_compile_options(vulkanvm_torch PRIVATE
    ${TORCH_CXX_FLAGS}
    -Wall -Wextra
)

# Set output name
set_target_properties(vulkanvm_torch PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Install
install(TARGETS vulkanvm_torch
    LIBRARY DESTINATION ${Python3_SITEARCH}/vulkanvm
    RUNTIME DESTINATION ${Python3_SITEARCH}/vulkanvm
)