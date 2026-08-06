cmake_minimum_required(VERSION 3.20)
project(vulkanvm_onnx LANGUAGES CXX)

# Find ONNX Runtime
find_package(onnxruntime REQUIRED)

# Find VulkanVM
set(VULKANVM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "VulkanVM root")
find_package(VulkanVM REQUIRED PATHS ${VULKANVM_ROOT} NO_DEFAULT_PATH)

# Extension
add_library(vulkanvm_onnx MODULE vulkanvm_onnx.cpp)

target_link_libraries(vulkanvm_onnx
    ${onnxruntime_LIBRARIES}
    vulkan_vm
    vulkan_vm_network
)

target_include_directories(vulkanvm_onnx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${onnxruntime_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
)

set_target_properties(vulkanvm_onnx PROPERTIES
    PREFIX ""
    SUFFIX ""
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Install
install(TARGETS vulkanvm_onnx
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)