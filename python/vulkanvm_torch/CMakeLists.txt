cmake_minimum_required(VERSION 3.20)
project(vulkanvm_torch LANGUAGES CXX)

# Find PyTorch
find_package(Torch REQUIRED)

# Find VulkanVM
set(VULKANVM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "VulkanVM root")
find_package(VulkanVM REQUIRED PATHS ${VULKANVM_ROOT} NO_DEFAULT_PATH)

# Extension
add_library(vulkanvm_torch MODULE vulkanvm_torch.cpp)

target_link_libraries(vulkanvm_torch
    ${TORCH_LIBRARIES}
    vulkan_vm
    vulkan_vm_network
)

target_include_directories(vulkanvm_torch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
)

set_target_properties(vulkanvm_torch PROPERTIES
    PREFIX ""
    SUFFIX ""
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Install
install(TARGETS vulkanvm_torch
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)