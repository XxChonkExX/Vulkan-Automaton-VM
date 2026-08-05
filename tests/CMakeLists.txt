cmake_minimum_required(VERSION 3.20)

project(VulkanVMTests LANGUAGES CXX)

# Link directly to the parent target (no install required for in-tree tests)
# find_package(VulkanVM REQUIRED)  # only for external consumers after install

# Basic test
add_executable(basic_test basic_test.cpp)
target_link_libraries(basic_test PRIVATE vulkan_vm)
target_include_directories(basic_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

add_test(NAME basic_test COMMAND basic_test)

# Minimal test for debugging
add_executable(minimal_test minimal_test.cpp)
target_link_libraries(minimal_test PRIVATE vulkan_vm)
target_include_directories(minimal_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

add_test(NAME minimal_test COMMAND minimal_test)

# Multi-GPU test (if multiple GPUs available)
# add_executable(multi_gpu_test multi_gpu_test.cpp)
# target_link_libraries(multi_gpu_test PRIVATE vulkan_vm)

# Stress test
# add_executable(stress_test stress_test.cpp)
# target_link_libraries(stress_test PRIVATE vulkan_vm)