cmake_minimum_required(VERSION 3.20)

# Tensor compute example
add_executable(tensor_compute tensor_compute.cpp)
target_link_libraries(tensor_compute PRIVATE vulkan_vm)
target_include_directories(tensor_compute PRIVATE ${Vulkan_INCLUDE_DIRS})

# Multi-GPU example (requires multiple GPUs)
# add_executable(multi_gpu multi_gpu.cpp)
# target_link_libraries(multi_gpu PRIVATE vulkan_vm)