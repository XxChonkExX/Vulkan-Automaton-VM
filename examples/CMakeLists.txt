cmake_minimum_required(VERSION 3.20)

# Tensor compute example
add_executable(tensor_compute tensor_compute.cpp)
target_link_libraries(tensor_compute PRIVATE vulkan_vm)
target_include_directories(tensor_compute PRIVATE ${Vulkan_INCLUDE_DIRS})

# Network test (two-node loopback cluster)
add_executable(network_test network_test.cpp)
target_link_libraries(network_test PRIVATE vulkan_vm vulkan_vm_network)
target_include_directories(network_test PRIVATE ${Vulkan_INCLUDE_DIRS})

# Offload/reload verification test (device <-> host shadow round-trip)
add_executable(offload_test offload_test.cpp)
target_link_libraries(offload_test PRIVATE vulkan_vm)
target_include_directories(offload_test PRIVATE ${Vulkan_INCLUDE_DIRS})

# Multi-GPU example (requires multiple GPUs)
# add_executable(multi_gpu multi_gpu.cpp)
# target_link_libraries(multi_gpu PRIVATE vulkan_vm)