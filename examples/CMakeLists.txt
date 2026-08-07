cmake_minimum_required(VERSION 3.20)

# Windows SDK + MSVC include paths (needed for windows.h, stddef.h, vcruntime.h, etc.)
set(EXAMPLE_EXTRA_INCLUDES ${Vulkan_INCLUDE_DIRS})
if(WIN32)
    list(APPEND EXAMPLE_EXTRA_INCLUDES
        ${WINDOWS_SDK_UM}
        ${WINDOWS_SDK_SHARED}
        ${WINDOWS_SDK_UCRT}
        ${MSVC_INCLUDE_PATH}
    )
endif()

# Tensor compute example
add_executable(tensor_compute tensor_compute.cpp)
target_link_libraries(tensor_compute PRIVATE vulkan_vm)
target_include_directories(tensor_compute PRIVATE ${EXAMPLE_EXTRA_INCLUDES})

# Network test (two-node loopback cluster)
add_executable(network_test network_test.cpp)
target_link_libraries(network_test PRIVATE vulkan_vm vulkan_vm_network)
target_include_directories(network_test PRIVATE ${EXAMPLE_EXTRA_INCLUDES})

# Tensor network test (GPU-to-GPU VRAM share cross-vendor over TCP) - DISABLED (WIP)
# add_executable(tensor_network_test tensor_network_test.cpp)
# target_link_libraries(tensor_network_test PRIVATE vulkan_vm vulkan_vm_network)
# target_include_directories(tensor_network_test PRIVATE ${EXAMPLE_EXTRA_INCLUDES})

# Offload/reload verification test (device <-> host shadow round-trip)
add_executable(offload_test offload_test.cpp)
target_link_libraries(offload_test PRIVATE vulkan_vm)
target_include_directories(offload_test PRIVATE ${EXAMPLE_EXTRA_INCLUDES})

# Model registry test (publish/fetch model weights)
add_executable(model_registry_test model_registry_test.cpp)
target_link_libraries(model_registry_test PRIVATE vulkan_vm vulkan_vm_network)
target_include_directories(model_registry_test PRIVATE ${EXAMPLE_EXTRA_INCLUDES})

# Multi-GPU example (requires multiple GPUs)
# add_executable(multi_gpu multi_gpu.cpp)
# target_link_libraries(multi_gpu PRIVATE vulkan_vm)