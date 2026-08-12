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

# Remote network test (Windows client -> Evo-X2 server over LAN)
add_executable(network_remote_test network_remote_test.cpp)
target_link_libraries(network_remote_test PRIVATE vulkan_vm vulkan_vm_network)
target_include_directories(network_remote_test PRIVATE ${EXAMPLE_EXTRA_INCLUDES})

# Tensor network test (GPU-to-GPU VRAM share cross-vendor over TCP) - DISABLED (WIP)
# add_executable(tensor_network_test tensor_network_test.cpp)
# target_link_libraries(tensor_network_test PRIVATE vulkan_vm vulkan_vm_network)
# target_include_directories(tensor_network_test PRIVATE ${EXAMPLE_EXTRA_INCLUDES})

# Cross-machine tensor network tests (client <-> server over TCP/RDMA).
# Client runs on Windows or Linux; server is Linux-only (verbs transport).
add_executable(tensor_client_test tensor_client_test.cpp)
target_link_libraries(tensor_client_test PRIVATE vulkan_vm vulkan_vm_network)
target_include_directories(tensor_client_test PRIVATE ${EXAMPLE_EXTRA_INCLUDES})

if(NOT WIN32)
    # Linux server test (requires verbs/RDMA for the GPU-direct path)
    add_executable(tensor_server_test tensor_server_test.cpp)
    target_link_libraries(tensor_server_test PRIVATE vulkan_vm vulkan_vm_network)
    target_include_directories(tensor_server_test PRIVATE ${EXAMPLE_EXTRA_INCLUDES})
endif()

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