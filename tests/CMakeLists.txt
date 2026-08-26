cmake_minimum_required(VERSION 3.20)

project(VulkanVMTests LANGUAGES CXX)

# Link directly to the parent target (no install required for in-tree tests)
# find_package(VulkanVM REQUIRED)  # only for external consumers after install

# Windows SDK + MSVC include paths (needed for windows.h, stddef.h, vcruntime.h, etc.)
set(TEST_EXTRA_INCLUDES ${Vulkan_INCLUDE_DIRS})
if(WIN32)
    list(APPEND TEST_EXTRA_INCLUDES
        ${WINDOWS_SDK_UM}
        ${WINDOWS_SDK_SHARED}
        ${WINDOWS_SDK_UCRT}
        ${MSVC_INCLUDE_PATH}
    )
endif()

# Basic test
add_executable(basic_test basic_test.cpp)
target_link_libraries(basic_test PRIVATE vulkan_vm)
target_include_directories(basic_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)
add_test(NAME basic_test COMMAND basic_test)

# Minimal test for debugging
add_executable(minimal_test minimal_test.cpp)
target_link_libraries(minimal_test PRIVATE vulkan_vm)
target_include_directories(minimal_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)
add_test(NAME minimal_test COMMAND minimal_test)

# CPU-only buddy allocator tests (no Vulkan device required)
add_executable(buddy_test buddy_test.cpp)
target_link_libraries(buddy_test PRIVATE vulkan_vm)
target_include_directories(buddy_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)
add_test(NAME buddy_test COMMAND buddy_test)

# CPU-only Chonk slab allocator invariants + fuzz (audit Priority 3).
# No Vulkan/HIP - the slab core is provider-agnostic.
add_executable(chonk_slab_test chonk_slab_test.cpp
    ${CMAKE_SOURCE_DIR}/python/vulkanvm_torch/allocator/chonk_slab.cpp
)
target_include_directories(chonk_slab_test PRIVATE
    ${CMAKE_SOURCE_DIR}
)
add_test(NAME chonk_slab_test COMMAND chonk_slab_test)

# CPU-only external-handle ownership tests (dup/multi-import, consume semantics)
add_executable(external_handle_test external_handle_test.cpp)
target_link_libraries(external_handle_test PRIVATE vulkan_vm)
target_include_directories(external_handle_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)
add_test(NAME external_handle_test COMMAND external_handle_test)

# Sparse / residency virtual memory test (requires sparseResidencyBuffer feature)
add_executable(sparse_test sparse_test.cpp)
target_link_libraries(sparse_test PRIVATE vulkan_vm)
target_include_directories(sparse_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)
add_test(NAME sparse_test COMMAND sparse_test)

# Python integration tests (require built extensions)
if(VVM_BUILD_PYTORCH)
    add_test(NAME pytorch_integration_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pytorch_integration_test.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

if(VVM_BUILD_ONNX)
    add_test(NAME onnx_integration_test
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/onnx_integration_test.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

# Multi-GPU P2P test (requires >= 2 physical devices; skipped on limited HW)
add_executable(multi_gpu_test multi_gpu_test.cpp)
target_link_libraries(multi_gpu_test PRIVATE vulkan_vm)
target_include_directories(multi_gpu_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)

# Stress test
# add_executable(stress_test stress_test.cpp)
# target_link_libraries(stress_test PRIVATE vulkan_vm)

# CPU-only wire-format regression tests (no Vulkan device required)
if(VVM_BUILD_NETWORK)
add_executable(serialization_test serialization_test.cpp)
target_link_libraries(serialization_test PRIVATE vulkan_vm_network)
target_include_directories(serialization_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)
add_test(NAME serialization_test COMMAND serialization_test)
endif()

# Placement API tests (pure logic, no Vulkan device required)
add_executable(placement_test placement_test.cpp)
target_link_libraries(placement_test PRIVATE vulkan_vm)
target_include_directories(placement_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)
add_test(NAME placement_test COMMAND placement_test)

# Tensor collective operations (requires a Vulkan device; single GPU is enough)
if(VVM_BUILD_TENSOR_TRANSPORT AND VVM_BUILD_NETWORK)
add_executable(tensor_collective_test tensor_collective_test.cpp)
target_link_libraries(tensor_collective_test PRIVATE vulkan_vm vulkan_vm_tensor)
target_include_directories(tensor_collective_test PRIVATE ${CMAKE_SOURCE_DIR}/include ${TEST_EXTRA_INCLUDES})
add_test(NAME tensor_collective_test COMMAND tensor_collective_test)
endif()

# Dtype conversion tests (CPU-only, but uses tensor-layer utilities from the
# Compute module - TensorShape/readElem/writeElem)
if(VVM_BUILD_TENSOR_TRANSPORT AND VVM_BUILD_NETWORK)
add_executable(dtype_conversion_test dtype_conversion_test.cpp)
target_link_libraries(dtype_conversion_test PRIVATE vulkan_vm vulkan_vm_tensor)
target_include_directories(dtype_conversion_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)
add_test(NAME dtype_conversion_test COMMAND dtype_conversion_test)
endif()

# Windows-only NDK persistent pin ref-count test (requires network layer)
if(WIN32 AND VVM_BUILD_NETWORK)
    add_executable(persistent_pin_test persistent_pin_test.cpp)
    target_link_libraries(persistent_pin_test PRIVATE
        vulkan_vm_network
        ${CMAKE_DL_LIBS}
    )
    target_include_directories(persistent_pin_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${TEST_EXTRA_INCLUDES}
    )
    # Copy fake provider DLL next to test exe for runtime loading
    add_custom_command(TARGET persistent_pin_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:ndfake_provider>
            $<TARGET_FILE_DIR:persistent_pin_test>/
    )
    add_test(NAME persistent_pin_test COMMAND persistent_pin_test)
endif()

# Windows-only ND fake provider + loopback test (requires network layer)
if(WIN32 AND VVM_BUILD_NETWORK)
    # Fake ND provider DLL (implements IND2 SPI on host RAM)
    add_library(ndfake_provider SHARED
        ndfake/ndfake_provider.cpp
    )
    target_include_directories(ndfake_provider PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/ndk
        ${TEST_EXTRA_INCLUDES}
    )
    target_compile_definitions(ndfake_provider PRIVATE NDFAKE_BUILD)
    target_link_libraries(ndfake_provider PRIVATE ws2_32)
    # DllGetClassObject is predeclared by combaseapi.h (via windows.h), so the
    # definition cannot carry a second linkage spec; force the export instead.
    if(MSVC)
        target_link_options(ndfake_provider PRIVATE "/EXPORT:DllGetClassObject")
    endif()

    # Loopback test: exercises NdkRdmaTransport via fake provider
    # Requires VVM_ND_PROVIDER_DLL env var pointing to ndfake_provider.dll at runtime
    add_executable(ndk_transport_test
        ndfake/ndk_transport_test.cpp
    )
    target_link_libraries(ndk_transport_test PRIVATE
        vulkan_vm_network
        ${CMAKE_DL_LIBS}
    )
    target_include_directories(ndk_transport_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${TEST_EXTRA_INCLUDES}
    )
    # Copy fake provider DLL next to test exe for runtime loading
    add_custom_command(TARGET ndk_transport_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:ndfake_provider>
            $<TARGET_FILE_DIR:ndk_transport_test>/
    )
    add_test(NAME ndk_transport_test COMMAND ndk_transport_test)
endif()

# Multi-vendor RDMA test (cross-vendor external memory round-trip)
# Requires >= 2 GPUs from different vendors with compatible external memory handle types.
# On Linux: uses Verbs transport (IBVERBS + RDMA_CM)
# On Windows: uses ND transport (WIN32)
# Skips gracefully if hardware/transport not available.
if(IBVERBS_FOUND AND RDMA_CM_FOUND)
    message(STATUS "Adding multi-vendor RDMA test (Linux/Verbs)")
    add_executable(multi_vendor_rdma_test multi_vendor_rdma_test.cpp)
    target_link_libraries(multi_vendor_rdma_test PRIVATE
        vulkan_vm_network
        ${CMAKE_DL_LIBS}
    )
    target_include_directories(multi_vendor_rdma_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${TEST_EXTRA_INCLUDES}
    )
    add_test(NAME multi_vendor_rdma_test COMMAND multi_vendor_rdma_test)
elseif(WIN32 AND VVM_BUILD_NETWORK)
    message(STATUS "Adding multi-vendor RDMA test (Windows/ND)")
    add_executable(multi_vendor_rdma_test multi_vendor_rdma_test.cpp)
    target_link_libraries(multi_vendor_rdma_test PRIVATE
        vulkan_vm_network
        ${CMAKE_DL_LIBS}
    )
    target_include_directories(multi_vendor_rdma_test PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${TEST_EXTRA_INCLUDES}
    )
    add_test(NAME multi_vendor_rdma_test COMMAND multi_vendor_rdma_test)
endif()

# UDP mini-verb transport test (requires the network layer).
if(VVM_BUILD_NETWORK)
add_executable(udp_verb_test udp_verb_test.cpp)
target_link_libraries(udp_verb_test PRIVATE
    vulkan_vm_network
    ${CMAKE_DL_LIBS}
)
target_include_directories(udp_verb_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)
add_test(NAME udp_verb_test COMMAND udp_verb_test)
endif()