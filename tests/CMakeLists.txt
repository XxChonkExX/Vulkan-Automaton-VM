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
add_executable(serialization_test serialization_test.cpp)
target_link_libraries(serialization_test PRIVATE vulkan_vm_network)
target_include_directories(serialization_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)
add_test(NAME serialization_test COMMAND serialization_test)

# Placement API tests (pure logic, no Vulkan device required)
add_executable(placement_test placement_test.cpp)
target_link_libraries(placement_test PRIVATE vulkan_vm)
target_include_directories(placement_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)
add_test(NAME placement_test COMMAND placement_test)

# Tensor collective operations (requires a Vulkan device; single GPU is enough)
add_executable(tensor_collective_test tensor_collective_test.cpp)
target_link_libraries(tensor_collective_test PRIVATE vulkan_vm)
target_include_directories(tensor_collective_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TEST_EXTRA_INCLUDES}
)
add_test(NAME tensor_collective_test COMMAND tensor_collective_test)

# Windows-only ND fake provider + loopback test
if(WIN32)
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