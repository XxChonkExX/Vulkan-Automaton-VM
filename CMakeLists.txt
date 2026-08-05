cmake_minimum_required(VERSION 3.20)
project(VulkanVM LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Platform detection
if(WIN32)
    set(VVM_PLATFORM_WINDOWS 1)
    add_compile_definitions(VVM_PLATFORM_WINDOWS=1 VK_USE_PLATFORM_WIN32_KHR=1)
elseif(UNIX AND NOT APPLE)
    set(VVM_PLATFORM_LINUX 1)
    add_compile_definitions(VVM_PLATFORM_LINUX=1)
endif()

# Build configuration
option(VVM_BUILD_SHARED "Build as shared library" ON)
option(VVM_BUILD_TESTS "Build tests" ON)
option(VVM_BUILD_EXAMPLES "Build examples" OFF)
option(VVM_ENABLE_VALIDATION "Enable Vulkan validation layers" ON)
option(VVM_USE_VOLK "Use Volk for Vulkan loading" ON)
option(VVM_BUILD_NETWORK "Build network/multi-node module" ON)

# Vulkan
find_package(Vulkan REQUIRED)

# Volk (optional, for dynamic Vulkan loading)
if(VVM_USE_VOLK)
    find_package(Volk QUIET)
    if(NOT Volk_FOUND)
        message(WARNING "Volk not found, using static Vulkan loading")
        set(VVM_USE_VOLK OFF)
    endif()
endif()

# Threads for sync primitives
find_package(Threads REQUIRED)

# Windows SDK include path for windows.h
if(WIN32)
    # Try to find Windows 10 SDK
    set(WIN10_SDK_VERSIONS "10.0.26100.0" "10.0.22621.0" "10.0.19041.0" "10.0.18362.0" "10.0.17763.0")
    foreach(ver ${WIN10_SDK_VERSIONS})
        if(EXISTS "C:/Program Files (x86)/Windows Kits/10/Include/${ver}/um/windows.h")
            set(WINDOWS_SDK_INCLUDE_PATH "C:/Program Files (x86)/Windows Kits/10/Include/${ver}/um")
            set(WINDOWS_SDK_INCLUDE_PATH_SHARED "C:/Program Files (x86)/Windows Kits/10/Include/${ver}/shared")
            message(STATUS "Found Windows SDK ${ver}")
            break()
        endif()
    endforeach()
    if(NOT WINDOWS_SDK_INCLUDE_PATH)
        message(WARNING "Windows SDK not found in standard locations")
    endif()
endif()

# Network module dependencies (core builds standalone; gRPC/verbs optional)
if(VVM_BUILD_NETWORK)
    find_package(Protobuf QUIET)
    find_package(gRPC QUIET)
    find_package(ibverbs QUIET)
    find_package(rdma_cm QUIET)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Library
add_library(vulkan_vm
    src/core/unified_memory_pool.cpp
    src/allocator/buddy_allocator.cpp
    src/cross_gpu/external_memory.cpp
    src/cross_gpu/multi_gpu_manager.cpp
    src/offload/host_shadow.cpp
    src/utils/vulkan_utils.cpp
    src/utils/error_handling.cpp
)

target_include_directories(vulkan_vm PRIVATE 
    ${Vulkan_INCLUDE_DIRS}
    ${WINDOWS_SDK_INCLUDE_PATH}
    ${WINDOWS_SDK_INCLUDE_PATH_SHARED}
)
target_link_libraries(vulkan_vm PRIVATE
    ${Vulkan_LIBRARIES}
    Threads::Threads
)

if(VVM_USE_VOLK)
    target_link_libraries(vulkan_vm PRIVATE Volk::Volk)
    target_compile_definitions(vulkan_vm PRIVATE VVM_USE_VOLK=1)
endif()

target_compile_definitions(vulkan_vm PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VMA_STATIC_VULKAN_FUNCTIONS=0
    VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)

if(VVM_BUILD_SHARED)
    set_target_properties(vulkan_vm PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    target_compile_definitions(vulkan_vm PRIVATE VVM_EXPORT=1)
else()
    target_compile_definitions(vulkan_vm PRIVATE VVM_STATIC=1)
endif()

# Public headers
target_include_directories(vulkan_vm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Network module
if(VVM_BUILD_NETWORK)
    add_subdirectory(src/network)
    target_link_libraries(vulkan_vm PUBLIC vulkan_vm_network)
endif()

# Install
install(TARGETS vulkan_vm
    EXPORT VulkanVMTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/vulkan_vm/
    DESTINATION include/vulkan_vm
    FILES_MATCHING PATTERN "*.hpp"
)

# Tests
if(VVM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(VVM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/VulkanVMConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    cmake/VulkanVMConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/VulkanVMConfig.cmake
    INSTALL_DESTINATION lib/cmake/VulkanVM
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/VulkanVMConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/VulkanVMConfigVersion.cmake
    DESTINATION lib/cmake/VulkanVM
)

install(EXPORT VulkanVMTargets
    FILE VulkanVMTargets.cmake
    DESTINATION lib/cmake/VulkanVM
)