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

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Library
add_library(vulkan_vm
    src/core/unified_memory_pool.cpp
    src/allocator/buddy_allocator.cpp
    src/allocator/block_manager.cpp
    src/cross_gpu/external_memory.cpp
    src/cross_gpu/multi_gpu_manager.cpp
    src/offload/host_shadow.cpp
    src/offload/migration.cpp
    src/utils/vulkan_utils.cpp
    src/utils/error_handling.cpp
)

target_include_directories(vulkan_vm PRIVATE ${Vulkan_INCLUDE_DIRS})
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