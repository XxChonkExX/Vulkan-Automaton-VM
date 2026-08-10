cmake_minimum_required(VERSION 3.20)

# Set Windows SDK version BEFORE project() to avoid defaulting to unavailable 10.0.28000.0
if(WIN32)
    set(CMAKE_SYSTEM_VERSION "10.0.22621.0" CACHE STRING "Windows SDK version" FORCE)
    set(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION "10.0.22621.0" CACHE STRING "Windows SDK version" FORCE)
endif()

project(VulkanVM LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Platform detection
if(WIN32)
    set(VVM_PLATFORM_WINDOWS 1)
    add_compile_definitions(VVM_PLATFORM_WINDOWS=1 VK_USE_PLATFORM_WIN32_KHR=1)
elseif(APPLE)
    set(VVM_PLATFORM_MACOS 1)
    add_compile_definitions(VVM_PLATFORM_MACOS=1 VK_USE_PLATFORM_MACOS_MVK=1)
elseif(UNIX)
    set(VVM_PLATFORM_LINUX 1)
    add_compile_definitions(VVM_PLATFORM_LINUX=1)
endif()

# Build configuration
option(VVM_BUILD_SHARED "Build as shared library" ON)
option(VVM_BUILD_TESTS "Build tests" ON)
option(VVM_BUILD_EXAMPLES "Build examples (includes integration tests)" ON)
option(VVM_ENABLE_VALIDATION "Enable Vulkan validation layers" ON)
option(VVM_USE_VOLK "Use Volk for Vulkan loading" ON)
option(VVM_BUILD_NETWORK "Build network/multi-node module" ON)
option(VVM_BUILD_PYTORCH "Build PyTorch C++ extension" OFF)
option(VVM_BUILD_ONNX "Build ONNX Runtime integration" OFF)

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

# Python3 for PyTorch/ONNX integration tests
if(VVM_BUILD_PYTORCH OR VVM_BUILD_ONNX)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
endif()

# Windows SDK and MSVC include paths
if(WIN32)
    # Find Windows SDK paths
    set(WIN10_SDK_VERSIONS "10.0.26100.0" "10.0.22621.0" "10.0.19041.0" "10.0.18362.0" "10.0.17763.0")
    foreach(ver ${WIN10_SDK_VERSIONS})
        if(EXISTS "C:/Program Files (x86)/Windows Kits/10/Include/${ver}/um/windows.h")
            set(WINDOWS_SDK_UM     "C:/Program Files (x86)/Windows Kits/10/Include/${ver}/um")
            set(WINDOWS_SDK_SHARED "C:/Program Files (x86)/Windows Kits/10/Include/${ver}/shared")
            set(WINDOWS_SDK_UCRT   "C:/Program Files (x86)/Windows Kits/10/Include/${ver}/ucrt")
            message(STATUS "Found Windows SDK ${ver}")
            break()
        endif()
    endforeach()
    if(NOT WINDOWS_SDK_UM)
        if(DEFINED ENV{WindowsSdkDir})
            set(WINDOWS_SDK_UM     "$ENV{WindowsSdkDir}um")
            set(WINDOWS_SDK_SHARED "$ENV{WindowsSdkDir}shared")
            set(WINDOWS_SDK_UCRT   "$ENV{WindowsSdkDir}ucrt")
            message(STATUS "Using Windows SDK from WindowsSdkDir environment variable")
        else()
            message(WARNING "Windows SDK not found - build may fail without proper include paths")
        endif()
    endif()

    # Explicitly set Windows SDK version for MSBuild to avoid defaulting to unavailable 10.0.28000.0
    if(WINDOWS_SDK_UM)
        string(REGEX MATCH "10\\.0\\.[0-9]+" WINDOWS_SDK_VERSION "${WINDOWS_SDK_UM}")
        set(CMAKE_SYSTEM_VERSION "${WINDOWS_SDK_VERSION}" CACHE STRING "Windows SDK version" FORCE)
        set(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION "${WINDOWS_SDK_VERSION}" CACHE STRING "Windows SDK version" FORCE)
        message(STATUS "Setting CMAKE_SYSTEM_VERSION=${CMAKE_SYSTEM_VERSION} and CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION=${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
    endif()

    # Find Windows SDK library path (for ws2_32.lib, kernel32.lib, etc.)
    foreach(ver ${WIN10_SDK_VERSIONS})
        if(EXISTS "C:/Program Files (x86)/Windows Kits/10/Lib/${ver}/um/x64/ws2_32.lib")
            set(WINDOWS_SDK_LIB_PATH "C:/Program Files (x86)/Windows Kits/10/Lib/${ver}/um/x64")
            set(WINDOWS_SDK_UCRT_LIB_PATH "C:/Program Files (x86)/Windows Kits/10/Lib/${ver}/ucrt/x64")
            message(STATUS "Found Windows SDK libraries: ${WINDOWS_SDK_LIB_PATH}")
            break()
        elseif(EXISTS "C:/Program Files (x86)/Windows Kits/10/Lib/${ver}/um/x86/ws2_32.lib")
            set(WINDOWS_SDK_LIB_PATH "C:/Program Files (x86)/Windows Kits/10/Lib/${ver}/um/x86")
            set(WINDOWS_SDK_UCRT_LIB_PATH "C:/Program Files (x86)/Windows Kits/10/Lib/${ver}/ucrt/x86")
            message(STATUS "Found Windows SDK libraries: ${WINDOWS_SDK_LIB_PATH}")
            break()
        endif()
    endforeach()

    # Find MSVC library path (needed for vcruntime.lib, msvcprt.lib, etc.)
    # Try to auto-detect from the compiler path
    if(CMAKE_CXX_COMPILER)
        get_filename_component(MSVC_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(MSVC_TOOLSET_DIR "${MSVC_BIN_DIR}/../../.." ABSOLUTE)
        if(EXISTS "${MSVC_TOOLSET_DIR}/include/vcruntime.h")
            set(MSVC_INCLUDE_PATH "${MSVC_TOOLSET_DIR}/include")
            # Determine architecture subdirectory
            if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                set(MSVC_LIB_ARCH "x64")
            else()
                set(MSVC_LIB_ARCH "x86")
            endif()
            if(EXISTS "${MSVC_TOOLSET_DIR}/lib/${MSVC_LIB_ARCH}")
                set(MSVC_LIBRARY_PATH "${MSVC_TOOLSET_DIR}/lib/${MSVC_LIB_ARCH}")
                message(STATUS "Found MSVC: ${MSVC_TOOLSET_DIR}")
            endif()
        endif()
    endif()
    if(NOT MSVC_INCLUDE_PATH)
        # Search common MSVC install locations
        file(GLOB MSVC_VERSIONS
            "C:/Program Files/Microsoft Visual Studio/2022/*/VC/Tools/MSVC/*"
            "C:/Program Files (x86)/Microsoft Visual Studio/2022/*/VC/Tools/MSVC/*"
            "C:/Program Files/Microsoft Visual Studio/2019/*/VC/Tools/MSVC/*"
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/*/VC/Tools/MSVC/*"
        )
        foreach(dir ${MSVC_VERSIONS})
            if(EXISTS "${dir}/include/vcruntime.h")
                set(MSVC_INCLUDE_PATH "${dir}/include")
                message(STATUS "Found MSVC includes: ${dir}/include")
                break()
            endif()
        endforeach()
    endif()
    if(NOT MSVC_INCLUDE_PATH)
        message(WARNING "MSVC include path not found - build may fail without vcruntime.h")
    endif()
endif()

# Network module dependencies (core builds standalone; gRPC/verbs/TLS optional)
if(VVM_BUILD_NETWORK)
    find_package(Protobuf QUIET)
    find_package(gRPC QUIET)
    find_package(ibverbs QUIET)
    find_package(rdma_cm QUIET)
    # Level Zero (Intel GPU-direct on Windows)
    if(WIN32)
        find_package(LevelZero QUIET)
        if(LevelZero_FOUND)
            add_compile_definitions(VVM_HAS_LEVEL_ZERO=1)
            message(STATUS "Level Zero found - Intel GPU-direct enabled")
        else()
            message(WARNING "Level Zero not found - Intel GPU-direct disabled")
        endif()
    endif()
    # TLS support via OpenSSL
    find_package(OpenSSL QUIET)
    if(OPENSSL_FOUND)
        add_compile_definitions(VVM_NETWORK_HAS_TLS=1)
        message(STATUS "OpenSSL found - TLS support enabled")
    else()
        message(WARNING "OpenSSL not found - TLS support disabled")
    endif()
endif()

# Include directories (target-based)
# Library
add_library(vulkan_vm
    src/core/unified_memory_pool.cpp
    src/core/sparse_memory.cpp
    src/allocator/buddy_allocator.cpp
    src/cross_gpu/external_memory.cpp
    src/cross_gpu/multi_gpu_manager.cpp
    src/offload/host_shadow.cpp
    src/utils/vulkan_utils.cpp
    src/utils/error_handling.cpp
    src/placement/shard_placer.cpp
    src/placement/placement_executor.cpp
)

target_include_directories(vulkan_vm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${Vulkan_INCLUDE_DIRS}
        ${WINDOWS_SDK_UM}
        ${WINDOWS_SDK_SHARED}
        ${WINDOWS_SDK_UCRT}
        ${MSVC_INCLUDE_PATH}
)
target_link_libraries(vulkan_vm PRIVATE
    ${Vulkan_LIBRARIES}
    Threads::Threads
)

# Add Windows SDK + MSVC library paths to linker search
if(WIN32)
    if(WINDOWS_SDK_LIB_PATH)
        target_link_directories(vulkan_vm PUBLIC ${WINDOWS_SDK_LIB_PATH})
    endif()
    if(WINDOWS_SDK_UCRT_LIB_PATH)
        target_link_directories(vulkan_vm PUBLIC ${WINDOWS_SDK_UCRT_LIB_PATH})
    endif()
    if(MSVC_LIBRARY_PATH)
        target_link_directories(vulkan_vm PUBLIC ${MSVC_LIBRARY_PATH})
    endif()
endif()

if(VVM_USE_VOLK)
    target_link_libraries(vulkan_vm PRIVATE Volk::Volk)
    target_compile_definitions(vulkan_vm PRIVATE VVM_USE_VOLK=1)
endif()

target_compile_definitions(vulkan_vm PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
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

# Tensor Transport module
option(VVM_BUILD_TENSOR_TRANSPORT "Build tensor transport module" ON)
if(VVM_BUILD_TENSOR_TRANSPORT AND VVM_BUILD_NETWORK)
    add_subdirectory(src/tensor)
    target_link_libraries(vulkan_vm PUBLIC vulkan_vm_tensor)
endif()

# Shader compilation for tensor transport
if(VVM_BUILD_TENSOR_TRANSPORT)
    find_program(GLSLANG_VALIDATOR glslangValidator
        PATHS "$ENV{VULKAN_SDK}/Bin" "C:/VulkanSDK/*/Bin"
        DOC "glslangValidator for SPIR-V compilation")
    
    if(GLSLANG_VALIDATOR)
        # Compile layout conversion shader
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/shaders/layout_conversion.spv
            COMMAND ${GLSLANG_VALIDATOR} -V ${CMAKE_CURRENT_SOURCE_DIR}/shaders/layout_conversion.comp
                        -o ${CMAKE_CURRENT_BINARY_DIR}/shaders/layout_conversion.spv
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/layout_conversion.comp
            COMMENT "Compiling layout conversion shader to SPIR-V"
            VERBATIM
        )
        add_custom_target(compile_shaders ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/shaders/layout_conversion.spv)
        message(STATUS "Shader compilation enabled with glslangValidator: ${GLSLANG_VALIDATOR}")
    else()
        message(WARNING "glslangValidator not found - layout conversion shaders will not be compiled. Install Vulkan SDK with glslangValidator for full tensor transport functionality.")
    endif()
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

# Python bindings
if(VVM_BUILD_PYTORCH)
    add_subdirectory(python/vulkanvm_torch)
endif()

if(VVM_BUILD_ONNX)
    add_subdirectory(python/vulkanvm_onnx)
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