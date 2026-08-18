# PyTorch C++ Extension for VulkanVM
# Build with: VVM_BUILD_PYTORCH=ON
#
# NOTE: vulkanvm_pool_test (the Chonk allocator binding) has NO torch
# dependency and builds unconditionally. The full vulkanvm_torch extension
# is only built when torch's CMake config is usable (VVM_BUILD_TORCH_EXT).
# On this machine the ROCm torch wheel requires miopen/hipfft/hipsparse
# configs that are not installed, so the torch extension is skipped and the
# pool test is built directly (scripts/build_pool_test.sh).

cmake_minimum_required(VERSION 3.20)

# Set Windows SDK version BEFORE project() to avoid defaulting to unavailable 10.0.28000.0
if(WIN32)
    set(CMAKE_SYSTEM_VERSION "10.0.26100.0" CACHE STRING "Windows SDK version" FORCE)
    set(CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION "10.0.26100.0" CACHE STRING "Windows SDK version" FORCE)
    set(CMAKE_RC_COMPILER "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe" CACHE STRING "Windows RC compiler" FORCE)
    set(CMAKE_MT "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/mt.exe" CACHE STRING "Windows Manifest Tool" FORCE)
endif()

project(vulkanvm_torch LANGUAGES CXX)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Use pip-installed pybind11 (newer, MSVC-compatible) instead of conda's
# The pip version lives at: <python>/site-packages/pybind11/share/cmake/pybind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

message(STATUS "Python found: ${Python3_VERSION}")
message(STATUS "pybind11 cmake dir: ${pybind11_DIR}")

# Find VulkanVM - use local targets when building as part of main project
if(TARGET vulkan_vm)
    message(STATUS "Using in-tree VulkanVM targets")
else()
    find_package(VulkanVM REQUIRED)
endif()

# Vulkan SDK include directory
set(VULKAN_SDK_INCLUDE "C:/VulkanSDK/1.4.304.1/Include")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include ${VULKAN_SDK_INCLUDE})

# Pool test module sources (no torch needed - built directly via
# scripts/build_pool_test.sh; also available as a CMake target here)
set(POOL_TEST_SRCS
    _pool_test_module.cpp
)

# Create pool test module
pybind11_add_module(vulkanvm_pool_test ${POOL_TEST_SRCS})

target_link_libraries(vulkanvm_pool_test PRIVATE
    vulkan_vm
    pybind11::module
)

target_include_directories(vulkanvm_pool_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(vulkanvm_pool_test PRIVATE -DVVM_PLATFORM_LINUX=1)

# Force C++20 (VulkanVM library uses C++20, we must match)
set_target_properties(vulkanvm_pool_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Set output name
set_target_properties(vulkanvm_pool_test PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Full torch extension - only when torch's CMake config is usable
option(VVM_BUILD_TORCH_EXT "Build the full vulkanvm_torch extension (requires torch CMake)" OFF)
if(VVM_BUILD_TORCH_EXT)
    find_package(Torch REQUIRED)

    message(STATUS "PyTorch found: ${TORCH_VERSION}")

    # Source files — JIT-style extension built via setup.py uses vulkanvm_torch.cpp.
    # The richer pybind11 binding (pytorch_binding.cpp) is built separately when
    # VVM_BUILD_FULL_BINDING=ON.
    set(TORCH_SRCS
        vulkanvm_torch.cpp
        vulkanvm_autograd_bind.cpp
    )

    # Create the extension module
    pybind11_add_module(vulkanvm_torch ${TORCH_SRCS})

    target_link_libraries(vulkanvm_torch PRIVATE
        vulkan_vm
        ${TORCH_LIBRARIES}
        pybind11::module
    )

    target_include_directories(vulkanvm_torch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if(MSVC)
        target_compile_options(vulkanvm_torch PRIVATE
            ${TORCH_CXX_FLAGS}
            /W4
            /std:c++20
            /permissive-
            /Zc:preprocessor
            /Zc:externC
            /Zc:twoPhase-
            /Zc:__cplusplus
        )
    else()
        target_compile_options(vulkanvm_torch PRIVATE
            ${TORCH_CXX_FLAGS}
            -Wall -Wextra
            -std=c++20
        )
    endif()

    set_target_properties(vulkanvm_torch PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        PREFIX ""
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )

    install(TARGETS vulkanvm_torch
        LIBRARY DESTINATION ${Python3_SITEARCH}/vulkanvm
        RUNTIME DESTINATION ${Python3_SITEARCH}/vulkanvm
    )
endif()