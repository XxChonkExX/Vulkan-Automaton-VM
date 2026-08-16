cmake_minimum_required(VERSION 3.20)
project(vulkanvm_ahb_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Android toolchain is passed via -DCMAKE_TOOLCHAIN_FILE

# Find Vulkan
find_library(VULKAN_LIB vulkan)
find_library(ANDROID_NATIVE_LIB android)
find_library(LOG_LIB log)

# Add test executable
add_executable(vulkanvm_ahb_test test_ahardwarebuffer.cpp)

target_include_directories(vulkanvm_ahb_test PRIVATE
    ${CMAKE_SOURCE_DIR}/../include
)

target_link_libraries(vulkanvm_ahb_test PRIVATE
    vulkan_vm
    vulkan_vm_network
    ${VULKAN_LIB}
    ${ANDROID_NATIVE_LIB}
    ${LOG_LIB}
)

target_compile_definitions(vulkanvm_ahb_test PRIVATE
    VK_USE_PLATFORM_ANDROID_KHR=1
    VVM_PLATFORM_ANDROID=1
)

# Install to device
set(ANDROID_TEST_DIR "/data/local/tmp")
install(TARGETS vulkanvm_ahb_test
    RUNTIME DESTINATION ${ANDROID_TEST_DIR}
)