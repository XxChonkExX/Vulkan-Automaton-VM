# Android CMake Toolchain for VulkanVM
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 ...

# Android NDK path - set this to your NDK location
if(NOT DEFINED ANDROID_NDK)
    set(ANDROID_NDK "C:/Users/mikeh/AppData/Local/Android/Sdk/ndk/27.0.12077973" CACHE PATH "Android NDK path")
endif()

# Android ABI (arm64-v8a, armeabi-v7a, x86, x86_64)
if(NOT DEFINED ANDROID_ABI)
    set(ANDROID_ABI "arm64-v8a" CACHE STRING "Android ABI")
endif()

# Android platform version (android-24, android-34, etc.)
if(NOT DEFINED ANDROID_PLATFORM)
    set(ANDROID_PLATFORM "android-34" CACHE STRING "Android platform version")
endif()

# Android NDK toolchain file
set(CMAKE_ANDROID_NDK ${ANDROID_NDK})
set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION "clang")
set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})
set(CMAKE_ANDROID_NATIVE_API_LEVEL ${ANDROID_PLATFORM})

# Use NDK's CMake toolchain
set(CMAKE_TOOLCHAIN_FILE "${ANDROID_NDK}/build/cmake/android.toolchain.cmake")

# Android-specific settings
set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION ${ANDROID_PLATFORM})
set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})
set(CMAKE_ANDROID_NATIVE_API_LEVEL ${ANDROID_PLATFORM})
set(CMAKE_ANDROID_STL_TYPE c++_shared)

# Android-specific compile definitions
add_compile_definitions(
    VVM_PLATFORM_ANDROID=1
    VK_USE_PLATFORM_ANDROID_KHR=1
)

# Android-specific linker flags
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -latomic -lm")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -latomic -lm")

# Android-specific include directories
include_directories(${ANDROID_NDK}/sysroot/usr/include)
include_directories(${ANDROID_NDK}/sysroot/usr/include/${CMAKE_ANDROID_ARCH_ABI})

# Find Android log library
find_library(ANDROID_LOG_LIB log)
find_library(ANDROID_EGL_LIB EGL)
find_library(ANDROID_GLES_LIB GLESv3)
find_library(ANDROID_NATIVE_WINDOW_LIB android)
find_library(ANDROID_NATIVE_ACTIVITY_LIB native_activity)

# Android-specific libraries
set(ANDROID_LIBS ${ANDROID_LOG_LIB} ${ANDROID_EGL_LIB} ${ANDROID_GLES_LIB} ${ANDROID_NATIVE_WINDOW_LIB} ${ANDROID_NATIVE_ACTIVITY_LIB})

# Set Android-specific Vulkan features
set(VVM_ANDROID_HARDWARE_BUFFER 1)
set(VVM_ANDROID_EXTERNAL_MEMORY 1)

message(STATUS "Android toolchain configured:")
message(STATUS "  NDK: ${ANDROID_NDK}")
message(STATUS "  ABI: ${ANDROID_ABI}")
message(STATUS "  Platform: ${ANDROID_PLATFORM}")
message(STATUS "  Toolchain: ${CMAKE_TOOLCHAIN_FILE}")