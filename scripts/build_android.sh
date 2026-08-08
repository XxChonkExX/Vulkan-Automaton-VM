#!/bin/bash
# Android build script for VulkanVM
# Usage: ./build_android.sh [abi] [platform] [build_type]

set -e

# Default values
ABI=${1:-arm64-v8a}
PLATFORM=${2:-android-34}
BUILD_TYPE=${3:-Release}
BUILD_DIR="build_android_${ABI}"

# Android SDK/NDK paths - adjust if needed
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}"
export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-$ANDROID_SDK_ROOT/ndk/27.0.12077973}"

echo "Building VulkanVM for Android"
echo "  ABI: $ABI"
echo "  Platform: $PLATFORM"
echo "  Build Type: $BUILD_TYPE"
echo "  NDK: $ANDROID_NDK_ROOT"

# Check NDK exists
if [ ! -d "$ANDROID_NDK_ROOT" ]; then
    echo "ERROR: Android NDK not found at $ANDROID_NDK_ROOT"
    echo "Set ANDROID_NDK_ROOT environment variable"
    exit 1
fi

# Configure
cmake -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE=cmake/android.toolchain.cmake \
    -DANDROID_ABI="$ABI" \
    -DANDROID_PLATFORM="$PLATFORM" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DVVM_BUILD_TESTS=OFF \
    -DVVM_BUILD_EXAMPLES=OFF \
    -DVVM_BUILD_PYTORCH=OFF \
    -DVVM_BUILD_ONNX=OFF \
    -DVVM_BUILD_NETWORK=ON \
    -DVVM_BUILD_TENSOR_TRANSPORT=ON \
    -DVVM_BUILD_SHARED=ON \
    -DVVM_ANDROID_HARDWARE_BUFFER=ON \
    -DVVM_ANDROID_EXTERNAL_MEMORY=ON \
    -DCMAKE_TOOLCHAIN_FILE=cmake/android.toolchain.cmake \
    .

# Build
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j$(nproc)

echo "Build complete! Output in $BUILD_DIR"
echo ""
echo "To install on device:"
echo "  adb push $BUILD_DIR/libvulkan_vm.so /data/local/tmp/"
echo ""
echo "Or create AAR for Android Studio integration"