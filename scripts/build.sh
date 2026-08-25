#!/bin/bash
# Build script for Linux/macOS

set -e

BUILD_DIR="build"
BUILD_TYPE="Release"
INSTALL_PREFIX="/usr/local"
RUN_TESTS=false
BUILD_SHARED=true
ENABLE_VALIDATION=true

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -b, --build-dir DIR       Build directory (default: build)"
    echo "  -t, --type TYPE           Build type: Debug, Release, RelWithDebInfo (default: Release)"
    echo "  -p, --prefix PREFIX       Install prefix (default: /usr/local)"
    echo "  --tests                   Build and run tests"
    echo "  --static                  Build static library"
    echo "  --no-validation           Disable Vulkan validation layers"
    echo "  -h, --help                Show this help"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--build-dir) BUILD_DIR="$2"; shift 2 ;;
        -t|--type) BUILD_TYPE="$2"; shift 2 ;;
        -p|--prefix) INSTALL_PREFIX="$2"; shift 2 ;;
        --tests) RUN_TESTS=true; shift ;;
        --static) BUILD_SHARED=false; shift ;;
        --no-validation) ENABLE_VALIDATION=false; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

cmake_args=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    -DVVM_BUILD_SHARED="$BUILD_SHARED"
    -DVVM_BUILD_TESTS="$RUN_TESTS"
    -DVVM_ENABLE_VALIDATION="$ENABLE_VALIDATION"
)

echo "Configuring VulkanVM..."
cmake -B "$BUILD_DIR" "${cmake_args[@]}" .

echo "Building..."
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -- -j$(nproc)

if [ "$RUN_TESTS" = true ]; then
    echo "Running tests..."
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

echo "Build complete. Install with: cmake --install $BUILD_DIR"