#!/bin/bash
# Build script with PyTorch and ONNX support
# Usage: ./build_with_python.sh [--pytorch] [--onnx] [--tests]

set -euo pipefail

BUILD_PYTORCH=OFF
BUILD_ONNX=OFF
BUILD_TESTS=ON

for arg in "$@"; do
    case $arg in
        --pytorch) BUILD_PYTORCH=ON ;;
        --onnx) BUILD_ONNX=ON ;;
        --tests) BUILD_TESTS=ON ;;
        *) echo "Unknown arg: $arg" ;;
    esac
done

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DVVM_BUILD_TESTS=$BUILD_TESTS \
    -DVVM_BUILD_PYTORCH=$BUILD_PYTORCH \
    -DVVM_BUILD_ONNX=$BUILD_ONNX \
    -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"

cmake --build build --config Release -j$(nproc)

if [ "$BUILD_TESTS" = "ON" ]; then
    ctest --test-dir build --output-on-failure
fi