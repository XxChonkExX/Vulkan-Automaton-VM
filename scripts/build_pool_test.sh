#!/bin/bash
# build_pool_test.sh - Build vulkanvm_pool_test.so (the Chonk allocator binding)
#
# The pool test module is a pure pybind11 extension: NO torch dependency.
# The CMake find_package(Torch) path is unusable on this machine (the ROCm
# wheel's LoadHIP.cmake requires miopen/hipfft/hipsparse configs which are not
# installed), so we build directly:
#   1. fresh PIC static build of the core VulkanVM library
#      (ABI must match current headers - the checked-in build/ tree is stale)
#   2. g++ compile of _pool_test_module.cpp against it
#   3. deploy to _build/vulkanvm_pool_test.so (what train_qwen_chonk.py loads)
#
# Usage: ./scripts/build_pool_test.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LIB_BUILD_DIR="${LIB_BUILD_DIR:-$REPO/build_pool_lib}"
OUT_DIR="$REPO/_build"
PYTHON="${PYTHON:-$REPO/venv-ds4/bin/python3}"

echo "[build_pool_test] step 1/3: building core VulkanVM static lib (PIC)..."
cmake -S "$REPO" -B "$LIB_BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-fPIC" \
    -DVVM_BUILD_PYTORCH=OFF \
    -DVVM_BUILD_ONNX=OFF \
    -DVVM_BUILD_TESTS=OFF \
    -DVVM_BUILD_EXAMPLES=OFF \
    -DVVM_BUILD_NETWORK=ON \
    -DVVM_BUILD_SHARED=OFF
cmake --build "$LIB_BUILD_DIR" -j"$(nproc)"

echo "[build_pool_test] step 2/3: compiling the split integration modules..."
PYBIND11_INC="$("$PYTHON" -c 'import pybind11; print(pybind11.get_include())')"
PYTHON_INC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"

g++ -O3 -std=c++20 -fPIC -shared \
    -DVVM_PLATFORM_LINUX=1 -D__HIP_PLATFORM_AMD__ \
    -I"$REPO/include" \
    -I"$REPO/python/vulkanvm_torch" \
    -I"$PYBIND11_INC" \
    -I"$PYTHON_INC" \
    "$REPO/python/vulkanvm_torch/bindings/pool_bindings.cpp" \
    "$REPO/python/vulkanvm_torch/device/pool_device.cpp" \
    "$REPO/python/vulkanvm_torch/allocator/chonk_allocator.cpp" \
    "$REPO/python/vulkanvm_torch/interop/hip_external_memory.cpp" \
    "$LIB_BUILD_DIR/libvulkan_vm.a" \
    -lhiprtc -lamdhip64 -lvulkan -lpthread -ldl \
    -o "$OUT_DIR/vulkanvm_pool_test.so"

echo "[build_pool_test] step 3/3: deployed to $OUT_DIR/vulkanvm_pool_test.so"
"$PYTHON" -c "
import sys
sys.path.insert(0, '$OUT_DIR')
import vulkanvm_pool_test
print('[build_pool_test] module loads OK')
"
