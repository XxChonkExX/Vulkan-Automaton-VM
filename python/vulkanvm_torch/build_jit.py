#!/usr/bin/env python3
"""
build_jit.py - JIT-compile the VulkanVM PyTorch C++ extension.

Uses torch.utils.cpp_extension.load() which handles all the platform-specific
compiler setup correctly (MSVC flags, PyTorch headers, linking).

This is PyTorch's officially recommended approach for C++ extensions and
sidesteps the CMake/Visual Studio generator / SDK version mismatches that
plague the standalone CMake build on Windows.
"""

import os
import sys
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
SHADER_DIR = PROJECT_ROOT / "shaders"
INCLUDE_DIR = PROJECT_ROOT / "include"

BUILD_DIR = THIS_DIR / "_jit_build"
BUILD_DIR.mkdir(exist_ok=True)

SPIRV_DIR = BUILD_DIR / "spirv"
SPIRV_DIR.mkdir(exist_ok=True)

GLSLANG = os.environ.get("GLSLANG_VALIDATOR")
if GLSLANG is None:
    candidates = [
        Path(r"C:/VulkanSDK/1.4.304.1/Bin/glslangValidator.exe"),
    ]
    for c in candidates:
        if c.exists():
            GLSLANG = str(c)
            break

if GLSLANG and SHADER_DIR.exists():
    for comp in SHADER_DIR.glob("*.comp"):
        out = SPIRV_DIR / (comp.stem + ".spv")
        try:
            subprocess.run(
                [GLSLANG, "-V", str(comp), "-o", str(out)],
                check=True, capture_output=True,
            )
            print(f"  [shader] {comp.name} -> {out.name}")
        except Exception as e:
            print(f"  [shader] FAIL {comp.name}: {e}")


# ---------------------------------------------------------------------------
# JIT load via torch.utils.cpp_extension.load
# ---------------------------------------------------------------------------

import torch
from torch.utils.cpp_extension import load

sources = [
    str(THIS_DIR / "vulkanvm_torch.cpp"),
    str(THIS_DIR / "vulkanvm_compute.cpp"),
]

print(f"[build_jit] sources: {sources}")
print(f"[build_jit] include: {INCLUDE_DIR}")
print(f"[build_jit] torch:   {torch.__version__} @ {torch.__file__}")

extra_cflags = ["-O3", "-std=c++20", '-DVERSION_INFO=\\"0.2.0\\"']
if sys.platform == "win32":
    extra_cflags = ["/std:c++20", "/EHsc", "/O2", '-DVERSION_INFO=\\"0.2.0\\"']

module = load(
    name="vulkanvm_torch",
    sources=sources,
    extra_include_paths=[str(INCLUDE_DIR), str(THIS_DIR)],
    extra_cflags=extra_cflags,
    extra_cuda_cflags=[],
    extra_ldflags=[],
    build_directory=str(BUILD_DIR),
    verbose=True,
    is_python_module=True,
)

print(f"[build_jit] SUCCESS: {module}")
