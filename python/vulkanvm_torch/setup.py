#!/usr/bin/env python3
# setup.py for vulkanvm_torch PyTorch C++ extension

import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Get PyTorch paths
import torch

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_BUILD_TYPE=Release',
        ]

        # PyTorch paths
        cmake_args += [
            '-DTORCH_INCLUDE_DIRS=' + ';'.join(torch.utils.cpp_extension.include_paths()),
            '-DTORCH_LIBRARY_DIRS=' + ';'.join(torch.utils.cpp_extension.library_paths()),
        ]

        # VulkanVM install path
        vulkanvm_root = os.environ.get('VULKANVM_ROOT', os.path.abspath('../../..'))
        cmake_args += ['-DVULKANVM_ROOT=' + vulkanvm_root]

        # Vulkan SDK
        vulkan_sdk = os.environ.get('VULKAN_SDK')
        if vulkan_sdk:
            cmake_args += ['-DVULKAN_SDK=' + vulkan_sdk]

        cfg = 'Release'
        build_args = ['--config', cfg, '--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# Read version
with open('VERSION', 'r') as f:
    version = f.read().strip()

setup(
    name='vulkanvm_torch',
    version=version,
    author='VulkanVM Team',
    description='PyTorch integration for VulkanVM',
    long_description='PyTorch C++ extension for VulkanVM unified memory pool',
    ext_modules=[CMakeExtension('vulkanvm_torch')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=['torch'],
    python_requires='>=3.8',
)