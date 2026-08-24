// hip_external_memory.hpp - HIP <-> Vulkan external-memory bridge.
// Imports a Vulkan-exported DMA-BUF fd into HIP address space so PyTorch
// tensors can alias Chonk Buffer memory directly.

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

namespace vvm_torch {

// Import `fd` (opaque fd exported from a Vulkan VkDeviceMemory) as HIP
// external memory of `size` bytes. Returns the mapped base pointer, or
// nullptr on failure (outExt is then untouched).
void* hipImportFromFd(int fd, size_t size, hipExternalMemory_t* outExt);

}  // namespace vvm_torch
