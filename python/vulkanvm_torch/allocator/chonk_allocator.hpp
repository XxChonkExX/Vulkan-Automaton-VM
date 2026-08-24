// chonk_allocator.hpp - Production slab allocator over Chonk Buffer blocks.
//
// Wires the pure slab core (chonk_slab.hpp) to the real block provider:
// UnifiedMemoryPool allocation -> DMA-BUF export -> HIP external-memory
// import. Owns the retention policy env knobs and the C ABI that the
// PyTorch pluggable allocator installs. Python-free.

#pragma once

#include <vulkan_vm/vulkan_vm.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vvm_torch {

// C ABI installed into PyTorch's pluggable allocator. `size` < 0 is rejected
// at this boundary (returns nullptr).
extern "C" {
void* chonk_allocator_alloc(ssize_t size, int device, void* stream);
void chonk_allocator_free(void* ptr, size_t size, void* stream);
}

// Release all allocator blocks and live-size bookkeeping (pool shutdown).
void vvm_torch_chonk_allocator_reset();


}  // namespace vvm_torch


