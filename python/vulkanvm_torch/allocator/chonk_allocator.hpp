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

// Release fully-free slab blocks down to `keepFloor` blocks. Blocks are the
// 2-GiB-class committed pool chunks; freeing them returns their capacity to
// the pool. Used periodically (not on the hot path) to stop kc-proportional
// transient blocks from ratcheting `totalUsed` upward every chunk.
// Returns the number of blocks released.
size_t vvm_torch_chonk_allocator_release_empty(size_t keepFloor);

// Return the slab's live block/liveBytes/freeBytes accounting as a compact
// text string (no allocation). For diagnosing why releaseEmptyBlocks isn't
// reclaiming: how many blocks, how many are fully-free, liveBytes each.
const char* vvm_torch_chonk_allocator_slab_stats();


}  // namespace vvm_torch


