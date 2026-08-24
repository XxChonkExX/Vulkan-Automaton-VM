// chonk_allocator.hpp - Slab allocator over Chonk Buffer blocks.
//
// Owns the block lifecycle: Vulkan allocation (via UnifiedMemoryPool) ->
// DMA-BUF export -> HIP external-memory import -> first-fit slab allocation
// with coalescing, double-free guards, and pressure-driven block release.
//
// This module is Python-free: the C ABI below (chonk_allocator_alloc/free)
// is what the PyTorch pluggable allocator installs.

#pragma once

#include <vulkan_vm/vulkan_vm.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vvm_torch {

struct AllocBlock {
    vvm::Allocation alloc;
    int fd = -1;
    hipExternalMemory_t ext = nullptr;
    void* base = nullptr;
    size_t size = 0;
    size_t liveBytes = 0;
    // Free chunks as (offset, size), kept sorted by offset.
    std::vector<std::pair<size_t, size_t>> freeChunks;
};

struct ChonkAllocator {
    std::mutex mtx;
    std::vector<std::unique_ptr<AllocBlock>> blocks;
    std::unordered_map<void*, size_t> liveSizes;
    // Block retention policy:
    //   warmBlocks   - fully-free blocks are kept warm up to this many blocks
    //                  (avoids GTT churn from release/re-create ping-pong)
    //   maxBlocks    - hard cap; when a new block is needed and we are at the
    //                  cap, fully-free blocks are released first to make room
    //                  (bounded GTT commitment - the "better router")
    // Previously blocks were NEVER released (warm=512). That made the
    // monotonic growth of backward-attention temporaries (recompute
    // re-materializes attn weights that grow with kv_len; freed chunks are
    // always smaller than the next request) accumulate exact-fit blocks until
    // the driver's GTT ceiling refused vkAllocateMemory -> zero-storage tensor
    // -> "data is not allocated yet" crash. Bucket-rounded block sizes make a
    // single block absorb the whole growth curve up to its bucket size, and
    // release-on-cap keeps committed GTT bounded.
    size_t warmBlocks = 8;
    size_t maxBlocks = 24;
    size_t minBlocksOnOOM = 4;  // floor kept when releasing under pressure

    static constexpr size_t kAlign = 512;
    static constexpr size_t kMinBlock = 2ull * 1024 * 1024 * 1024;  // 2 GB
    static size_t minBlock();
    static size_t envSize(const char* name, size_t def);
    // Bucket list (GB) parsed once from CHONK_POOL_BLOCK_SIZES_GB ("auto"
    // generates the graduated 1..128 GB ladder).
    static std::vector<size_t> buckets();
    static size_t roundToBucket(size_t need);
};

// C ABI installed into PyTorch's pluggable allocator. `size` < 0 is rejected
// at this boundary (returns nullptr).
extern "C" {
void* chonk_allocator_alloc(ssize_t size, int device, void* stream);
void chonk_allocator_free(void* ptr, size_t size, void* stream);
}

}  // namespace vvm_torch

// Release all blocks and live-size bookkeeping (used by pool shutdown).
void vvm_torch_chonk_allocator_reset();

namespace vvm_torch {
// Internal reset used by the C wrapper above.
void chonk_allocator_reset_impl();
}  // namespace vvm_torch
