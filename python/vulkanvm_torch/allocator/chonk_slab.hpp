// chonk_slab.hpp - Pure slab allocator core over opaque blocks.
//
// Deliberately FREE of Vulkan/HIP/Python dependencies so the allocator
// invariants can be fuzz-tested on any CPU (audit Priority 3). Block
// resources are type-erased (void* extHandle, int fd); the production
// provider (chonk_allocator.cpp) attaches Vulkan/DMA-BUF/HIP semantics.
//
// Invariants maintained by Core (verifiable via checkInvariants):
//   I1  free chunks sorted strictly by offset, within [0, block size)
//   I2  no two free chunks overlap
//   I3  no two free chunks are adjacent (fully coalesced)
//   I4  sum(free chunks) + liveBytes == block size
//   I5  every live pointer maps into exactly one block, and each block's
//       liveBytes equals the sum of its live allocations

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vvm_torch {
namespace slab {

struct Block {
    void* base = nullptr;        // mapped base address (device or host)
    void* extHandle = nullptr;   // opaque: HIP external memory in production
    int fd = -1;                 // opaque: exported dma-buf fd in production
    size_t size = 0;
    size_t liveBytes = 0;
    // Free chunks as (offset, size), kept sorted by offset.
    std::vector<std::pair<size_t, size_t>> freeChunks;
};

class Core {
public:
    struct IProvider {
        // Try to create a new block with at least `need` usable bytes.
        // Implementations may call core.releaseEmptyBlocks() for pressure
        // relief. Returns nullptr on failure. The returned Block must have
        // base/size set and freeChunks initialized to {{0, size}}.
        virtual Block* createBlock(Core& core, size_t need) = 0;
        // Release all resources and delete the block.
        virtual void destroyBlock(Block* b) = 0;
        virtual ~IProvider() = default;
    };

    static constexpr size_t kAlign = 512;

    Core(IProvider* provider,
         size_t warmBlocks = 8,
         size_t maxBlocks = 24,
         size_t minBlocksOnOOM = 4);

    // Allocate `size` bytes (512-aligned). Returns nullptr when no block can
    // serve the request (provider failed to create one). Optionally reports
    // the granted (aligned) size.
    void* alloc(size_t size, size_t* grantedSize = nullptr);
    // Free a pointer. `sizeHint` is ignored when the live-size map knows the
    // pointer. Unknown pointers are ignored (other allocators may free here).
    void free(void* ptr, size_t sizeHint = 0);

    // Release fully-free blocks while more than `keepFloor` blocks exist.
    // Returns the number of blocks released.
    size_t releaseEmptyBlocks(size_t keepFloor);

    // Verify invariants. I1-I4 (structural) always; I5 (live-map cross-check,
    // O(live x blocks)) only when deep=true. Returns false and fills `err`.
    bool checkInvariants(std::string* err = nullptr, bool deep = true) const;

    struct Stats {
        size_t blocks = 0;
        size_t allocations = 0;   // live allocations
        size_t liveBytes = 0;
        size_t freeBytes = 0;
        size_t capacityBytes = 0;
    };
Stats stats() const;
    size_t blockCount() const { return blocks_.size(); }

    // Collect the counts/sizes of LIVE allocations, bucketed by size class
    // (power-of-two), as comma-separated "size:count" pairs (descending). For
    // diagnosing which transient is growing with sequence position.
    void liveSizeHistogram(std::vector<std::pair<size_t, size_t>>& out) const;

    // Destroy every block via the provider and clear bookkeeping.
    void reset();

private:
    IProvider* provider_;
    std::vector<Block*> blocks_;
    std::unordered_map<void*, size_t> liveSizes_;  // ptr -> aligned size
    size_t warmBlocks_;
    size_t maxBlocks_;
    size_t minBlocksOnOOM_;
};

}  // namespace slab
}  // namespace vvm_torch



