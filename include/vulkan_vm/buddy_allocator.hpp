#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <unordered_map>
#include <set>
#include <cstdint>
#include <cassert>
#include <mutex>

namespace vvm {

// Export marker for shared-library builds. Canonical definition lives in
// core.hpp; this fallback keeps the header self-contained.
#ifndef VVM_API
#ifdef VVM_BUILD_SHARED
#if defined(_MSC_VER)
#ifdef VVM_EXPORT
#define VVM_API __declspec(dllexport)
#else
#define VVM_API __declspec(dllimport)
#endif
#else
#define VVM_API __attribute__((visibility("default")))
#endif
#else
#define VVM_API
#endif
#endif

class VVM_API BuddyAllocator {
public:
    // blockSize and minSize MUST be powers of two, blockSize >= minSize.
    // Recommended defaults for general use: minSize = 4 * 1024 (or 64 * 1024 for tensors).
    BuddyAllocator(VkDeviceSize blockSize, VkDeviceSize minSize, bool threadSafe = false);
    ~BuddyAllocator() = default;

    BuddyAllocator(const BuddyAllocator&) = delete;
    BuddyAllocator& operator=(const BuddyAllocator&) = delete;

    // Returns offset into the block, or nullopt on failure / OOM of metadata.
    std::optional<VkDeviceSize> allocate(VkDeviceSize size);

    // size may be 0 ("unknown"); if non-zero it is validated against the recorded size.
    void deallocate(VkDeviceSize offset, VkDeviceSize size = 0);

    VkDeviceSize getLargestFree() const;
    float getFragmentation() const;
    size_t getAllocationCount() const;
    bool isValid() const;

    // Debug-only invariant checker. Returns true if all internal state is consistent.
    // Checks: free blocks don't overlap, free + allocated = blockSize, no duplicate
    // entries in free lists, all free blocks are properly aligned to their order.
    bool checkInvariants() const;

    VkDeviceSize blockSize() const { return blockSize_; }
    VkDeviceSize minSize()  const { return minSize_; }
    int maxOrder()          const { return maxOrder_; }

private:
    static bool isPowerOfTwo(VkDeviceSize v) {
        return v != 0 && (v & (v - 1)) == 0;
    }
    static std::optional<VkDeviceSize> ceilPowerOfTwo(VkDeviceSize v) {
        if (v == 0) return std::nullopt;
        if (v > (VkDeviceSize{1} << 63)) return std::nullopt;
        --v;
        v |= v >> 1;  v |= v >> 2;  v |= v >> 4;
        v |= v >> 8;  v |= v >> 16; v |= v >> 32;
        ++v;
        return v;
    }
    static VkDeviceSize floorPowerOfTwo(VkDeviceSize v) {
        if (v == 0) return 0;
        VkDeviceSize r = 1;
        while ((r << 1) != 0 && (r << 1) <= v) r <<= 1;
        return r;
    }

    // Order 0 = minSize, order maxOrder_ = blockSize.
    int sizeToOrder(VkDeviceSize size) const;
    VkDeviceSize orderToSize(int order) const;

    // Push a free block of the given order onto its free list.
    void pushFree(int order, VkDeviceSize offset);
    // Pop the LOWEST free block of exactly this order (or nullopt).
    // O(log n) using std::set ordered by offset.
    std::optional<VkDeviceSize> popFree(int order);

    // Decompose the region [offset, offset+len) into buddy-aligned power-of-two
    // chunks and push each onto its free list, coalescing as we go. Both offset
    // and len must be multiples of minSize_. This is the unified free path: a
    // full power-of-two grant decomposes to a single chunk (identical to the
    // classic buddy free), while exact-fit grants decompose into O(log) chunks.
    void pushFreeRange(VkDeviceSize offset, VkDeviceSize len);

    // Split a block of `order` down until we obtain a block of `targetOrder`.
    // Returns the offset of the resulting target-sized block, or nullopt.
    std::optional<VkDeviceSize> splitTo(int order, int targetOrder);

    // Try to coalesce starting from a just-freed block.
    void coalesce(int order, VkDeviceSize offset);

    VkDeviceSize blockSize_ = 0;
    VkDeviceSize minSize_   = 0;
    int          maxOrder_  = -1;   // -1 = invalid

    // freeLists_[order] holds offsets of free blocks of size (minSize << order)
    // Using std::set for O(log n) min-offset retrieval + O(log n) buddy lookup during coalesce.
    std::vector<std::set<VkDeviceSize>> freeLists_;

    // Validation / size recovery only. Not on the hot path for performance-critical
    // code that already knows the size. Can be disabled with a compile flag later.
    struct AllocInfo {
        int order;                  // block order the grant was carved from
        VkDeviceSize size;          // granted size: multiple of minSize_, <= orderToSize(order).
                                    // Exact-fit grants may be non-power-of-two; the unused
                                    // tail was returned to the free lists at allocate time.
    };
    std::unordered_map<VkDeviceSize, AllocInfo> allocated_;

    // Optional thread-safety
    mutable std::mutex mutex_;
    bool threadSafe_ = false;

    // Thread-safe wrapper helpers
    template<typename Func>
    auto withLock(Func&& f) const -> decltype(auto) {
        if (threadSafe_) {
            std::lock_guard<std::mutex> lock(mutex_);
            return f();
        }
        return f();
    }
};

} // namespace vvm