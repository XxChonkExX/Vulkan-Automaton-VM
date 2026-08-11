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

class BuddyAllocator {
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

    // Order 0 = minSize, order maxOrder_ = blockSize.
    int sizeToOrder(VkDeviceSize size) const;
    VkDeviceSize orderToSize(int order) const;

    // Push a free block of the given order onto its free list.
    void pushFree(int order, VkDeviceSize offset);
    // Pop the LOWEST free block of exactly this order (or nullopt).
    // O(log n) using std::set ordered by offset.
    std::optional<VkDeviceSize> popFree(int order);

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
        int order;
        VkDeviceSize size;          // power-of-two size actually granted
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