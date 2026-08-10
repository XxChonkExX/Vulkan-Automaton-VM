#define NOMINMAX
#include "vulkan_vm/buddy_allocator.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <utility>
#include <set>

namespace vvm {

BuddyAllocator::BuddyAllocator(VkDeviceSize blockSize, VkDeviceSize minSize, bool threadSafe)
    : blockSize_(blockSize), minSize_(minSize), threadSafe_(threadSafe)
{
    if (!isPowerOfTwo(blockSize_) || !isPowerOfTwo(minSize_) || blockSize_ < minSize_) {
        VVM_LOG_ERROR("BuddyAllocator: blockSize and minSize must be powers of two "
                      "and blockSize >= minSize (got block={}, min={})",
                      blockSize_, minSize_);
        maxOrder_ = -1;
        return;
    }

    // Compute number of orders: minSize << maxOrder == blockSize
    maxOrder_ = 0;
    VkDeviceSize s = minSize_;
    while (s < blockSize_) {
        s <<= 1;
        ++maxOrder_;
        if (maxOrder_ > 63) {           // defensive
            maxOrder_ = -1;
            return;
        }
    }

    freeLists_.assign(static_cast<size_t>(maxOrder_) + 1, {});
    // Initially the whole block is free at the highest order.
    freeLists_[maxOrder_].insert(0);
}

int BuddyAllocator::sizeToOrder(VkDeviceSize size) const {
    size = nextPowerOfTwo(size);
    if (size < minSize_) size = minSize_;
    if (size > blockSize_) return -1;

    int order = 0;
    VkDeviceSize s = minSize_;
    while (s < size) {
        s <<= 1;
        ++order;
    }
    return order;
}

VkDeviceSize BuddyAllocator::orderToSize(int order) const {
    assert(order >= 0 && order <= maxOrder_);
    return minSize_ << order;
}

void BuddyAllocator::pushFree(int order, VkDeviceSize offset) {
    assert(order >= 0 && order <= maxOrder_);
    freeLists_[order].insert(offset);
}

std::optional<VkDeviceSize> BuddyAllocator::popFree(int order) {
    if (order < 0 || order > maxOrder_) return std::nullopt;
    auto& set = freeLists_[order];
    if (set.empty()) return std::nullopt;
    // Deterministic placement: hand out the lowest free offset so that buddy
    // pairs stay adjacent and coalescing works as expected.
    // O(1) with std::set since begin() is the minimum.
    auto it = set.begin();
    VkDeviceSize off = *it;
    set.erase(it);
    return off;
}

std::optional<VkDeviceSize> BuddyAllocator::splitTo(int order, int targetOrder) {
    // We have a free block of `order`; we need one of `targetOrder` (<= order).
    while (order > targetOrder) {
        auto opt = popFree(order);
        if (!opt) return std::nullopt;  // should not happen if called correctly
        VkDeviceSize off = *opt;
        --order;
        VkDeviceSize half = orderToSize(order);
        // Two buddies: [off, off+half)
        pushFree(order, off + half);    // right buddy becomes free
        pushFree(order, off);           // left will be taken / further split
    }
    return popFree(targetOrder);
}

std::optional<VkDeviceSize> BuddyAllocator::allocate(VkDeviceSize size) {
    return withLock([this, size]() -> std::optional<VkDeviceSize> {
        if (maxOrder_ < 0) {
            VVM_LOG_ERROR("BuddyAllocator: not properly initialized");
            return std::nullopt;
        }
        if (size == 0) return std::nullopt;

        int target = sizeToOrder(size);
        if (target < 0) return std::nullopt;

        // Find the smallest order >= target that has a free block.
        int order = target;
        while (order <= maxOrder_ && freeLists_[order].empty()) {
            ++order;
        }
        if (order > maxOrder_) return std::nullopt;   // completely full

        std::optional<VkDeviceSize> result;
        if (order == target) {
            result = popFree(order);
        } else {
            result = splitTo(order, target);
        }

        if (!result) return std::nullopt;

        const VkDeviceSize granted = orderToSize(target);
        allocated_[*result] = {target, granted};
        return result;
    });
}

void BuddyAllocator::deallocate(VkDeviceSize offset, VkDeviceSize size) {
    withLock([this, offset, size]() {
        if (maxOrder_ < 0) {
            VVM_LOG_ERROR("BuddyAllocator: deallocate on uninitialized allocator");
            return;
        }

        auto it = allocated_.find(offset);
        if (it == allocated_.end()) {
            VVM_LOG_WARN("BuddyAllocator: deallocate unknown offset {} (double-free or bad offset)",
                         offset);
            return;
        }

        const int order = it->second.order;
        const VkDeviceSize recorded = it->second.size;

        if (size != 0 && size != recorded) {
            VVM_LOG_WARN("BuddyAllocator: size mismatch at offset {} (recorded {}, passed {})",
                         offset, recorded, size);
            // Still free using the recorded size – safer than trusting the caller.
        }

        allocated_.erase(it);
        coalesce(order, offset);
    });
}

void BuddyAllocator::coalesce(int order, VkDeviceSize offset) {
    while (order < maxOrder_) {
        const VkDeviceSize buddySize = orderToSize(order);
        const VkDeviceSize buddy = offset ^ buddySize;   // classic buddy address

        auto& set = freeLists_[order];
        auto pos = set.find(buddy);
        if (pos == set.end()) {
            // Buddy is still allocated → just free this block.
            pushFree(order, offset);
            return;
        }

        // Buddy is free → remove it and merge upward.
        set.erase(pos);
        offset = std::min(offset, buddy);   // parent starts at the lower address
        ++order;
    }
    // Fully coalesced to the root.
    pushFree(maxOrder_, 0);
}

VkDeviceSize BuddyAllocator::getLargestFree() const {
    return withLock([this]() -> VkDeviceSize {
        if (maxOrder_ < 0) return 0;
        for (int o = maxOrder_; o >= 0; --o) {
            if (!freeLists_[o].empty()) {
                return orderToSize(o);
            }
        }
        return 0;
    });
}

float BuddyAllocator::getFragmentation() const {
    return withLock([this]() -> float {
        if (maxOrder_ < 0) return 0.f;

        VkDeviceSize totalFree = 0;
        VkDeviceSize largest   = 0;
        for (int o = 0; o <= maxOrder_; ++o) {
            const VkDeviceSize sz = orderToSize(o);
            const size_t count = freeLists_[o].size();
            totalFree += sz * count;
            if (count > 0) largest = std::max(largest, sz);
        }
        if (totalFree == 0) return 0.f;
        return 1.f - static_cast<float>(largest) / static_cast<float>(totalFree);
    });
}

size_t BuddyAllocator::getAllocationCount() const {
    return withLock([this]() -> size_t {
        return allocated_.size();
    });
}

bool BuddyAllocator::isValid() const {
    return withLock([this]() -> bool {
        return maxOrder_ >= 0;
    });
}

} // namespace vvm