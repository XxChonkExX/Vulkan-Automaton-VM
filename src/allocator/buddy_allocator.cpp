#include "vulkan_vm/buddy_allocator.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <memory>
#include <cassert>
#include <functional>

namespace vvm {

// ============================================================================
// BuddyAllocator Implementation
// ============================================================================

// Helper: check if value is power of two
static inline bool isPowerOfTwo(VkDeviceSize value) {
    return value != 0 && (value & (value - 1)) == 0;
}

// Helper: round up to next power of two
static inline VkDeviceSize nextPowerOfTwo(VkDeviceSize value) {
    if (value == 0) return 1;
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return value + 1;
}

BuddyAllocator::BuddyAllocator(VkDeviceSize blockSize, VkDeviceSize minSize)
    : blockSize_(blockSize), minSize_(minSize) {
    
    // Enforce power-of-two sizes for buddy allocator correctness
    assert(isPowerOfTwo(blockSize_) && "blockSize must be power of two");
    assert(isPowerOfTwo(minSize_) && "minSize must be power of two");
    assert(blockSize_ >= minSize_ && "blockSize must be >= minSize");
    
    maxLevel_ = 0;
    VkDeviceSize temp = blockSize_;
    while (temp > minSize_) {
        temp >>= 1;
        maxLevel_++;
    }
    
    root_ = createNode(0, blockSize_, 0);
    if (!root_) {
        // Mark as invalid - allocate() will return nullopt
        maxLevel_ = -1;
    }
}

BuddyAllocator::~BuddyAllocator() {
    destroyNode(root_);
}

BuddyNode* BuddyAllocator::createNode(VkDeviceSize offset, VkDeviceSize size, int level) {
    // Use nothrow new to avoid exceptions, return nullptr on OOM
    BuddyNode* node = new (std::nothrow) BuddyNode();
    if (!node) {
        VVM_LOG_ERROR("BuddyAllocator: failed to allocate BuddyNode (out of memory)");
        return nullptr;
    }
    node->offset = offset;
    node->size = size;
    node->free = true;
    node->level = level;
    node->left = nullptr;
    node->right = nullptr;
    node->parent = nullptr;
    return node;
}

void BuddyAllocator::destroyNode(BuddyNode* node) {
    if (!node) return;
    destroyNode(node->left);
    destroyNode(node->right);
    delete node;
}

int BuddyAllocator::getLevelForSize(VkDeviceSize size) const {
    // Round up to power of two for buddy allocator
    size = nextPowerOfTwo(size);
    
    // Clamp to minSize
    if (size < minSize_) size = minSize_;
    
    int level = 0;
    VkDeviceSize temp = blockSize_;
    while (temp > size && level < maxLevel_) {
        temp >>= 1;
        level++;
    }
    return std::min(level, maxLevel_);
}

BuddyNode* BuddyAllocator::findFree(BuddyNode* node, VkDeviceSize size) {
    if (!node || node->size < size) {
        return nullptr;
    }

    // Already split -> search children only
    if (node->left != nullptr) {
        BuddyNode* result = findFree(node->left, size);
        if (result) {
            return result;
        }
        return findFree(node->right, size);
    }

    // Leaf
    if (!node->free) {
        return nullptr;
    }

    // Free leaf large enough. Split only if a half still fits the request
    // and we have not hit maxLevel_.
    if (node->level < maxLevel_ && (node->size / 2) >= size) {
        split(node);
        // If split failed (OOM), this node is still a leaf - use it if it fits
        if (node->left == nullptr) {
            return node;
        }
        // Prefer left child (deterministic, good locality)
        BuddyNode* result = findFree(node->left, size);
        if (result) {
            return result;
        }
        return findFree(node->right, size);
    }

    // This leaf is the right size (or the smallest we can give)
    return node;
}

void BuddyAllocator::split(BuddyNode* node) {
    if (!node || node->level >= maxLevel_ || node->left != nullptr) {
        return;
    }

    const VkDeviceSize halfSize = node->size / 2;
    node->left  = createNode(node->offset,            halfSize, node->level + 1);
    node->right = createNode(node->offset + halfSize, halfSize, node->level + 1);
    
    // Check if node creation failed
    if (!node->left || !node->right) {
        // Clean up any partial allocation
        if (node->left) {
            delete node->left;
            node->left = nullptr;
        }
        if (node->right) {
            delete node->right;
            node->right = nullptr;
        }
        VVM_LOG_ERROR("BuddyAllocator: failed to split node (out of memory)");
        return;
    }
    
    node->left->parent  = node;
    node->right->parent = node;

    // Parent is no longer a free leaf
    node->free = false;
}

BuddyNode* BuddyAllocator::getBuddy(BuddyNode* node) {
    if (!node->parent) return nullptr;
    if (node->parent->left == node) return node->parent->right;
    return node->parent->left;
}

void BuddyAllocator::merge(BuddyNode* node) {
    while (node && node->parent) {
        BuddyNode* buddy = getBuddy(node);
        // Buddy must exist, be free, and still be a leaf
        if (!buddy || !buddy->free || buddy->left != nullptr) {
            break;
        }

        BuddyNode* parent = node->parent;
        destroyNode(parent->left);
        destroyNode(parent->right);
        parent->left  = nullptr;
        parent->right = nullptr;
        parent->free  = true;
        node = parent;
    }
}

std::optional<VkDeviceSize> BuddyAllocator::allocate(VkDeviceSize size) {
    // Check if allocator was initialized properly
    if (maxLevel_ < 0 || !root_) {
        VVM_LOG_ERROR("BuddyAllocator: allocator not properly initialized (OOM during construction)");
        return std::nullopt;
    }
    
    if (size == 0) {
        return std::nullopt;
    }
    if (size < minSize_) {
        size = minSize_;
    }

    size = nextPowerOfTwo(size);
    if (size > blockSize_) {
        return std::nullopt;
    }

    BuddyNode* node = findFree(root_, size);
    if (!node) {
        return std::nullopt;
    }

    // findFree already produced a leaf of appropriate size
    node->free = false;
    allocatedNodes_[node->offset] = {node, size};
    return node->offset;
}

void BuddyAllocator::deallocate(VkDeviceSize offset, VkDeviceSize size) {
    if (maxLevel_ < 0 || !root_) {
        VVM_LOG_ERROR("BuddyAllocator: cannot deallocate from uninitialized allocator");
        return;
    }
    
    auto it = allocatedNodes_.find(offset);
    if (it == allocatedNodes_.end()) {
        VVM_LOG_WARN("deallocate: offset %llu not found (double-free or invalid)", offset);
        return;
    }
    
    // Validate size matches (allow size==0 as "unknown" from caller)
    if (size != 0 && it->second.size != size) {
        VVM_LOG_WARN("deallocate: size mismatch for offset %llu - stored %llu vs passed %llu",
                     offset, it->second.size, size);
    }
    
    BuddyNode* node = it->second.node;
    
    // Double-free validation: node must not be free already
    if (!node->free) {
        node->free = true;
        merge(node);
    } else {
        VVM_LOG_WARN("deallocate: offset %llu already free (double-free)", offset);
    }
    
    allocatedNodes_.erase(it);
}

VkDeviceSize BuddyAllocator::getLargestFree() const {
    std::function<VkDeviceSize(BuddyNode*)> findLargest = [&](BuddyNode* node) -> VkDeviceSize {
        if (!node) return 0;
        if (node->free) return node->size;
        return std::max(findLargest(node->left), findLargest(node->right));
    };
    return findLargest(root_);
}

float BuddyAllocator::getFragmentation() const {
    VkDeviceSize totalFree = 0;
    VkDeviceSize largestFree = 0;
    
    std::function<void(BuddyNode*)> traverse = [&](BuddyNode* node) {
        if (!node) return;
        if (node->free) {
            totalFree += node->size;
            largestFree = std::max(largestFree, node->size);
        }
        traverse(node->left);
        traverse(node->right);
    };
traverse(root_);
    
    if (totalFree == 0) return 0.0f;
    return 1.0f - static_cast<float>(largestFree) / static_cast<float>(totalFree);
}

} // namespace vvm