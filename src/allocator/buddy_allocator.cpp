#include "vulkan_vm/allocator.hpp"
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
}

BuddyAllocator::~BuddyAllocator() {
    destroyNode(root_);
}

BuddyNode* BuddyAllocator::createNode(VkDeviceSize offset, VkDeviceSize size, int level) {
    // Use unique_ptr internally but expose raw pointer for tree structure
    // The node is owned by the tree and will be deleted in destroyNode
    BuddyNode* node = new BuddyNode();
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
    auto it = allocatedNodes_.find(offset);
    if (it == allocatedNodes_.end()) {
        VVM_LOG_WARN("deallocate: offset %llu not found (double-free or invalid)", offset);
        return;
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

// ============================================================================
// BlockManager Implementation
// ============================================================================

BlockManager::BlockManager(VkDevice device, const PoolConfig& config)
    : device_(device), config_(config) {
}

BlockManager::~BlockManager() {
    for (auto& block : blocks_) {
        if (block.memory) {
            if (block.hostPtr) {
                vkUnmapMemory(device_, block.memory);
            }
            vkFreeMemory(device_, block.memory, nullptr);
        }
    }
}

std::optional<uint32_t> BlockManager::createBlock(VkDeviceSize size, 
                                                  uint32_t memoryTypeIndex, 
                                                  bool exportable) {
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    
    VkExportMemoryAllocateInfo exportInfo{};
    VkMemoryAllocateFlagsInfo flagsInfo{};
    void* pNext = nullptr;
    
    if (exportable && config_.enableExternal) {
        exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        #ifdef VVM_PLATFORM_LINUX
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        #elif defined(VVM_PLATFORM_WINDOWS)
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        #endif
        exportInfo.pNext = pNext;
        pNext = &exportInfo;
    }
    
    if (config_.enableDeviceAddress) {
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flagsInfo.pNext = pNext;
        pNext = &flagsInfo;
    }
    
    allocInfo.pNext = pNext;
    
    VkDeviceMemory memory;
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        return std::nullopt;
    }
    
    void* hostPtr = nullptr;
    bool hostVisible = false;
    
    int exportFd = -1;
    HANDLE exportHandle = nullptr;
    if (exportable && config_.enableExternal) {
        #ifdef VVM_PLATFORM_LINUX
        VkMemoryGetFdInfoKHR fdInfo{};
        fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fdInfo.memory = memory;
        fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        
        PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = 
            (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
        if (vkGetMemoryFdKHR) {
            vkGetMemoryFdKHR(device_, &fdInfo, &exportFd);
        }
        #elif defined(VVM_PLATFORM_WINDOWS)
        VkMemoryGetWin32HandleInfoKHR handleInfo{};
        handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handleInfo.memory = memory;
        handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        
        PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = 
            (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR");
        if (vkGetMemoryWin32HandleKHR) {
            vkGetMemoryWin32HandleKHR(device_, &handleInfo, &exportHandle);
        }
        #endif
    }
    
    Block block;
    block.memory = memory;
    block.size = size;
    block.used = 0;
    block.hostPtr = hostPtr;
    block.hostVisible = hostVisible;
    block.exportable = exportable;
    block.exportFd = exportFd;
    block.exportHandle = exportHandle;
    block.allocator = std::make_unique<BuddyAllocator>(size, config_.minAlignment);
    
    blocks_.push_back(std::move(block));
    return static_cast<uint32_t>(blocks_.size() - 1);
}

void BlockManager::destroyBlock(uint32_t index) {
    if (index >= blocks_.size()) return;
    
    auto& block = blocks_[index];
    if (block.memory) {
        if (block.hostPtr) {
            vkUnmapMemory(device_, block.memory);
        }
        vkFreeMemory(device_, block.memory, nullptr);
        block.memory = VK_NULL_HANDLE;
    }
}

std::optional<BlockAllocation> BlockManager::allocate(VkDeviceSize size, 
                                                      VkDeviceSize alignment) {
    size = (size + alignment - 1) & ~(alignment - 1);
    
    // Try existing blocks
    for (uint32_t i = 0; i < blocks_.size(); ++i) {
        auto offset = blocks_[i].allocator->allocate(size);
        if (offset.has_value()) {
            blocks_[i].used += size;
            return BlockAllocation{i, *offset, size};
        }
    }
    
    // Need new block (would need memory type index)
    // Simplified - in real impl, pass mem type or use config
    return std::nullopt;
}

void BlockManager::deallocate(const BlockAllocation& alloc) {
    if (alloc.blockIndex >= blocks_.size()) return;
    
    auto& block = blocks_[alloc.blockIndex];
    block.allocator->deallocate(alloc.offset, alloc.size);
    block.used -= alloc.size;
}

} // namespace vvm