#include "vulkan_vm/allocator.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace vvm {

// ============================================================================
// BuddyAllocator Implementation
// ============================================================================

struct AllocatedNode {
    BuddyNode* node;
    VkDeviceSize size;
};

BuddyAllocator::BuddyAllocator(VkDeviceSize blockSize, VkDeviceSize minSize)
    : blockSize_(blockSize), minSize_(minSize) {
    
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
    BuddyNode* node = new BuddyNode();
    node->offset = offset;
    node->size = size;
    node->free = true;
    node->level = level;
    return node;
}

void BuddyAllocator::destroyNode(BuddyNode* node) {
    if (!node) return;
    destroyNode(node->left);
    destroyNode(node->right);
    delete node;
}

int BuddyAllocator::getLevelForSize(VkDeviceSize size) const {
    int level = 0;
    VkDeviceSize temp = blockSize_;
    while (temp > size && level < maxLevel_) {
        temp >>= 1;
        level++;
    }
    return std::min(level, maxLevel_);
}

BuddyNode* BuddyAllocator::findFree(BuddyNode* node, VkDeviceSize size) {
    if (!node || !node->free || node->size < size) return nullptr;
    if (node->level == maxLevel_ || node->size / 2 < size) return node;
    
    if (!node->left) split(node);
    
    BuddyNode* result = findFree(node->left, size);
    if (!result) result = findFree(node->right, size);
    return result;
}

void BuddyAllocator::split(BuddyNode* node) {
    if (node->level >= maxLevel_ || node->left) return;
    
    VkDeviceSize halfSize = node->size / 2;
    node->left = createNode(node->offset, halfSize, node->level + 1);
    node->right = createNode(node->offset + halfSize, halfSize, node->level + 1);
    node->left->parent = node;
    node->right->parent = node;
}

BuddyNode* BuddyAllocator::getBuddy(BuddyNode* node) {
    if (!node->parent) return nullptr;
    if (node->parent->left == node) return node->parent->right;
    return node->parent->left;
}

void BuddyAllocator::merge(BuddyNode* node) {
    while (node->parent) {
        BuddyNode* buddy = getBuddy(node);
        if (!buddy || !buddy->free) break;
        
        // Both free, merge
        node->parent->free = true;
        destroyNode(node->parent->left);
        destroyNode(node->parent->right);
        node->parent->left = nullptr;
        node->parent->right = nullptr;
        node = node->parent;
    }
}

std::optional<VkDeviceSize> BuddyAllocator::allocate(VkDeviceSize size) {
    BuddyNode* node = findFree(root_, size);
    if (!node) return std::nullopt;
    
    node->free = false;
    allocatedNodes_[node->offset] = {node, size};
    
    // Split if larger than needed (but not below minSize)
    while (node->level < maxLevel_ && node->size / 2 >= size) {
        split(node);
        node = node->left;  // Use left child
        node->free = false;
        allocatedNodes_[node->offset] = {node, size};
    }
    
    return node->offset;
}

void BuddyAllocator::deallocate(VkDeviceSize offset, VkDeviceSize size) {
    auto it = allocatedNodes_.find(offset);
    if (it == allocatedNodes_.end()) return;
    
    BuddyNode* node = it->second.node;
    allocatedNodes_.erase(it);
    
    node->free = true;
    merge(node);
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