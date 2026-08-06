#pragma once

#include <vulkan/vulkan.h>
#include <unordered_map>
#include <memory>
#include <optional>

namespace vvm {

// ============================================================================
// Buddy Allocator (power-of-2, for large tensor allocations)
// ============================================================================

struct BuddyNode {
    VkDeviceSize offset;
    VkDeviceSize size;
    bool free;
    BuddyNode* left = nullptr;
    BuddyNode* right = nullptr;
    BuddyNode* parent = nullptr;
    int level;  // 0 = root (block size), maxLevel = min allocation
};

class BuddyAllocator {
public:
    BuddyAllocator(VkDeviceSize blockSize, VkDeviceSize minSize);
    ~BuddyAllocator();
    
    BuddyAllocator(const BuddyAllocator&) = delete;
    BuddyAllocator& operator=(const BuddyAllocator&) = delete;
    
    std::optional<VkDeviceSize> allocate(VkDeviceSize size);
    void deallocate(VkDeviceSize offset, VkDeviceSize size);
    VkDeviceSize getLargestFree() const;
    float getFragmentation() const;
    size_t getAllocationCount() const { return allocatedNodes_.size(); }
    bool isValid() const { return maxLevel_ >= 0 && root_ != nullptr; }
    
private:
    struct AllocatedNode {
        BuddyNode* node;
        VkDeviceSize size;
    };
    
    BuddyNode* root_ = nullptr;
    VkDeviceSize blockSize_;
    VkDeviceSize minSize_;
    int maxLevel_;
    std::unordered_map<VkDeviceSize, AllocatedNode> allocatedNodes_;
    
    BuddyNode* createNode(VkDeviceSize offset, VkDeviceSize size, int level);
    void destroyNode(BuddyNode* node);
    BuddyNode* findFree(BuddyNode* node, VkDeviceSize size);
    void split(BuddyNode* node);
    void merge(BuddyNode* node);
    BuddyNode* getBuddy(BuddyNode* node);
    int getLevelForSize(VkDeviceSize size) const;
};

} // namespace vvm