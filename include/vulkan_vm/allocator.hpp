#pragma once

#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/buddy_allocator.hpp"

namespace vvm {

// ============================================================================
// Block Manager (manages multiple VkDeviceMemory blocks)
// ============================================================================

struct BlockAllocation {
    uint32_t blockIndex;
    VkDeviceSize offset;
    VkDeviceSize size;
};

class BlockManager {
public:
    struct Block {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        VkDeviceSize used = 0;
        void* hostPtr = nullptr;
        bool hostVisible = false;
        bool exportable = false;
        int exportFd = -1;
#ifdef VVM_PLATFORM_WINDOWS
        HANDLE exportHandle = nullptr;
#endif
        std::unique_ptr<BuddyAllocator> allocator;
    };
    
    BlockManager(VkDevice device, const PoolConfig& config);
    ~BlockManager();
    
    BlockManager(const BlockManager&) = delete;
    BlockManager& operator=(const BlockManager&) = delete;
    
    std::optional<BlockAllocation> allocate(VkDeviceSize size, VkDeviceSize alignment);
    void deallocate(const BlockAllocation& alloc);
    
    std::optional<uint32_t> createBlock(VkDeviceSize size, uint32_t memoryTypeIndex, bool exportable);
    void destroyBlock(uint32_t index);
    
    Block& getBlock(uint32_t index) { return blocks_[index]; }
    const Block& getBlock(uint32_t index) const { return blocks_[index]; }
    size_t getBlockCount() const { return blocks_.size(); }
    
    VkDevice getDevice() const { return device_; }
    
private:
    VkDevice device_;
    PoolConfig config_;
    std::vector<Block> blocks_;
    uint32_t deviceLocalMemType_ = UINT32_MAX;
    uint32_t hostVisibleMemType_ = UINT32_MAX;
};

} // namespace vvm