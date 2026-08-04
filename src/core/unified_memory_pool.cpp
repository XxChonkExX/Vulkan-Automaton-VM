#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/allocator.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <cstring>

namespace vvm {

// ============================================================================
// UnifiedMemoryPool Implementation
// ============================================================================

std::optional<UnifiedMemoryPool> UnifiedMemoryPool::create(
    const DeviceConfig& device, const PoolConfig& config) {
    
    UnifiedMemoryPool pool;
    if (pool.initialize(device, config)) {
        return pool;
    }
    return std::nullopt;
}

UnifiedMemoryPool::UnifiedMemoryPool(UnifiedMemoryPool&& other) noexcept
    : deviceConfig_(std::move(other.deviceConfig_))
    , config_(std::move(other.config_))
    , device_(other.device_)
    , blocks_(std::move(other.blocks_))
    , deviceLocalMemoryType_(other.deviceLocalMemoryType_)
    , hostVisibleMemoryType_(other.hostVisibleMemoryType_)
    , transferCmdPool_(other.transferCmdPool_) {
    
    other.device_ = VK_NULL_HANDLE;
    other.transferCmdPool_ = VK_NULL_HANDLE;
}

UnifiedMemoryPool& UnifiedMemoryPool::operator=(UnifiedMemoryPool&& other) noexcept {
    if (this != &other) {
        // Cleanup current
        if (device_) {
            for (auto& block : blocks_) {
                if (block.memory) {
                    vkFreeMemory(device_, block.memory, nullptr);
                }
            }
            if (transferCmdPool_) {
                vkDestroyCommandPool(device_, transferCmdPool_, nullptr);
            }
        }
        
        deviceConfig_ = std::move(other.deviceConfig_);
        config_ = std::move(other.config_);
        device_ = other.device_;
        blocks_ = std::move(other.blocks_);
        deviceLocalMemoryType_ = other.deviceLocalMemoryType_;
        hostVisibleMemoryType_ = other.hostVisibleMemoryType_;
        transferCmdPool_ = other.transferCmdPool_;
        
        other.device_ = VK_NULL_HANDLE;
        other.transferCmdPool_ = VK_NULL_HANDLE;
    }
    return *this;
}

UnifiedMemoryPool::~UnifiedMemoryPool() {
    if (device_) {
        for (auto& block : blocks_) {
            if (block.memory) {
                if (block.hostPtr) {
                    vkUnmapMemory(device_, block.memory);
                }
                vkFreeMemory(device_, block.memory, nullptr);
            }
        }
        if (transferCmdPool_) {
            vkDestroyCommandPool(device_, transferCmdPool_, nullptr);
        }
    }
}

bool UnifiedMemoryPool::initialize(const DeviceConfig& device, const PoolConfig& config) {
    deviceConfig_ = device;
    config_ = config;
    device_ = device.device;
    
    // Find memory types
    auto memProps = getDeviceMemoryInfo().memProps;
    
    // Prefer DEVICE_LOCAL for primary allocations
    auto devLocalType = findMemoryTypeIndex(
        memProps,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        config_.preferredFlags
    );
    
    if (!devLocalType.has_value()) {
        return false;
    }
    deviceLocalMemoryType_ = *devLocalType;
    
    // Host-visible for shadow/offload
    if (config_.enableHostVisible) {
        auto hostVisibleType = findMemoryTypeIndex(
            memProps,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            0
        );
        if (hostVisibleType.has_value()) {
            hostVisibleMemoryType_ = *hostVisibleType;
        }
    }
    
    // Create transfer command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = deviceConfig_.transferQueueFamily != UINT32_MAX 
        ? deviceConfig_.transferQueueFamily 
        : deviceConfig_.graphicsQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    
    if (vkCreateCommandPool(device_, &poolInfo, nullptr, &transferCmdPool_) != VK_SUCCESS) {
        return false;
    }
    
    // Pre-allocate first block
    return allocateBlock(config_.blockSize, deviceLocalMemoryType_, config_.enableExternal).has_value();
}

std::optional<uint32_t> UnifiedMemoryPool::findMemoryType(
    VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) {
    
    return findMemoryTypeIndex(getDeviceMemoryInfo().memProps, required, preferred);
}

std::optional<VkDeviceMemory> UnifiedMemoryPool::allocateBlock(
    VkDeviceSize size, uint32_t memoryTypeIndex, bool exportable) {
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    
    // Export support for cross-GPU
    VkExportMemoryAllocateInfo exportInfo{};
    if (exportable && config_.enableExternal) {
        exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        
        #ifdef VVM_PLATFORM_LINUX
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        #elif defined(VVM_PLATFORM_WINDOWS)
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        #endif
        
        allocInfo.pNext = &exportInfo;
    }
    
    // Device address support for bindless
    VkMemoryAllocateFlagsInfo flagsInfo{};
    if (config_.enableDeviceAddress) {
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flagsInfo.pNext = allocInfo.pNext;
        allocInfo.pNext = &flagsInfo;
    }
    
    VkDeviceMemory memory;
    VkResult result = vkAllocateMemory(device_, &allocInfo, nullptr, &memory);
    if (result != VK_SUCCESS) {
        return std::nullopt;
    }
    
    // Map if host-visible
    void* hostPtr = nullptr;
    VkMemoryPropertyFlags memFlags;
    getMemoryTypeProperties(memoryTypeIndex, memFlags, getDeviceMemoryInfo().memProps);
    
    if (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        vkMapMemory(device_, memory, 0, VK_WHOLE_SIZE, 0, &hostPtr);
    }
    
    BlockInfo block;
    block.memory = memory;
    block.size = size;
    block.used = 0;
    block.hostPtr = hostPtr;
    block.freeRanges.emplace_back(0, size);
    
    // Export handle if requested
    if (exportable && config_.enableExternal) {
        #ifdef VVM_PLATFORM_LINUX
        VkMemoryGetFdInfoKHR fdInfo{};
        fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fdInfo.memory = memory;
        fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        
        PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = 
            (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
        if (vkGetMemoryFdKHR) {
            vkGetMemoryFdKHR(device_, &fdInfo, &block.exportFd);
        }
        #elif defined(VVM_PLATFORM_WINDOWS)
        VkMemoryGetWin32HandleInfoKHR handleInfo{};
        handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handleInfo.memory = memory;
        handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        
        PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = 
            (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR");
        if (vkGetMemoryWin32HandleKHR) {
            vkGetMemoryWin32HandleKHR(device_, &handleInfo, &block.exportHandle);
        }
        #endif
    }
    
    blocks_.push_back(std::move(block));
    return memory;
}

std::optional<Allocation> UnifiedMemoryPool::allocate(VkDeviceSize size,
                                                       VkBufferUsageFlags usage,
                                                       VkMemoryPropertyFlags flags) {
    // Align size
    size = alignUp(size, config_.minAlignment);
    
    // Try existing blocks first
    for (uint32_t i = 0; i < blocks_.size(); ++i) {
        auto range = findFreeRange(i, size, config_.minAlignment);
        if (range.first != UINT64_MAX) {
            return subAllocate(size, config_.minAlignment, i);
        }
    }
    
    // Need new block
    if (blocks_.size() >= config_.maxBlocks) {
        return std::nullopt;
    }
    
    uint32_t memType = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) 
        ? (hostVisibleMemoryType_ != UINT32_MAX ? hostVisibleMemoryType_ : deviceLocalMemoryType_)
        : deviceLocalMemoryType_;
    
    if (!allocateBlock(config_.blockSize, memType, config_.enableExternal).has_value()) {
        return std::nullopt;
    }
    
    return subAllocate(size, config_.minAlignment, blocks_.size() - 1);
}

std::optional<Allocation> UnifiedMemoryPool::allocateTensor(VkDeviceSize size,
                                                             VkBufferUsageFlags usage) {
    return allocate(size, usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
}

void UnifiedMemoryPool::deallocate(Allocation&& alloc) {
    if (alloc.blockIndex < blocks_.size()) {
        subDeallocate(std::move(alloc));
    }
}

std::optional<ExternalMemoryInfo> UnifiedMemoryPool::exportMemory(
    const Allocation& alloc, ExternalHandleType type) {
    
    if (alloc.blockIndex >= blocks_.size()) return std::nullopt;
    
    const auto& block = blocks_[alloc.blockIndex];
    ExternalMemoryInfo info;
    info.type = type;
    info.size = alloc.size;
    info.memoryTypeIndex = deviceLocalMemoryType_;  // plain uint32_t now
    
    #ifdef VVM_PLATFORM_LINUX
    if (type == ExternalHandleType::OpaqueFd && block.exportFd >= 0) {
        info.fd = dup(block.exportFd);  // caller owns this fd
        return info;
    }
    #elif defined(VVM_PLATFORM_WINDOWS)
    if (type == ExternalHandleType::OpaqueWin32 && block.exportHandle) {
        HANDLE dupHandle;
        DuplicateHandle(GetCurrentProcess(), block.exportHandle,
                       GetCurrentProcess(), &dupHandle, 0, FALSE, DUPLICATE_SAME_ACCESS);
        info.handle = dupHandle;
        return info;
    }
    #endif
    
    return std::nullopt;
}

std::optional<Allocation> UnifiedMemoryPool::importMemory(
    const ExternalMemoryInfo& info, VkBufferUsageFlags usage) {
    
    VkImportMemoryFdInfoKHR importFdInfo{};
    VkImportMemoryWin32HandleInfoKHR importWin32Info{};
    VkMemoryDedicatedAllocateInfo dedicatedInfo{};
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = info.size;
    allocInfo.memoryTypeIndex = info.memoryTypeIndex;
    
    #ifdef VVM_PLATFORM_LINUX
    if (info.type == ExternalHandleType::OpaqueFd && info.fd >= 0) {
        importFdInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        importFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        importFdInfo.fd = info.fd;
        allocInfo.pNext = &importFdInfo;
    }
    #elif defined(VVM_PLATFORM_WINDOWS)
    if (info.type == ExternalHandleType::OpaqueWin32 && info.handle) {
        importWin32Info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        importWin32Info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        importWin32Info.handle = info.handle;
        allocInfo.pNext = &importWin32Info;
    }
    #endif
    
    VkDeviceMemory memory;
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        return std::nullopt;
    }
    
    // Create buffer bound to imported memory
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = info.size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkBuffer buffer;
    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        vkFreeMemory(device_, memory, nullptr);
        return std::nullopt;
    }
    
    if (vkBindBufferMemory(device_, buffer, memory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer, nullptr);
        vkFreeMemory(device_, memory, nullptr);
        return std::nullopt;
    }
    
    Allocation alloc;
    alloc.buffer = buffer;
    alloc.memory = memory;
    alloc.offset = 0;
    alloc.size = info.size;
    alloc.isExternal = true;
    
    if (config_.enableDeviceAddress) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = buffer;
        alloc.deviceAddress = vkGetBufferDeviceAddress(device_, &addrInfo);
    }
    
    return alloc;
}

std::optional<MigrationOperation> UnifiedMemoryPool::offloadToHost(Allocation& alloc) {
    // TODO: Implement in migration.cpp
    return std::nullopt;
}

std::optional<MigrationOperation> UnifiedMemoryPool::reloadToDevice(Allocation& alloc) {
    // TODO: Implement in migration.cpp
    return std::nullopt;
}

void UnifiedMemoryPool::waitMigration(const MigrationOperation& op) {
    if (op.completionFence) {
        vkWaitForFences(device_, 1, &op.completionFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device_, 1, &op.completionFence);
    }
}

PoolStats UnifiedMemoryPool::getStats() const {
    PoolStats stats;
    for (const auto& block : blocks_) {
        stats.totalAllocated += block.size;
        stats.totalUsed += block.used;
        stats.totalFree += (block.size - block.used);
        stats.blockCount++;
        
        for (const auto& range : block.freeRanges) {
            stats.largestFreeBlock = std::max(stats.largestFreeBlock, range.second);
        }
    }
    
    if (stats.totalAllocated > 0) {
        stats.fragmentationRatio = 1.0f - 
            static_cast<float>(stats.largestFreeBlock) / 
            static_cast<float>(stats.totalFree > 0 ? stats.totalFree : 1);
    }
    
    return stats;
}

DeviceMemoryInfo UnifiedMemoryPool::getDeviceMemoryInfo() const {
    DeviceMemoryInfo info;
    vkGetPhysicalDeviceMemoryProperties(deviceConfig_.physicalDevice, &info.memProps);
    
    // Query budget if extension available
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    
    VkPhysicalDeviceMemoryProperties2 memProps2{};
    memProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memProps2.pNext = &budget;
    
    vkGetPhysicalDeviceMemoryProperties2(deviceConfig_.physicalDevice, &memProps2);
    info.budget = budget;
    
    info.heapSizes.resize(info.memProps.memoryHeapCount);
    info.heapUsed.resize(info.memProps.memoryHeapCount);
    for (uint32_t i = 0; i < info.memProps.memoryHeapCount; ++i) {
        info.heapSizes[i] = info.memProps.memoryHeaps[i].size;
        info.heapUsed[i] = budget.heapUsage[i];
    }
    
    return info;
}

void UnifiedMemoryPool::defragment() {
    // TODO: Implement block compaction
}

void UnifiedMemoryPool::trim() {
    // TODO: Release empty blocks
}

// ============================================================================
// Private Helpers
// ============================================================================

std::optional<Allocation> UnifiedMemoryPool::subAllocate(VkDeviceSize size,
                                                          VkDeviceSize alignment,
                                                          uint32_t blockIndex) {
    auto range = findFreeRange(blockIndex, size, alignment);
    if (range.first == UINT64_MAX) return std::nullopt;
    
    VkDeviceSize offset = range.first;
    
    // Create buffer
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkBuffer buffer;
    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        return std::nullopt;
    }
    
    // Bind to memory at offset
    if (vkBindBufferMemory(device_, buffer, blocks_[blockIndex].memory, offset) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer, nullptr);
        return std::nullopt;
    }
    
    // Update free ranges
    blocks_[blockIndex].freeRanges.erase(
        std::find(blocks_[blockIndex].freeRanges.begin(), 
                  blocks_[blockIndex].freeRanges.end(), range));
    
    if (range.second > size) {
        addFreeRange(blockIndex, offset + size, range.second - size);
    }
    
    blocks_[blockIndex].used += size;
    
    Allocation alloc;
    alloc.buffer = buffer;
    alloc.memory = blocks_[blockIndex].memory;
    alloc.offset = offset;
    alloc.size = size;
    alloc.blockIndex = blockIndex;
    alloc.isHostVisible = (blocks_[blockIndex].hostPtr != nullptr);
    alloc.hostPtr = alloc.isHostVisible 
        ? static_cast<char*>(blocks_[blockIndex].hostPtr) + offset 
        : nullptr;
    
    if (config_.enableDeviceAddress) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = buffer;
        alloc.deviceAddress = vkGetBufferDeviceAddress(device_, &addrInfo);
    }
    
    return alloc;
}

void UnifiedMemoryPool::subDeallocate(Allocation&& alloc) {
    if (alloc.blockIndex >= blocks_.size()) return;
    
    vkDestroyBuffer(device_, alloc.buffer, nullptr);
    
    addFreeRange(alloc.blockIndex, alloc.offset, alloc.size);
    mergeFreeRanges(alloc.blockIndex);
    blocks_[alloc.blockIndex].used -= alloc.size;
}

VkDeviceSize UnifiedMemoryPool::alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

std::pair<VkDeviceSize, VkDeviceSize> UnifiedMemoryPool::findFreeRange(
    uint32_t blockIndex, VkDeviceSize size, VkDeviceSize alignment) {
    
    auto& ranges = blocks_[blockIndex].freeRanges;
    for (auto it = ranges.begin(); it != ranges.end(); ++it) {
        VkDeviceSize alignedOffset = alignUp(it->first, alignment);
        VkDeviceSize available = it->second - (alignedOffset - it->first);
        
        if (available >= size) {
            return {alignedOffset, available};
        }
    }
    return {UINT64_MAX, 0};
}

void UnifiedMemoryPool::addFreeRange(uint32_t blockIndex, VkDeviceSize offset, VkDeviceSize size) {
    blocks_[blockIndex].freeRanges.emplace_back(offset, size);
    // Keep sorted by offset
    std::sort(blocks_[blockIndex].freeRanges.begin(), 
              blocks_[blockIndex].freeRanges.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
}

void UnifiedMemoryPool::mergeFreeRanges(uint32_t blockIndex) {
    auto& ranges = blocks_[blockIndex].freeRanges;
    if (ranges.size() < 2) return;
    
    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> merged;
    merged.push_back(ranges[0]);
    
    for (size_t i = 1; i < ranges.size(); ++i) {
        auto& last = merged.back();
        if (last.first + last.second == ranges[i].first) {
            last.second += ranges[i].second;
        } else {
            merged.push_back(ranges[i]);
        }
    }
    ranges = std::move(merged);
}

} // namespace vvm