#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/buddy_allocator.hpp"
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
    
    // Use MemoryTypeSelector for optimal memory type selection
    MemoryTypeSelector selector(deviceConfig_.physicalDevice);
    
    // Prefer DEVICE_LOCAL for primary allocations
    auto devLocalResult = selector.select(
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        config_.preferredFlags,
        config_.blockSize  // Ensure enough budget for at least one block
    );
    
    if (devLocalResult.memoryTypeIndex == UINT32_MAX) {
        VVM_LOG_ERROR("Failed to find DEVICE_LOCAL memory type with sufficient budget");
        return false;
    }
    deviceLocalMemoryType_ = devLocalResult.memoryTypeIndex;
    VVM_LOG_INFO("Selected DEVICE_LOCAL memory type %u (heap budget: %llu MB, utilization: %.1f%%)",
                 deviceLocalMemoryType_,
                 devLocalResult.heapBudget / (1024*1024),
                 devLocalResult.heapUtilization * 100.0f);
    
    // Host-visible for shadow/offload
    if (config_.enableHostVisible) {
        auto hostVisibleResult = selector.select(
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            0,
            256 * 1024 * 1024  // 256MB minimum
        );
        if (hostVisibleResult.memoryTypeIndex != UINT32_MAX) {
            hostVisibleMemoryType_ = hostVisibleResult.memoryTypeIndex;
            VVM_LOG_INFO("Selected HOST_VISIBLE memory type %u (heap budget: %llu MB)",
                         hostVisibleMemoryType_,
                         hostVisibleResult.heapBudget / (1024*1024));
        } else {
            VVM_LOG_WARN("No suitable HOST_VISIBLE memory type found");
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
        VVM_LOG_ERROR("Failed to create transfer command pool");
        return false;
    }
    
    // Pre-allocate first block
    if (!allocateBlock(config_.blockSize, deviceLocalMemoryType_, config_.enableExternal).has_value()) {
        VVM_LOG_ERROR("Failed to allocate initial memory block");
        return false;
    }
    
    // Create OffloadManager if offload is enabled
    if (config_.enableHostVisible) {
        OffloadConfig offloadConfig;
        offloadConfig.hostShadowSize = config_.blockSize * 4;  // Default to 4x block size
        offloadConfig.useMadvise = true;
        offloadConfig.useMprotect = true;
        offloadConfig.transferQueue = deviceConfig_.transferQueue;
        offloadConfig.transferQueueFamily = deviceConfig_.transferQueueFamily != UINT32_MAX 
            ? deviceConfig_.transferQueueFamily 
            : deviceConfig_.graphicsQueueFamily;
        
        offloadManager_ = std::make_unique<OffloadManager>(this, offloadConfig);
        VVM_LOG_INFO("OffloadManager created with host shadow size: %llu MB", 
                     offloadConfig.hostShadowSize / (1024*1024));
    }
    
    VVM_LOG_INFO("UnifiedMemoryPool initialized successfully (blockSize=%llu MB, alignment=%llu KB)",
                 config_.blockSize / (1024*1024), config_.minAlignment / 1024);
    return true;
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
    void* pNext = nullptr;
    
    if (exportable && config_.enableExternal) {
        exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        
        #ifdef VVM_PLATFORM_LINUX
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT | 
                                 VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        #elif defined(VVM_PLATFORM_WINDOWS)
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT | 
                                 VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
        #endif
        
        exportInfo.pNext = pNext;
        pNext = &exportInfo;
    }
    
    // Device address support for bindless
    VkMemoryAllocateFlagsInfo flagsInfo{};
    if (config_.enableDeviceAddress) {
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flagsInfo.pNext = pNext;
        pNext = &flagsInfo;
    }
    
    // Dedicated allocation for external memory (more robust)
    VkMemoryDedicatedAllocateInfo dedicatedInfo{};
    if (exportable && config_.enableExternal) {
        dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicatedInfo.buffer = VK_NULL_HANDLE;  // Will be set when buffer is created
        dedicatedInfo.image = VK_NULL_HANDLE;
        dedicatedInfo.pNext = pNext;
        pNext = &dedicatedInfo;
    }
    
    allocInfo.pNext = pNext;
    
    VkDeviceMemory memory;
    VkResult result = vkAllocateMemory(device_, &allocInfo, nullptr, &memory);
    if (result != VK_SUCCESS) {
        VVM_LOG_ERROR("vkAllocateMemory failed: %s (size=%llu, type=%u)", 
                      vkResultToString(result).c_str(), size, memoryTypeIndex);
        return std::nullopt;
    }
    
    // Map if host-visible
    void* hostPtr = nullptr;
    VkMemoryPropertyFlags memFlags;
    getMemoryTypeProperties(memoryTypeIndex, memFlags, getDeviceMemoryInfo().memProps);
    
    bool isHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    bool isCoherent = (memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    
    if (isHostVisible) {
        result = vkMapMemory(device_, memory, 0, VK_WHOLE_SIZE, 0, &hostPtr);
        if (result != VK_SUCCESS) {
            VVM_LOG_WARN("vkMapMemory failed: %s", vkResultToString(result).c_str());
            hostPtr = nullptr;
            isHostVisible = false;
        }
    }
    
    BlockInfo block;
    block.memory = memory;
    block.size = size;
    block.used = 0;
    block.hostPtr = hostPtr;
    block.memoryFlags = memFlags;
    block.isHostVisible = isHostVisible;
    block.isCoherent = isCoherent;
    
    // Create buddy allocator for this block
    block.buddy = std::make_unique<BuddyAllocator>(size, config_.minAlignment);
    
    // Export handle if requested
    if (exportable && config_.enableExternal) {
        #ifdef VVM_PLATFORM_LINUX
        VkMemoryGetFdInfoKHR fdInfo{};
        fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fdInfo.memory = memory;
        // Try OPAQUE_FD first, fallback to DMA_BUF
        fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        
        PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = 
            (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
        if (vkGetMemoryFdKHR) {
            VkResult res = vkGetMemoryFdKHR(device_, &fdInfo, &block.exportFd);
            if (res != VK_SUCCESS) {
                // Try DMA_BUF
                fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
                res = vkGetMemoryFdKHR(device_, &fdInfo, &block.exportFd);
            }
        }
        #elif defined(VVM_PLATFORM_WINDOWS)
        VkMemoryGetWin32HandleInfoKHR handleInfo{};
        handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handleInfo.memory = memory;
        handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        
        PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = 
            (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR");
        if (vkGetMemoryWin32HandleKHR) {
            VkResult res = vkGetMemoryWin32HandleKHR(device_, &handleInfo, &block.exportHandle);
            if (res != VK_SUCCESS) {
                // Try D3D12_HEAP
                handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
                res = vkGetMemoryWin32HandleKHR(device_, &handleInfo, &block.exportHandle);
            }
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
    
    // Try existing blocks first - check buddy allocator
    for (uint32_t i = 0; i < blocks_.size(); ++i) {
        if (blocks_[i].buddy && blocks_[i].buddy->getLargestFree() >= size) {
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
    info.memoryTypeIndex = deviceLocalMemoryType_;
    info.dedicatedAllocation = false;
    
    #ifdef VVM_PLATFORM_LINUX
    if (type == ExternalHandleType::OpaqueFd && block.exportFd >= 0) {
        info.fd = dup(block.exportFd);  // caller owns this fd
        return info;
    } else if (type == ExternalHandleType::DmaBuf && block.exportFd >= 0) {
        info.fd = dup(block.exportFd);
        return info;
    }
    #elif defined(VVM_PLATFORM_WINDOWS)
    if (type == ExternalHandleType::OpaqueWin32 && block.exportHandle) {
        HANDLE dupHandle;
        DuplicateHandle(GetCurrentProcess(), block.exportHandle,
                       GetCurrentProcess(), &dupHandle, 0, FALSE, DUPLICATE_SAME_ACCESS);
        info.handle = dupHandle;
        return info;
    } else if (type == ExternalHandleType::D3D12Heap && block.exportHandle) {
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
    
    void* pNext = nullptr;
    
    #ifdef VVM_PLATFORM_LINUX
    if (info.type == ExternalHandleType::OpaqueFd && info.fd >= 0) {
        importFdInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        importFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        importFdInfo.fd = info.fd;
        importFdInfo.pNext = pNext;
        pNext = &importFdInfo;
    } else if (info.type == ExternalHandleType::DmaBuf && info.fd >= 0) {
        importFdInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        importFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        importFdInfo.fd = info.fd;
        importFdInfo.pNext = pNext;
        pNext = &importFdInfo;
    }
    #elif defined(VVM_PLATFORM_WINDOWS)
    if (info.type == ExternalHandleType::OpaqueWin32 && info.handle) {
        importWin32Info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        importWin32Info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        importWin32Info.handle = info.handle;
        importWin32Info.pNext = pNext;
        pNext = &importWin32Info;
    } else if (info.type == ExternalHandleType::D3D12Heap && info.handle) {
        importWin32Info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        importWin32Info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
        importWin32Info.handle = info.handle;
        importWin32Info.pNext = pNext;
        pNext = &importWin32Info;
    }
    #endif
    
    // Dedicated allocation for imported memory
    dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedInfo.buffer = VK_NULL_HANDLE;
    dedicatedInfo.image = VK_NULL_HANDLE;
    dedicatedInfo.pNext = pNext;
    pNext = &dedicatedInfo;
    
    allocInfo.pNext = pNext;
    
    VkDeviceMemory memory;
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to allocate memory for import");
        return std::nullopt;
    }
    
    // Create buffer with external memory support
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = info.size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    // External memory buffer create info
    VkExternalMemoryBufferCreateInfo extBufferInfo{};
    #ifdef VVM_PLATFORM_LINUX
    extBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
                                VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    #elif defined(VVM_PLATFORM_WINDOWS)
    extBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT |
                                VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
    #endif
    extBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extBufferInfo.pNext = bufferInfo.pNext;
    bufferInfo.pNext = &extBufferInfo;
    
    VkBuffer buffer;
    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to create buffer for imported memory");
        vkFreeMemory(device_, memory, nullptr);
        return std::nullopt;
    }
    
    // Bind with bindInfo2 for device group support
    VkBindBufferMemoryInfo bindInfo{};
    bindInfo.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
    bindInfo.buffer = buffer;
    bindInfo.memory = memory;
    bindInfo.memoryOffset = 0;
    
    if (vkBindBufferMemory2(device_, 1, &bindInfo) != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to bind imported memory");
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
    alloc.isHostVisible = false;
    alloc.isMapped = false;
    
    if (config_.enableDeviceAddress) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = buffer;
        alloc.deviceAddress = vkGetBufferDeviceAddress(device_, &addrInfo);
    }
    
    return alloc;
}

std::optional<MigrationOperation> UnifiedMemoryPool::offloadToHost(Allocation& alloc) {
    if (!offloadManager_) {
        VVM_LOG_WARN("offloadToHost called but OffloadManager not initialized");
        return std::nullopt;
    }
    return offloadManager_->offload(alloc);
}

std::optional<MigrationOperation> UnifiedMemoryPool::reloadToDevice(Allocation& alloc) {
    if (!offloadManager_) {
        VVM_LOG_WARN("reloadToDevice called but OffloadManager not initialized");
        return std::nullopt;
    }
    return offloadManager_->reload(alloc);
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
        
        if (block.buddy) {
            stats.largestFreeBlock = std::max(stats.largestFreeBlock, block.buddy->getLargestFree());
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
    auto& block = blocks_[blockIndex];
    if (!block.buddy) {
        VVM_LOG_ERROR("Block %u has no buddy allocator", blockIndex);
        return std::nullopt;
    }
    
    // Align size
    size = alignUp(size, alignment);
    
    // Allocate from buddy allocator
    auto offsetOpt = block.buddy->allocate(size);
    if (!offsetOpt) {
        return std::nullopt;
    }
    
    VkDeviceSize offset = *offsetOpt;
    
    // Create buffer with external memory support if needed
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    // External memory buffer create info for cross-GPU
    VkExternalMemoryBufferCreateInfo extBufferInfo{};
    if (block.exportFd >= 0 || block.exportHandle) {
        extBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        #ifdef VVM_PLATFORM_LINUX
        extBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
                                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        #elif defined(VVM_PLATFORM_WINDOWS)
        extBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT |
                                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
        #endif
        extBufferInfo.pNext = bufferInfo.pNext;
        bufferInfo.pNext = &extBufferInfo;
    }
    
    VkBuffer buffer;
    VkResult result = vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        VVM_LOG_ERROR("vkCreateBuffer failed: %s", vkResultToString(result).c_str());
        block.buddy->deallocate(offset, size);
        return std::nullopt;
    }
    
    // Bind to memory at offset
    VkBindBufferMemoryInfo bindInfo{};
    bindInfo.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
    bindInfo.buffer = buffer;
    bindInfo.memory = block.memory;
    bindInfo.memoryOffset = offset;
    
    VkBindBufferMemoryDeviceGroupInfo deviceGroupInfo{};
    // For cross-GPU, we'd set device indices here
    
    result = vkBindBufferMemory2(device_, 1, &bindInfo);
    if (result != VK_SUCCESS) {
        VVM_LOG_ERROR("vkBindBufferMemory2 failed: %s", vkResultToString(result).c_str());
        vkDestroyBuffer(device_, buffer, nullptr);
        block.buddy->deallocate(offset, size);
        return std::nullopt;
    }
    
    block.used += size;
    
    Allocation alloc;
    alloc.buffer = buffer;
    alloc.memory = block.memory;
    alloc.offset = offset;
    alloc.size = size;
    alloc.blockIndex = blockIndex;
    alloc.isHostVisible = block.isHostVisible;
    alloc.isMapped = (block.hostPtr != nullptr);
    alloc.isCoherent = block.isCoherent;
    alloc.memoryFlags = block.memoryFlags;
    alloc.hostPtr = alloc.isHostVisible 
        ? static_cast<char*>(block.hostPtr) + offset 
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
    
    auto& block = blocks_[alloc.blockIndex];
    if (block.buddy) {
        block.buddy->deallocate(alloc.offset, alloc.size);
    }
    block.used -= alloc.size;
}

VkDeviceSize UnifiedMemoryPool::alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace vvm