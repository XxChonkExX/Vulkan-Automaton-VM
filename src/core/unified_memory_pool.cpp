#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/buddy_allocator.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>

namespace vvm {

// ============================================================================
// UnifiedMemoryPool Implementation
// ============================================================================

MemoryTopology detectMemoryTopology(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);

    bool hasUnifiedType = false;   // DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
    bool hasDevLocal = false;
    uint32_t devLocalHeaps = 0;
    for (uint32_t h = 0; h < props.memoryHeapCount; ++h) {
        const auto& heap = props.memoryHeaps[h];
        const bool heapDevLocal = (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        if (heapDevLocal) devLocalHeaps++;
        for (uint32_t t = 0; t < props.memoryTypeCount; ++t) {
            const auto& mt = props.memoryTypes[t];
            if (mt.heapIndex != h) continue;
            if (heapDevLocal) hasDevLocal = true;
            const VkMemoryPropertyFlags f = mt.propertyFlags;
            if (heapDevLocal &&
                (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                hasUnifiedType = true;
            }
        }
    }

    if (!hasDevLocal) return MemoryTopology::Discrete;
    if (devLocalHeaps <= 1 && hasUnifiedType) return MemoryTopology::Unified;
    if (hasUnifiedType) return MemoryTopology::Hybrid;
    return MemoryTopology::Discrete;
}

PoolConfig PoolConfig::forDevice(VkPhysicalDevice physicalDevice) {
    PoolConfig cfg;
    switch (detectMemoryTopology(physicalDevice)) {
        case MemoryTopology::Unified:
            // APU / Strix Halo style: one shared heap. Bigger blocks, fewer of
            // them; host shadow is largely unnecessary since VRAM is
            // host-visible. Still cap the fraction so we don't starve the OS.
            cfg.blockSize = 1024ull * 1024 * 1024;  // 1 GB
            cfg.maxBlocks = 8;
            cfg.enableHostVisible = true;
            cfg.maxHeapFraction = 0.7f;
            cfg.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
        case MemoryTopology::Hybrid:
            cfg.blockSize = 512ull * 1024 * 1024;
            cfg.maxBlocks = 12;
            cfg.maxHeapFraction = 0.75f;
            break;
        case MemoryTopology::Discrete:
        default:
            cfg.blockSize = 512ull * 1024 * 1024;
            cfg.maxBlocks = 16;
            cfg.maxHeapFraction = 0.75f;
            break;
    }
    return cfg;
}

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
    , dedicatedAllocations_(std::move(other.dedicatedAllocations_))
    , deviceLocalMemoryType_(other.deviceLocalMemoryType_)
    , hostVisibleMemoryType_(other.hostVisibleMemoryType_)
    , deviceLocalHeapIndex_(other.deviceLocalHeapIndex_)
    , memoryBudgetAvailable_(other.memoryBudgetAvailable_)
    , transferCmdPool_(other.transferCmdPool_)
    , debugUtilsEnabled_(other.debugUtilsEnabled_)
    , fnSetDebugName_(other.fnSetDebugName_)
    , offloadManager_(std::move(other.offloadManager_))
    , mutex_() {
    
    other.device_ = VK_NULL_HANDLE;
    other.transferCmdPool_ = VK_NULL_HANDLE;
    other.fnSetDebugName_ = nullptr;
}

UnifiedMemoryPool& UnifiedMemoryPool::operator=(UnifiedMemoryPool&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    
    // Lock both mutexes to avoid deadlock (self-assignment already handled above)
    std::lock(mutex_, other.mutex_);
    std::lock_guard<std::mutex> lock_this(mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lock_other(other.mutex_, std::adopt_lock);
    
    // Cleanup current
    if (device_) {
        for (auto& alloc : dedicatedAllocations_) {
            if (alloc.buffer) vkDestroyBuffer(device_, alloc.buffer, nullptr);
            if (alloc.memory) vkFreeMemory(device_, alloc.memory, nullptr);
        }
        dedicatedAllocations_.clear();
        
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
    dedicatedAllocations_ = std::move(other.dedicatedAllocations_);
    deviceLocalMemoryType_ = other.deviceLocalMemoryType_;
    hostVisibleMemoryType_ = other.hostVisibleMemoryType_;
    deviceLocalHeapIndex_ = other.deviceLocalHeapIndex_;
    memoryBudgetAvailable_ = other.memoryBudgetAvailable_;
    transferCmdPool_ = other.transferCmdPool_;
    debugUtilsEnabled_ = other.debugUtilsEnabled_;
    fnSetDebugName_ = other.fnSetDebugName_;
    offloadManager_ = std::move(other.offloadManager_);
    // mutex_ is not moved - keep our own
    
    other.device_ = VK_NULL_HANDLE;
    other.transferCmdPool_ = VK_NULL_HANDLE;
    other.fnSetDebugName_ = nullptr;
    return *this;
}

UnifiedMemoryPool::~UnifiedMemoryPool() {
    if (device_) {
        // Clean up dedicated allocations (exportable/imported)
        for (auto& alloc : dedicatedAllocations_) {
            if (alloc.buffer) {
                vkDestroyBuffer(device_, alloc.buffer, nullptr);
            }
            if (alloc.memory) {
                if (alloc.hostPtr) {
                    vkUnmapMemory(device_, alloc.memory);
                }
                vkFreeMemory(device_, alloc.memory, nullptr);
            }
        }
        dedicatedAllocations_.clear();
        
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
    
    // Verify the device was created with the features/extensions the pool needs.
    if (!validateDeviceCapabilities()) {
        return false;
    }
    
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
    
    // Remember which heap the device-local type lives on (for budget checks).
    {
        auto memProps = getDeviceMemoryInfo().memProps;
        if (deviceLocalMemoryType_ < memProps.memoryTypeCount) {
            deviceLocalHeapIndex_ = memProps.memoryTypes[deviceLocalMemoryType_].heapIndex;
        }
    }
    
    // Detect VK_EXT_memory_budget for live budget checks.
    memoryBudgetAvailable_ = checkDeviceExtensionSupport(
        deviceConfig_.physicalDevice, {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME});
    if (memoryBudgetAvailable_) {
        VVM_LOG_INFO("VK_EXT_memory_budget available; budget-aware pool growth enabled");
    }
    
    // VK_EXT_debug_utils: name buffers/memory for RenderDoc/validation.
    debugUtilsEnabled_ = checkDeviceExtensionSupport(
        deviceConfig_.physicalDevice, {VK_EXT_DEBUG_UTILS_EXTENSION_NAME});
    if (debugUtilsEnabled_) {
        fnSetDebugName_ = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(
            device_, "vkSetDebugUtilsObjectNameEXT");
        if (!fnSetDebugName_) debugUtilsEnabled_ = false;
    }
    
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
    
    // Pre-allocate first block (pool blocks are never exportable). Respect the
    // budget cap: do not steal memory past maxHeapFraction / maxPoolBytes.
    if (wouldExceedBudget(config_.blockSize)) {
        VVM_LOG_ERROR("Initial pool block (%llu MB) exceeds configured budget; "
                      "lower blockSize or raise maxHeapFraction/maxPoolBytes",
                      config_.blockSize / (1024 * 1024));
        return false;
    }
    if (!allocateBlock(config_.blockSize, deviceLocalMemoryType_).has_value()) {
        VVM_LOG_ERROR("Failed to allocate initial memory block");
        return false;
    }
    
    // Create OffloadManager if offload is enabled
    if (config_.enableHostVisible) {
        OffloadConfig offloadConfig;
        offloadConfig.hostShadowSize = config_.blockSize * 4;  // Default to 4x block size
        // madvise/mprotect on vkMapMemory regions is unsafe (see OffloadConfig
        // docs); these MUST stay disabled by default.
        offloadConfig.useMadvise = false;
        offloadConfig.useMprotect = false;
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

VkMemoryPropertyFlags UnifiedMemoryPool::usageToFlags(MemoryUsage usage) const {
    switch (usage) {
        case MemoryUsage::GpuOnly:
            return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        case MemoryUsage::CpuToGpu:   // staging / upload
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        case MemoryUsage::GpuToCpu:   // readback
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        case MemoryUsage::CpuCopy:    // HOST_VISIBLE staging
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        case MemoryUsage::Auto:
        default:
            return 0;  // pool default (device-local)
    }
}

bool UnifiedMemoryPool::wouldExceedBudget(VkDeviceSize additionalBytes) const {
    // Note: caller must hold mutex_ if called from within another locked method
    VkDeviceSize currentPool = 0;
    for (const auto& block : blocks_) currentPool += block.size;
    for (const auto& alloc : dedicatedAllocations_) currentPool += alloc.size;

    // Hard byte cap.
    if (config_.maxPoolBytes > 0 && currentPool + additionalBytes > config_.maxPoolBytes) {
        VVM_LOG_WARN("budget: pool (%llu MB) + %llu MB would exceed maxPoolBytes (%llu MB)",
                     currentPool / (1024 * 1024), additionalBytes / (1024 * 1024),
                     config_.maxPoolBytes / (1024 * 1024));
        return true;
    }

    // Heap-fraction cap (VK_EXT_memory_budget when available).
    if (config_.maxHeapFraction > 0.0f) {
        VkDeviceSize heapBudget = 0;
        VkDeviceSize heapUsed = 0;
        VkPhysicalDeviceMemoryProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        if (memoryBudgetAvailable_) {
            props2.pNext = &budget;
        }
        vkGetPhysicalDeviceMemoryProperties2(deviceConfig_.physicalDevice, &props2);
        const auto& memProps = props2.memoryProperties;
        if (deviceLocalHeapIndex_ < memProps.memoryHeapCount) {
            if (memoryBudgetAvailable_ && deviceLocalHeapIndex_ < VK_MAX_MEMORY_HEAPS) {
                heapBudget = budget.heapBudget[deviceLocalHeapIndex_];
                heapUsed = budget.heapUsage[deviceLocalHeapIndex_];
            }
            if (heapBudget == 0) {
                heapBudget = memProps.memoryHeaps[deviceLocalHeapIndex_].size;
            }
        }
        const VkDeviceSize cap = static_cast<VkDeviceSize>(heapBudget * config_.maxHeapFraction);
        if (heapUsed + additionalBytes > cap) {
            VVM_LOG_WARN("budget: heap usage %llu MB + %llu MB would exceed %llu MB (%.0f%% of %llu MB); "
                         "allocate() failing soft instead of stealing VRAM",
                         heapUsed / (1024 * 1024), additionalBytes / (1024 * 1024),
                         cap / (1024 * 1024), config_.maxHeapFraction * 100.0f,
                         heapBudget / (1024 * 1024));
            return true;
        }
    }
    return false;
}

void UnifiedMemoryPool::setDebugName(VkObjectType objectType, uint64_t objectHandle,
                                     const char* name) const {
    if (!debugUtilsEnabled_ || !fnSetDebugName_ || !name || objectHandle == 0) return;
    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = objectType;
    info.objectHandle = objectHandle;
    info.pObjectName = name;
    fnSetDebugName_(device_, &info);
}

bool UnifiedMemoryPool::validateDeviceCapabilities() const {
    bool ok = true;

    // Query 1.2 features (bufferDeviceAddress + timelineSemaphore live here,
    // not in VkPhysicalDeviceFeatures).
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features12;
    vkGetPhysicalDeviceFeatures2(deviceConfig_.physicalDevice, &features2);

    // Device address requires the bufferDeviceAddress feature at device creation.
    if (config_.enableDeviceAddress) {
        if (!features12.bufferDeviceAddress) {
            VVM_LOG_ERROR("PoolConfig.enableDeviceAddress is true but the device was NOT created "
                          "with the bufferDeviceAddress feature enabled");
            ok = false;
        }
    }

    // External memory export/import requires the KHR external memory extensions.
    if (config_.enableExternal) {
        std::vector<const char*> required = {
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        };
        #ifdef VVM_PLATFORM_LINUX
        required.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        #elif defined(VVM_PLATFORM_WINDOWS)
        required.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        #endif
        if (!checkDeviceExtensionSupport(deviceConfig_.physicalDevice, required)) {
            VVM_LOG_ERROR("PoolConfig.enableExternal is true but the device is missing required "
                          "external memory extensions (VK_KHR_external_memory + fd/win32)");
            ok = false;
        }
    }

    // Soft checks: warn but do not fail (graceful fallbacks exist).
    if (!features12.timelineSemaphore) {
        VVM_LOG_WARN("timelineSemaphore feature not enabled; migration sync falls back to fences");
    }
    if (!checkDeviceExtensionSupport(deviceConfig_.physicalDevice,
                                     {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME})) {
        VVM_LOG_WARN("VK_EXT_memory_budget not available; budget-aware selection disabled");
    }

    if (!ok) {
        VVM_LOG_ERROR("Pool creation failed: device was not created with the required Vulkan "
                      "features/extensions. Recreate the VkDevice with them enabled (see "
                      "validateDeviceCapabilities in unified_memory_pool.cpp).");
    }
    return ok;
}

std::optional<VkDeviceMemory> UnifiedMemoryPool::allocateBlock(
    VkDeviceSize size, uint32_t memoryTypeIndex) {
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    
    // Pool blocks are strictly NON-exportable. Cross-GPU sharing must use
    // allocateDedicatedExportable() which gives each exported allocation its
    // own dedicated VkDeviceMemory (required for reliable external import).
    void* pNext = nullptr;
    
    // Device address support for bindless
    VkMemoryAllocateFlagsInfo flagsInfo{};
    if (config_.enableDeviceAddress) {
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flagsInfo.pNext = pNext;
        pNext = &flagsInfo;
    }
    
    // NOTE: Do NOT chain VkMemoryDedicatedAllocateInfo here for sub-allocated blocks.
    // A dedicated allocation is bound to a SINGLE resource. Sub-allocating multiple
    // buffers from one "dedicated" memory violates the spec. Exportable allocations
    // should use allocateDedicatedExportable() instead.
    
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
    
    blocks_.push_back(std::move(block));
    return memory;
}

// Allocate a dedicated VkDeviceMemory for a single exportable buffer.
    // Each exportable allocation gets its own dedicated memory (not sub-allocated).
    // This is required by the Vulkan spec for reliable external memory import.
    std::optional<Allocation> UnifiedMemoryPool::allocateDedicatedExportable(
        VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) {
        
        VVM_LOG_INFO("allocateDedicatedExportable: size={}, usage=0x{:x}, flags=0x{:x}", size, usage, flags);
        
        std::lock_guard<std::mutex> lock(mutex_);
        VVM_LOG_INFO("allocateDedicatedExportable: lock acquired");
        
        size = alignUp(size, config_.minAlignment);
        VVM_LOG_INFO("allocateDedicatedExportable: aligned size={}", size);
        
        // Select memory type - prefer device-local for exportable
        uint32_t memType = deviceLocalMemoryType_;
        if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            memType = (hostVisibleMemoryType_ != UINT32_MAX) ? hostVisibleMemoryType_ : deviceLocalMemoryType_;
        }
        VVM_LOG_INFO("allocateDedicatedExportable: memType={}, deviceLocal={}, hostVisible={}", 
                     memType, deviceLocalMemoryType_, hostVisibleMemoryType_);
        
        // Step 1: Create buffer with VkExternalMemoryBufferCreateInfo
        VkExternalMemoryBufferCreateInfo extBufferInfo{};
        extBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        #ifdef VVM_PLATFORM_LINUX
        extBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
                                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        #elif defined(VVM_PLATFORM_WINDOWS)
        extBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT |
                                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
        #endif
        
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferInfo.pNext = &extBufferInfo;
        
        VkBuffer buffer;
        VkResult result = vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer);
        VVM_LOG_INFO("allocateDedicatedExportable: vkCreateBuffer result={}", result);
        if (result != VK_SUCCESS) {
            VVM_LOG_ERROR("vkCreateBuffer failed for dedicated exportable: %s", vkResultToString(result).c_str());
            return std::nullopt;
        }
        
        // Step 2: Get memory requirements
        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device_, buffer, &memReq);
        VVM_LOG_INFO("allocateDedicatedExportable: memReq.size={}, memReq.memoryTypeBits=0x{:x}", memReq.size, memReq.memoryTypeBits);
        
        // Budget check: fail soft instead of stealing VRAM past the configured cap.
        if (wouldExceedBudget(memReq.size)) {
            VVM_LOG_ERROR("allocateDedicatedExportable: would exceed budget");
            vkDestroyBuffer(device_, buffer, nullptr);
            return std::nullopt;
        }
        VVM_LOG_INFO("allocateDedicatedExportable: budget check passed");
        
        // Step 3: Allocate dedicated memory for this buffer
        VkMemoryDedicatedAllocateInfo dedicatedInfo{};
        dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicatedInfo.buffer = buffer;
        dedicatedInfo.image = VK_NULL_HANDLE;
        
        VkExportMemoryAllocateInfo exportInfo{};
        exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        #ifdef VVM_PLATFORM_LINUX
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
                                 VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        #elif defined(VVM_PLATFORM_WINDOWS)
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT |
                                 VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
        #endif
        exportInfo.pNext = &dedicatedInfo;
        
        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flagsInfo.pNext = &exportInfo;
        
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memType;
        allocInfo.pNext = &flagsInfo;
        
        VkDeviceMemory memory;
        VVM_LOG_INFO("allocateDedicatedExportable: calling vkAllocateMemory (memType={}, size={})", memType, memReq.size);
        result = vkAllocateMemory(device_, &allocInfo, nullptr, &memory);
        VVM_LOG_INFO("allocateDedicatedExportable: vkAllocateMemory result={}", result);
        if (result != VK_SUCCESS) {
            VVM_LOG_ERROR("vkAllocateMemory failed for dedicated exportable: %s", vkResultToString(result).c_str());
            vkDestroyBuffer(device_, buffer, nullptr);
            return std::nullopt;
        }
        VVM_LOG_INFO("allocateDedicatedExportable: vkAllocateMemory succeeded");
        
        // Step 4: Bind buffer to memory
        VkBindBufferMemoryInfo bindInfo{};
        bindInfo.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
        bindInfo.buffer = buffer;
        bindInfo.memory = memory;
        bindInfo.memoryOffset = 0;
        
        result = vkBindBufferMemory2(device_, 1, &bindInfo);
        VVM_LOG_INFO("allocateDedicatedExportable: vkBindBufferMemory2 result={}", result);
        if (result != VK_SUCCESS) {
            VVM_LOG_ERROR("vkBindBufferMemory2 failed for dedicated exportable: %s", vkResultToString(result).c_str());
            vkFreeMemory(device_, memory, nullptr);
            vkDestroyBuffer(device_, buffer, nullptr);
            return std::nullopt;
        }
VVM_LOG_INFO("allocateDedicatedExportable: bind succeeded");
     
    // Step 5: Map if host-visible
    VVM_LOG_INFO("allocateDedicatedExportable: checking host visibility (memType={})", memType);
    VVM_LOG_INFO("allocateDedicatedExportable: calling getMemoryTypeProperties");
    void* hostPtr = nullptr;
    VkMemoryPropertyFlags memFlags;
    getMemoryTypeProperties(memType, memFlags, getDeviceMemoryInfo().memProps);
    VVM_LOG_INFO("allocateDedicatedExportable: getMemoryTypeProperties returned, memFlags=0x{:x}", memFlags);
    
    bool isHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    bool isCoherent = (memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    VVM_LOG_INFO("allocateDedicatedExportable: isHostVisible={}, isCoherent={}", isHostVisible, isCoherent);
    
    if (isHostVisible) {
        VVM_LOG_INFO("allocateDedicatedExportable: calling vkMapMemory");
        result = vkMapMemory(device_, memory, 0, VK_WHOLE_SIZE, 0, &hostPtr);
        VVM_LOG_INFO("allocateDedicatedExportable: vkMapMemory result={}", result);
        if (result != VK_SUCCESS) {
            VVM_LOG_ERROR("vkMapMemory failed for dedicated exportable: %s (will not be mappable)", 
                          vkResultToString(result).c_str());
            hostPtr = nullptr;
            // Don't clear isHostVisible - the memory type IS host-visible, just mapping failed
        }
        VVM_LOG_INFO("allocateDedicatedExportable: hostPtr={}", hostPtr ? "non-null" : "null");
    } else {
        VVM_LOG_INFO("allocateDedicatedExportable: memory NOT host-visible, skipping vkMapMemory");
    }
    
    // Step 6: Get device address
    VkDeviceAddress deviceAddress = 0;
    if (config_.enableDeviceAddress) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = buffer;
        deviceAddress = vkGetBufferDeviceAddress(device_, &addrInfo);
    }
    
    // Step 7: Create Allocation (blockIndex = UINT32_MAX indicates dedicated)
    Allocation alloc;
    alloc.buffer = buffer;
    alloc.memory = memory;
    alloc.offset = 0;
    alloc.size = size;
    alloc.blockIndex = UINT32_MAX;  // Special marker for dedicated allocation
    alloc.isHostVisible = isHostVisible;
    alloc.isMapped = isHostVisible;
    alloc.isCoherent = isCoherent;
    alloc.isExternal = true;
    alloc.memoryFlags = memFlags;
    alloc.hostPtr = hostPtr;
    alloc.deviceAddress = deviceAddress;
    
    VVM_LOG_INFO("allocateDedicatedExportable: pushing to dedicatedAllocations_");
    // Track dedicated allocation for cleanup in destructor
    dedicatedAllocations_.push_back(alloc);
    
    VVM_LOG_INFO("allocateDedicatedExportable: returning allocation (hostPtr={})", hostPtr ? "non-null" : "null");
    return alloc;
}

std::optional<Allocation> UnifiedMemoryPool::allocate(VkDeviceSize size,
                                                        VkBufferUsageFlags usage,
                                                        VkMemoryPropertyFlags flags) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Align size
    size = alignUp(size, config_.minAlignment);
    
    // Try existing blocks first - check buddy allocator
    for (uint32_t i = 0; i < blocks_.size(); ++i) {
        if (blocks_[i].buddy && blocks_[i].buddy->getLargestFree() >= size) {
            return subAllocate(size, config_.minAlignment, i, usage);
        }
    }
    
    // Need new block
    if (blocks_.size() >= config_.maxBlocks) {
        return std::nullopt;
    }
    
    // Budget check: fail soft instead of stealing VRAM past the configured cap.
    if (wouldExceedBudget(config_.blockSize)) {
        return std::nullopt;
    }
    
    uint32_t memType = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) 
        ? (hostVisibleMemoryType_ != UINT32_MAX ? hostVisibleMemoryType_ : deviceLocalMemoryType_)
        : deviceLocalMemoryType_;
    
    if (!allocateBlock(config_.blockSize, memType).has_value()) {
        return std::nullopt;
    }
    
    return subAllocate(size, config_.minAlignment, blocks_.size() - 1, usage);
}

std::optional<Allocation> UnifiedMemoryPool::allocate(const AllocDesc& desc) {
    if (desc.exportable) {
        auto alloc = allocateDedicatedExportable(desc.size, desc.usage,
                                                 usageToFlags(desc.memoryUsage));
        if (alloc && desc.name) {
            setDebugName(VK_OBJECT_TYPE_BUFFER, (uint64_t)alloc->buffer, desc.name);
            setDebugName(VK_OBJECT_TYPE_DEVICE_MEMORY, (uint64_t)alloc->memory, desc.name);
        }
        return alloc;
    }
    auto alloc = allocate(desc.size, desc.usage, usageToFlags(desc.memoryUsage));
    if (alloc && desc.name) {
        setDebugName(VK_OBJECT_TYPE_BUFFER, (uint64_t)alloc->buffer, desc.name);
    }
    return alloc;
}

std::optional<Allocation> UnifiedMemoryPool::allocateTensor(VkDeviceSize size,
                                                             VkBufferUsageFlags usage) {
    return allocate(size, usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
}

void UnifiedMemoryPool::deallocate(Allocation&& alloc) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (alloc.blockIndex == UINT32_MAX) {
        // Dedicated allocation - destroy directly
        subDeallocate(std::move(alloc));
    } else if (alloc.blockIndex < blocks_.size()) {
        subDeallocate(std::move(alloc));
    }
}

std::optional<ExternalMemoryInfo> UnifiedMemoryPool::exportMemory(
    const Allocation& alloc, ExternalHandleType type) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Only support dedicated allocations for cross-GPU export.
    // Sub-allocated blocks are NOT supported for cross-GPU export because
    // Vulkan external memory import requires dedicated allocations for
    // reliable cross-device import. Sub-allocating from a shared block
    // and then exporting the whole block (or a sub-range) is fragile
    // across vendors and violates the Vulkan spec for reliable import.
    if (alloc.blockIndex != UINT32_MAX) {
        VVM_LOG_ERROR("exportMemory: only dedicated allocations (blockIndex == UINT32_MAX) are supported for cross-GPU export. Sub-allocated blocks are not supported.");
        return std::nullopt;
    }
    
    // Dedicated allocation - export directly from alloc.memory
    ExternalMemoryInfo info;
    info.type = type;
    info.size = alloc.size;
    info.memoryTypeIndex = UINT32_MAX;  // Will be re-selected by importer
    info.dedicatedAllocation = true;
    
    #ifdef VVM_PLATFORM_LINUX
    if (type == ExternalHandleType::OpaqueFd || type == ExternalHandleType::DmaBuf) {
        VkMemoryGetFdInfoKHR fdInfo{};
        fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fdInfo.memory = alloc.memory;
        
        switch (type) {
            case ExternalHandleType::OpaqueFd:
                fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
                break;
            case ExternalHandleType::DmaBuf:
                fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
                break;
            default:
                return std::nullopt;
        }
        
        PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = 
            (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
        if (!vkGetMemoryFdKHR) return std::nullopt;
        
        int fd = -1;
        if (vkGetMemoryFdKHR(device_, &fdInfo, &fd) != VK_SUCCESS) return std::nullopt;
        info.handle = ExternalHandle(fd);  // RAII wrapper takes ownership
        return info;
    }
    #elif defined(VVM_PLATFORM_WINDOWS)
    if (type == ExternalHandleType::OpaqueWin32 || type == ExternalHandleType::D3D12Heap) {
        VkMemoryGetWin32HandleInfoKHR handleInfo{};
        handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handleInfo.memory = alloc.memory;
        
        switch (type) {
            case ExternalHandleType::OpaqueWin32:
                handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
                break;
            case ExternalHandleType::D3D12Heap:
                handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
                break;
            default:
                return std::nullopt;
        }
        
        PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = 
            (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR");
        if (!vkGetMemoryWin32HandleKHR) return std::nullopt;
        
        HANDLE handle = nullptr;
        if (vkGetMemoryWin32HandleKHR(device_, &handleInfo, &handle) != VK_SUCCESS) return std::nullopt;
        info.handle = ExternalHandle(handle);  // RAII wrapper takes ownership
        return info;
    }
    #endif
    
    return std::nullopt;
}

std::optional<Allocation> UnifiedMemoryPool::importMemory(
    ExternalMemoryInfo&& info, VkBufferUsageFlags usage) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Step 1: Determine the VkExternalMemoryHandleTypeFlagBits for import
    VkExternalMemoryHandleTypeFlagBits importHandleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(0);
    #ifdef VVM_PLATFORM_LINUX
    if (info.type == ExternalHandleType::OpaqueFd) {
        importHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    } else if (info.type == ExternalHandleType::DmaBuf) {
        importHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    }
    #elif defined(VVM_PLATFORM_WINDOWS)
    if (info.type == ExternalHandleType::OpaqueWin32) {
        importHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    } else if (info.type == ExternalHandleType::D3D12Heap) {
        importHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
    }
    #endif
    
    if (importHandleType == 0) {
        VVM_LOG_ERROR("Unsupported external handle type for import");
        return std::nullopt;
    }
    
    // Step 2: Re-select memory type on THIS (destination) device
    // NEVER trust the source device's memoryTypeIndex - it's not portable across devices
    VkMemoryPropertyFlags requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT || usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) {
        // For staging buffers, we might need host-visible
    }
    
    auto memTypeOpt = findImportMemoryTypeIndex(
        deviceConfig_.physicalDevice,
        info.memoryTypeIndex,  // Source device's index (hint only)
        requiredFlags,
        importHandleType
    );
    
    if (!memTypeOpt) {
        VVM_LOG_ERROR("Failed to find compatible memory type for import on destination device");
        return std::nullopt;
    }
    uint32_t memoryTypeIndex = *memTypeOpt;
    VVM_LOG_INFO("Import: re-selected memory type index %u on destination device", memoryTypeIndex);
    
    // Step 3: Create buffer FIRST with VkExternalMemoryBufferCreateInfo
    // This is required for imported external memory
    VkExternalMemoryBufferCreateInfo extBufferInfo{};
    extBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extBufferInfo.handleTypes = importHandleType;
    
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = info.size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo.pNext = &extBufferInfo;
    
    VkBuffer buffer;
    VkResult result = vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to create buffer for imported memory: %s", vkResultToString(result).c_str());
        return std::nullopt;
    }
    
    // Step 4: Build import chain with the ACTUAL buffer in VkMemoryDedicatedAllocateInfo
    VkImportMemoryFdInfoKHR importFdInfo{};
    VkImportMemoryWin32HandleInfoKHR importWin32Info{};
    VkMemoryDedicatedAllocateInfo dedicatedInfo{};
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = info.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    
    void* pNext = nullptr;
    
#ifdef VVM_PLATFORM_LINUX
    if (info.type == ExternalHandleType::OpaqueFd && info.handle) {
        importFdInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        importFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        importFdInfo.fd = info.handle.get();  // Use RAII wrapper's get()
        importFdInfo.pNext = pNext;
        pNext = &importFdInfo;
    } else if (info.type == ExternalHandleType::DmaBuf && info.handle) {
        importFdInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        importFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        importFdInfo.fd = info.handle.get();  // Use RAII wrapper's get()
        importFdInfo.pNext = pNext;
        pNext = &importFdInfo;
    }
    #elif defined(VVM_PLATFORM_WINDOWS)
    if (info.type == ExternalHandleType::OpaqueWin32 && info.handle) {
        importWin32Info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        importWin32Info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        importWin32Info.handle = info.handle.get();  // Use RAII wrapper's get()
        importWin32Info.pNext = pNext;
        pNext = &importWin32Info;
    } else if (info.type == ExternalHandleType::D3D12Heap && info.handle) {
        importWin32Info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        importWin32Info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
        importWin32Info.handle = info.handle.get();  // Use RAII wrapper's get()
        importWin32Info.pNext = pNext;
        pNext = &importWin32Info;
    }
    #endif
    
    // Dedicated allocation for imported memory - NOW with the actual buffer!
    dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedInfo.buffer = buffer;
    dedicatedInfo.image = VK_NULL_HANDLE;
    dedicatedInfo.pNext = pNext;
    pNext = &dedicatedInfo;
    
    // Device address support
    VkMemoryAllocateFlagsInfo flagsInfo{};
    if (config_.enableDeviceAddress) {
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flagsInfo.pNext = pNext;
        pNext = &flagsInfo;
    }
    
    allocInfo.pNext = pNext;
    
    VkDeviceMemory memory;
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to allocate memory for import");
        vkDestroyBuffer(device_, buffer, nullptr);
        // Handle NOT consumed: info keeps ownership and its destructor closes it.
        return std::nullopt;
    }
    
    // On success the driver owns the OS handle (FD/HANDLE). Release it from
    // the RAII wrapper so its destructor does NOT double-close it.
    info.handle.release();
    
    // Step 5: Bind buffer to memory
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
    
    // Step 6: Get device address
    VkDeviceAddress deviceAddress = 0;
    if (config_.enableDeviceAddress) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = buffer;
        deviceAddress = vkGetBufferDeviceAddress(device_, &addrInfo);
    }
    
    // Step 7: Create Allocation (blockIndex = UINT32_MAX for dedicated import)
    Allocation alloc;
    alloc.buffer = buffer;
    alloc.memory = memory;
    alloc.offset = 0;
    alloc.size = info.size;
    alloc.blockIndex = UINT32_MAX;  // Dedicated import
    alloc.isHostVisible = false;
    alloc.isMapped = false;
    alloc.isCoherent = false;
    alloc.isExternal = true;
    alloc.memoryFlags = 0;
    alloc.hostPtr = nullptr;
    alloc.deviceAddress = deviceAddress;
    
    // Track dedicated allocation for cleanup in destructor
    dedicatedAllocations_.push_back(alloc);
    
    return alloc;
}

std::optional<MigrationOperation> UnifiedMemoryPool::offloadToHost(Allocation& alloc) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!offloadManager_) {
        VVM_LOG_WARN("offloadToHost called but OffloadManager not initialized");
        return std::nullopt;
    }
    return offloadManager_->offload(alloc);
}

std::optional<MigrationOperation> UnifiedMemoryPool::reloadToDevice(Allocation& alloc) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!offloadManager_) {
        VVM_LOG_WARN("reloadToDevice called but OffloadManager not initialized");
        return std::nullopt;
    }
    return offloadManager_->reload(alloc);
}

void UnifiedMemoryPool::waitMigration(const MigrationOperation& op) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (op.completionFence) {
        vkWaitForFences(device_, 1, &op.completionFence, VK_TRUE, UINT64_MAX);
    }
}

PoolStats UnifiedMemoryPool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    PoolStats stats;
    stats.dedicatedCount = static_cast<uint32_t>(dedicatedAllocations_.size());
    stats.totalCapacity = config_.maxPoolBytes;
    for (const auto& block : blocks_) {
        stats.totalAllocated += block.size;
        stats.totalUsed += block.used;
        stats.totalFree += (block.size - block.used);
        stats.blockCount++;
        stats.totalCapacity += block.size;
        
        if (block.buddy) {
            stats.largestFreeBlock = std::max(stats.largestFreeBlock, block.buddy->getLargestFree());
            stats.allocationCount += static_cast<uint32_t>(block.buddy->getAllocationCount());
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
    // Note: caller must hold mutex_ if called from within a locked method
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
    std::lock_guard<std::mutex> lock(mutex_);
    // The buddy allocator already coalesces adjacent free ranges on every
    // deallocate, and the pool has no registry of user-owned VkBuffer handles
    // for sub-allocated memory, so in-place data migration is not possible
    // here (moving a live sub-allocation would require rebinding user buffers).
    // What we CAN do safely: release any fully-idle blocks back to the driver.
    for (int i = static_cast<int>(blocks_.size()) - 1; i >= 0; --i) {
        if (blocks_.size() <= 1) break;
        auto& block = blocks_[static_cast<size_t>(i)];
        if (block.used != 0) continue;
        if (block.memory) {
            if (block.hostPtr) {
                vkUnmapMemory(device_, block.memory);
            }
            vkFreeMemory(device_, block.memory, nullptr);
        }
        blocks_.erase(blocks_.begin() + i);
    }
    VVM_LOG_INFO("defragment: released idle blocks; %zu block(s) remain "
                 "(sub-allocation compaction requires user-side rebinding "
                 "and is intentionally not performed)",
                 blocks_.size());
}

void UnifiedMemoryPool::trim() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Release empty blocks back to the driver, keeping at least one block alive.
    for (int i = static_cast<int>(blocks_.size()) - 1; i >= 0; --i) {
        if (blocks_.size() <= 1) break;
        auto& block = blocks_[static_cast<size_t>(i)];
        if (block.used != 0) continue;
        if (block.memory) {
            if (block.hostPtr) {
                vkUnmapMemory(device_, block.memory);
            }
            vkFreeMemory(device_, block.memory, nullptr);
        }
        blocks_.erase(blocks_.begin() + i);
    }
}

// ============================================================================
// Private Helpers
// ============================================================================

std::optional<Allocation> UnifiedMemoryPool::subAllocate(VkDeviceSize size,
                                                          VkDeviceSize alignment,
                                                          uint32_t blockIndex,
                                                          VkBufferUsageFlags usage) {
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
    
    // Create buffer with the caller's usage flags (device address is added when
    // enabled at pool creation; buffers in a shared block are NOT exportable).
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    if (config_.enableDeviceAddress) {
        bufferInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
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
    // Dedicated allocations (blockIndex == UINT32_MAX) have their own VkDeviceMemory
    // and VkBuffer - destroy both directly and remove from tracking
    if (alloc.blockIndex == UINT32_MAX) {
        if (alloc.buffer) vkDestroyBuffer(device_, alloc.buffer, nullptr);
        if (alloc.memory) {
            if (alloc.hostPtr) vkUnmapMemory(device_, alloc.memory);
            vkFreeMemory(device_, alloc.memory, nullptr);
        }
        // Remove from dedicatedAllocations_ to prevent double-free in destructor
        dedicatedAllocations_.erase(
            std::remove_if(dedicatedAllocations_.begin(), dedicatedAllocations_.end(),
                [&alloc](const Allocation& a) {
                    return a.buffer == alloc.buffer && a.memory == alloc.memory;
                }),
            dedicatedAllocations_.end());
        return;
    }

    if (alloc.blockIndex >= blocks_.size()) return;

    vkDestroyBuffer(device_, alloc.buffer, nullptr);

    auto& block = blocks_[alloc.blockIndex];
    if (block.buddy) {
        block.buddy->deallocate(alloc.offset, alloc.size);
    }
    block.used -= alloc.size;
}

VkDeviceSize UnifiedMemoryPool::alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    // Validate power-of-2 alignment (required for bitwise rounding)
    if (alignment == 0) return value;
    if ((alignment & (alignment - 1)) != 0) {
        VVM_LOG_ERROR("alignUp: non-power-of-2 alignment %llu; rounding up to next pow2", alignment);
        // Round alignment up to next power-of-2 to maintain correctness
        alignment--;
        alignment |= alignment >> 1;
        alignment |= alignment >> 2;
        alignment |= alignment >> 4;
        alignment |= alignment >> 8;
        alignment |= alignment >> 16;
        alignment |= alignment >> 32;
        alignment++;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace vvm