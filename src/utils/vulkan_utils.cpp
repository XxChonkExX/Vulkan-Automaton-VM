#include "vulkan_vm/utils.hpp"
#include "vulkan_vm/vulkan_vm.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <sstream>

namespace vvm {

// ============================================================================
// Queue Family Selection
// ============================================================================

QueueFamilies findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    QueueFamilies families;
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        const auto& props = queueFamilies[i];
        
        if (props.queueFlags & VK_QUEUE_GRAPHICS_BIT && !families.graphics) {
            families.graphics = i;
        }
        if (props.queueFlags & VK_QUEUE_COMPUTE_BIT && !families.compute) {
            families.compute = i;
        }
        if ((props.queueFlags & VK_QUEUE_TRANSFER_BIT) && 
            !(props.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            !(props.queueFlags & VK_QUEUE_COMPUTE_BIT) && !families.transfer) {
            families.transfer = i;
        }
        
        if (surface != VK_NULL_HANDLE) {
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
            if (presentSupport && !families.present) {
                families.present = i;
            }
        }
    }
    
    // Fallbacks
    if (!families.transfer) families.transfer = families.compute;
    if (!families.compute) families.compute = families.graphics;
    if (!families.present) families.present = families.graphics;
    
    return families;
}

// ============================================================================
// Memory Type Finding
// ============================================================================

std::optional<uint32_t> findMemoryTypeIndex(const VkPhysicalDeviceMemoryProperties& memProps,
                                             VkMemoryPropertyFlags required,
                                             VkMemoryPropertyFlags preferred) {
    // First pass: exact match with preferred
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memProps.memoryTypes[i].propertyFlags & required) == required &&
            (memProps.memoryTypes[i].propertyFlags & preferred) == preferred) {
            return i;
        }
    }
    
    // Second pass: required only
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memProps.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    
    return std::nullopt;
}

void getMemoryTypeProperties(uint32_t memoryTypeIndex,
                             VkMemoryPropertyFlags& flags,
                             const VkPhysicalDeviceMemoryProperties& memProps) {
    if (memoryTypeIndex < memProps.memoryTypeCount) {
        flags = memProps.memoryTypes[memoryTypeIndex].propertyFlags;
    }
}

std::optional<uint32_t> findImportMemoryTypeIndex(VkPhysicalDevice dstPhysicalDevice,
                                                   uint32_t srcMemoryTypeIndex,
                                                   VkMemoryPropertyFlags requiredFlags,
                                                   VkExternalMemoryHandleTypeFlagBits handleType) {
    (void)srcMemoryTypeIndex;  // Not directly portable, but can be used as hint
    
    // Query destination device's memory properties
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(dstPhysicalDevice, &memProps);
    
    // Query external memory properties for the handle type
    VkPhysicalDeviceExternalBufferInfo extInfo{};
    extInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    extInfo.handleType = handleType;
    extInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    
    VkExternalBufferProperties extProps{};
    extProps.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(dstPhysicalDevice, &extInfo, &extProps);
    
    // Check if the handle type is supported at all
    VkExternalMemoryHandleTypeFlags compatibleHandleTypes = extProps.externalMemoryProperties.compatibleHandleTypes;
    if ((compatibleHandleTypes & handleType) == 0) {
        VVM_LOG_WARN("findImportMemoryTypeIndex: handle type {} not supported on destination device",
                     handleType);
        return std::nullopt;
    }
    
    // Core Vulkan doesn't provide a direct "compatible memory types" bitmask.
    // We iterate all memory types and find one with required flags.
    // In practice, any memory type with the right flags should work if the
    // handle type is supported (per spec, the compatibleHandleTypes check is
    // the primary gate).
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memProps.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags) {
            return i;
        }
    }
    
    VVM_LOG_WARN("findImportMemoryTypeIndex: no memory type with required flags {:#x}",
                 requiredFlags);
    return std::nullopt;
}

// ============================================================================
// Device Enumeration & Selection
// ============================================================================

std::vector<DeviceScore> enumerateDevices(VkInstance instance) {
    std::vector<DeviceScore> devices;
    
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    
    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());
    
    for (auto physicalDevice : physicalDevices) {
        DeviceScore score;
        score.device = physicalDevice;
        
        vkGetPhysicalDeviceProperties(physicalDevice, &score.props);
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &score.memProps);
        
        score.vendorID = score.props.vendorID;
        score.deviceID = score.props.deviceID;
        score.discrete = (score.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
        score.integrated = (score.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
        
        // Scoring: prefer discrete, more VRAM, newer API version
        if (score.discrete) score.score += 1000;
        if (score.integrated) score.score += 100;
        
        for (uint32_t i = 0; i < score.memProps.memoryHeapCount; ++i) {
            if (score.memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                score.score += static_cast<int>(score.memProps.memoryHeaps[i].size / (1024 * 1024));
            }
        }
        
        // Vulkan version bonus
        score.score += (VK_API_VERSION_MAJOR(score.props.apiVersion) * 100 +
                       VK_API_VERSION_MINOR(score.props.apiVersion) * 10);
        
        devices.push_back(score);
    }
    
    std::sort(devices.begin(), devices.end(),
              [](const DeviceScore& a, const DeviceScore& b) {
                  return a.score > b.score;
              });
    
    return devices;
}

std::optional<DeviceScore> selectBestDevice(const std::vector<DeviceScore>& devices,
                                             bool preferDiscrete,
                                             uint32_t minHeapSizeMB) {
    auto fitsHeap = [&](const DeviceScore& dev) {
        bool hasHeap = false;
        for (uint32_t i = 0; i < dev.memProps.memoryHeapCount; ++i) {
            if ((dev.memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
                dev.memProps.memoryHeaps[i].size >= minHeapSizeMB * 1024 * 1024) {
                hasHeap = true;
                break;
            }
        }
        return minHeapSizeMB == 0 || hasHeap;
    };

    for (const auto& dev : devices) {
        if (preferDiscrete && !dev.discrete) continue;
        if (fitsHeap(dev)) return dev;
    }
    if (preferDiscrete) {
        for (const auto& dev : devices) {
            if (fitsHeap(dev)) return dev;
        }
    }
    return std::nullopt;
}

// ============================================================================
// Format Helpers
// ============================================================================

VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice,
                              const std::vector<VkFormat>& candidates,
                              VkImageTiling tiling,
                              VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
        
        VkFormatFeatureFlags supported = (tiling == VK_IMAGE_TILING_LINEAR)
            ? props.linearTilingFeatures
            : props.optimalTilingFeatures;
        
        if ((supported & features) == features) {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

// ============================================================================
// Extension Checking
// ============================================================================

bool checkDeviceExtensionSupport(VkPhysicalDevice device,
                                  const std::vector<const char*>& required) {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, available.data());
    
    for (const char* req : required) {
        bool found = false;
        for (const auto& avail : available) {
            if (strcmp(req, avail.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

bool checkInstanceExtensionSupport(const std::vector<const char*>& required) {
    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, available.data());
    
    for (const char* req : required) {
        bool found = false;
        for (const auto& avail : available) {
            if (strcmp(req, avail.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

// ============================================================================
// Error Strings
// ============================================================================

std::string vkErrorToString(VkResult result) {
    if (result >= 0) return vkResultToString(result);
    return "ERROR: " + vkResultToString(result);
}

void setDebugName(VkDevice device, VkObjectType type, uint64_t handle, const char* name) {
    VkDebugUtilsObjectNameInfoEXT nameInfo{};
    nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    nameInfo.objectType = type;
    nameInfo.objectHandle = handle;
    nameInfo.pObjectName = name;
    
    PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT =
        (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT");
    if (vkSetDebugUtilsObjectNameEXT) {
        vkSetDebugUtilsObjectNameEXT(device, &nameInfo);
    }
}

// ============================================================================
// FencePool
// ============================================================================

FencePool::FencePool(VkDevice device, uint32_t initialSize)
    : device_(device) {
    VkFenceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    
    for (uint32_t i = 0; i < initialSize; ++i) {
        VkFence fence;
        vkCreateFence(device_, &info, nullptr, &fence);
        available_.push_back(fence);
    }
}

FencePool::~FencePool() {
    for (auto fence : available_) vkDestroyFence(device_, fence, nullptr);
    for (auto fence : inUse_) vkDestroyFence(device_, fence, nullptr);
}

VkFence FencePool::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (available_.empty()) {
        VkFenceCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VkFence fence;
        vkCreateFence(device_, &info, nullptr, &fence);
        inUse_.push_back(fence);
        return fence;
    }
    VkFence fence = available_.back();
    available_.pop_back();
    inUse_.push_back(fence);
    return fence;
}

void FencePool::release(VkFence fence) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find(inUse_.begin(), inUse_.end(), fence);
    if (it != inUse_.end()) {
        inUse_.erase(it);
        vkResetFences(device_, 1, &fence);
        available_.push_back(fence);
    }
}

void FencePool::resetAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto fence : inUse_) {
        vkResetFences(device_, 1, &fence);
        available_.push_back(fence);
    }
    inUse_.clear();
}

// ============================================================================
// SemaphorePool
// ============================================================================

SemaphorePool::SemaphorePool(VkDevice device, bool timeline, uint32_t initialSize)
    : device_(device), timeline_(timeline) {
    VkSemaphoreTypeCreateInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineInfo.semaphoreType = timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
    timelineInfo.initialValue = 0;
    
    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = timeline ? &timelineInfo : nullptr;
    
    for (uint32_t i = 0; i < initialSize; ++i) {
        VkSemaphore sem;
        vkCreateSemaphore(device_, &info, nullptr, &sem);
        available_.push_back(sem);
    }
}

SemaphorePool::~SemaphorePool() {
    for (auto sem : available_) vkDestroySemaphore(device_, sem, nullptr);
    for (auto sem : inUse_) vkDestroySemaphore(device_, sem, nullptr);
}

VkSemaphore SemaphorePool::acquire(uint64_t initialValue) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (available_.empty()) {
        VkSemaphoreTypeCreateInfo timelineInfo{};
        timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        timelineInfo.semaphoreType = timeline_ ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
        timelineInfo.initialValue = initialValue;
        
        VkSemaphoreCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        info.pNext = timeline_ ? &timelineInfo : nullptr;
        
        VkSemaphore sem;
        vkCreateSemaphore(device_, &info, nullptr, &sem);
        inUse_.push_back(sem);
        return sem;
    }
    VkSemaphore sem = available_.back();
    available_.pop_back();
    inUse_.push_back(sem);
    return sem;
}

void SemaphorePool::release(VkSemaphore semaphore) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find(inUse_.begin(), inUse_.end(), semaphore);
    if (it != inUse_.end()) {
        inUse_.erase(it);
        available_.push_back(semaphore);
    }
}

// ============================================================================
// CommandBufferPool
// ============================================================================

CommandBufferPool::CommandBufferPool(VkDevice device, uint32_t queueFamily,
                                      VkCommandBufferLevel level, uint32_t initialSize)
    : device_(device), level_(level) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                     VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    vkCreateCommandPool(device_, &poolInfo, nullptr, &pool_);
    
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool_;
    allocInfo.level = level_;
    allocInfo.commandBufferCount = initialSize;
    
    available_.resize(initialSize);
    vkAllocateCommandBuffers(device_, &allocInfo, available_.data());
}

CommandBufferPool::~CommandBufferPool() {
    if (!available_.empty()) {
        vkFreeCommandBuffers(device_, pool_, static_cast<uint32_t>(available_.size()), available_.data());
    }
    if (!inUse_.empty()) {
        vkFreeCommandBuffers(device_, pool_, static_cast<uint32_t>(inUse_.size()), inUse_.data());
    }
    vkDestroyCommandPool(device_, pool_, nullptr);
}

VkCommandBuffer CommandBufferPool::acquire(bool begin) {
    std::lock_guard<std::mutex> lock(mutex_);
    VkCommandBuffer cmd;
    if (available_.empty()) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = pool_;
        allocInfo.level = level_;
        allocInfo.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_, &allocInfo, &cmd);
    } else {
        cmd = available_.back();
        available_.pop_back();
    }
    inUse_.push_back(cmd);
    
    if (begin) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);
    }
    return cmd;
}

void CommandBufferPool::release(VkCommandBuffer cmdBuffer, bool end) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find(inUse_.begin(), inUse_.end(), cmdBuffer);
    if (it != inUse_.end()) {
        inUse_.erase(it);
        if (end) {
            vkEndCommandBuffer(cmdBuffer);
        }
        vkResetCommandBuffer(cmdBuffer, 0);
        available_.push_back(cmdBuffer);
    }
}

void CommandBufferPool::resetAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto cmd : inUse_) {
        vkResetCommandBuffer(cmd, 0);
        available_.push_back(cmd);
    }
    inUse_.clear();
}

VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    
    return cmd;
}

void endSingleTimeCommands(VkDevice device, VkCommandPool pool, VkCommandBuffer cmdBuffer, VkQueue queue) {
    vkEndCommandBuffer(cmdBuffer);
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;
    
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    
    vkFreeCommandBuffers(device, pool, 1, &cmdBuffer);
}

// ============================================================================
// GpuTimer
// ============================================================================

GpuTimer::GpuTimer(VkDevice device, VkQueue queue, uint32_t queueFamily, uint32_t maxFrames)
    : device_(device), queue_(queue), maxFrames_(maxFrames) {
    
    VkQueryPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    poolInfo.queryCount = maxFrames * 64;  // 64 timestamps per frame
    vkCreateQueryPool(device_, &poolInfo, nullptr, &queryPool_);
}

GpuTimer::~GpuTimer() {
    if (queryPool_) vkDestroyQueryPool(device_, queryPool_, nullptr);
}

void GpuTimer::beginFrame() {
    // Reset query pool for this frame
    vkResetQueryPool(device_, queryPool_, currentFrame_ * 64, 64);
    queryCount_ = 0;
}

void GpuTimer::endFrame() {
    // Timestamps are collected in getResults()
    currentFrame_ = (currentFrame_ + 1) % maxFrames_;
}

void GpuTimer::timestamp(const char* name) {
    if (queryCount_ >= 64) return;
    vkCmdWriteTimestamp(VK_NULL_HANDLE, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 
                       queryPool_, currentFrame_ * 64 + queryCount_);
    timestamps_.emplace_back(name, currentFrame_ * 64 + queryCount_);
    queryCount_++;
}

std::vector<GpuTimer::Timestamp> GpuTimer::getResults() {
    std::vector<Timestamp> results;
    results.reserve(timestamps_.size());
    
    if (timestamps_.empty()) return results;
    
    uint32_t firstQuery = timestamps_.front().second;
    uint32_t lastQuery = timestamps_.back().second;
    uint32_t count = lastQuery - firstQuery + 1;
    
    std::vector<uint64_t> data(count);
    vkGetQueryPoolResults(device_, queryPool_, firstQuery, count,
                         data.size() * sizeof(uint64_t), data.data(),
                         sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(VK_NULL_HANDLE, &props);  // Would need physical device
    double period = 1.0;  // props.limits.timestampPeriod;
    
    for (size_t i = 0; i < timestamps_.size(); ++i) {
        uint32_t idx = timestamps_[i].second - firstQuery;
        if (i + 1 < timestamps_.size()) {
            uint32_t nextIdx = timestamps_[i + 1].second - firstQuery;
            double ms = (data[nextIdx] - data[idx]) * period / 1'000'000.0;
            results.push_back({timestamps_[i].first, data[idx], data[nextIdx], ms});
        }
    }
    
    return results;
}

void GpuTimer::reset() {
    timestamps_.clear();
    queryCount_ = 0;
}

// ============================================================================
// MemoryTypeSelector Implementation
// ============================================================================

MemoryTypeSelector::MemoryTypeSelector(VkPhysicalDevice pd)
    : physicalDevice(pd), hasBudgetExt(false) {
    if (physicalDevice != VK_NULL_HANDLE) {
        refresh();
    }
}

void MemoryTypeSelector::refresh() {
    if (physicalDevice == VK_NULL_HANDLE) return;
    
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    
    // Check for memory budget extension
    VkPhysicalDeviceMemoryProperties2 memProps2{};
    memProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memProps2.pNext = &budget;
    
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    budget.pNext = nullptr;
    
    // Query extensions to check if budget is available
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, extensions.data());
    
    for (const auto& ext : extensions) {
        if (strcmp(ext.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
            hasBudgetExt = true;
            break;
        }
    }
    
    if (hasBudgetExt) {
        vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memProps2);
    }
}

MemoryTypeSelector::SelectionResult MemoryTypeSelector::select(
    VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred,
    VkDeviceSize minHeapBudget) const {
    
    SelectionResult best;
    float bestScore = -1.0f;
    
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        const auto& type = memProps.memoryTypes[i];
        
        // Check required flags
        if ((type.propertyFlags & required) != required) continue;
        
        uint32_t heapIndex = type.heapIndex;
        if (heapIndex >= memProps.memoryHeapCount) continue;
        
        const auto& heap = memProps.memoryHeaps[heapIndex];
        
        VkDeviceSize heapBudget = hasBudgetExt ? budget.heapBudget[heapIndex] : heap.size;
        VkDeviceSize heapUsage = hasBudgetExt ? budget.heapUsage[heapIndex] : 0;
        VkDeviceSize available = (heapBudget > heapUsage) ? (heapBudget - heapUsage) : 0;
        
        if (minHeapBudget > 0 && available < minHeapBudget) continue;
        
        float utilization = (heapBudget > 0) ? 
            static_cast<float>(heapUsage) / static_cast<float>(heapBudget) : 0.0f;
        
        // Score: prefer lower utilization, preferred flags match, device-local
        float score = 1.0f - utilization;
        
        if ((type.propertyFlags & preferred) == preferred) score += 0.5f;
        if (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) score += 0.3f;
        if (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) score += 0.1f;
        if (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) score += 0.1f;
        
        if (score > bestScore) {
            bestScore = score;
            best.memoryTypeIndex = i;
            best.heapBudget = heapBudget;
            best.heapUsage = heapUsage;
            best.heapUtilization = utilization;
        }
    }
    
    return best;
}

DedicatedAllocationInfo MemoryTypeSelector::getDedicatedAllocationInfo(
    VkBufferCreateInfo* bufferInfo,
    VkImageCreateInfo* imageInfo) const {
    
    DedicatedAllocationInfo info;
    
    if (!hasBudgetExt) return info;
    
    VkPhysicalDeviceMemoryProperties2 memProps2{};
    memProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memProps2.pNext = nullptr;
    
    // For now, return basic info - full implementation would query
    // VkPhysicalDeviceDedicatedAllocationImageCreateInfoNV etc.
    info.requiresDedicatedAllocation = false;
    info.prefersDedicatedAllocation = false;
    
    return info;
}

// ============================================================================
// Logger
// ============================================================================

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

std::string vkResultToString(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        case VK_PIPELINE_COMPILE_REQUIRED: return "VK_PIPELINE_COMPILE_REQUIRED";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
        case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
        case VK_ERROR_INVALID_SHADER_NV: return "VK_ERROR_INVALID_SHADER_NV";
        case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT: return "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
        case VK_ERROR_NOT_PERMITTED_KHR: return "VK_ERROR_NOT_PERMITTED_KHR";
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: return "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
        case VK_THREAD_IDLE_KHR: return "VK_THREAD_IDLE_KHR";
        case VK_THREAD_DONE_KHR: return "VK_THREAD_DONE_KHR";
        case VK_OPERATION_DEFERRED_KHR: return "VK_OPERATION_DEFERRED_KHR";
        case VK_OPERATION_NOT_DEFERRED_KHR: return "VK_OPERATION_NOT_DEFERRED_KHR";
        case VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR: return "VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR";
        case VK_ERROR_COMPRESSION_EXHAUSTED_EXT: return "VK_ERROR_COMPRESSION_EXHAUSTED_EXT";
        default: return "VK_ERROR_UNKNOWN (" + std::to_string(static_cast<int>(result)) + ")";
    }
}

} // namespace vvm