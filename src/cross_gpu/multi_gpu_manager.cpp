#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"
#include "external_memory.hpp"

#include <algorithm>

namespace vvm {

// ============================================================================
// MultiGPUPoolManager Implementation
// ============================================================================

std::optional<MultiGPUPoolManager> MultiGPUPoolManager::create(
    const std::vector<DeviceConfig>& devices,
    const PoolConfig& config,
    uint32_t masterIndex) {
    
    if (devices.empty() || masterIndex >= devices.size()) {
        return std::nullopt;
    }
    
    MultiGPUPoolManager manager;
    manager.instances_.reserve(devices.size());
    
    // Query vendor properties and external memory caps for each device
    std::vector<ExternalMemoryCaps> deviceCaps;
    deviceCaps.reserve(devices.size());
    
    for (const auto& dev : devices) {
        deviceCaps.push_back(queryExternalMemoryCaps(dev.physicalDevice));
    }
    
    // Determine optimal handle type for cross-vendor sharing
    ExternalHandleType globalHandleType = ExternalHandleType::OpaqueFd;
    bool crossVendor = false;
    
    for (size_t i = 1; i < devices.size(); ++i) {
        auto pairCaps = getCrossVendorCaps(devices[masterIndex].physicalDevice, 
                                           devices[i].physicalDevice);
        if (pairCaps.nvidiaToAmd || pairCaps.nvidiaToIntel || pairCaps.amdToIntel) {
            crossVendor = true;
            globalHandleType = pairCaps.recommendedType;
        }
    }
    
    // Create pools for each device
    for (size_t i = 0; i < devices.size(); ++i) {
        GPUInstance instance;
        instance.config = devices[i];
        instance.deviceIndex = static_cast<uint32_t>(i);
        instance.isMaster = (i == masterIndex);
        
        // Adjust config per device for external sharing
        PoolConfig deviceConfig = config;
        deviceConfig.enableExternal = true;
        
        auto pool = UnifiedMemoryPool::create(devices[i], deviceConfig);
        if (!pool) {
            return std::nullopt;
        }
        instance.pool = std::move(*pool);
        manager.instances_.push_back(std::move(instance));
    }
    
    // Create timeline semaphore for cross-GPU sync
    VkSemaphoreTypeCreateInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineInfo.initialValue = 0;
    
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semInfo.pNext = &timelineInfo;
    
    VkDevice masterDevice = manager.instances_[masterIndex].pool.getDevice();
    if (vkCreateSemaphore(masterDevice, &semInfo, nullptr, &manager.timelineSemaphore_) != VK_SUCCESS) {
        return std::nullopt;
    }
    
    manager.timelineValue_ = 0;
    return manager;
}

std::vector<std::optional<Allocation>> MultiGPUPoolManager::allocateDistributed(
    VkDeviceSize size, VkBufferUsageFlags usage) {
    
    std::vector<std::optional<Allocation>> results(instances_.size());
    
    if (instances_.empty()) return results;
    
    // 1. Allocate on master
    auto& master = instances_[0];
    auto masterAlloc = master.pool.allocate(size, usage);
    if (!masterAlloc) return results;
    
    results[0] = masterAlloc;
    
    // 2. Export from master
    ExternalHandleType exportType = ExternalHandleType::OpaqueFd;
    
    // Determine best export type based on peer devices
    for (size_t i = 1; i < instances_.size(); ++i) {
        auto pairCaps = getCrossVendorCaps(
            master.config.physicalDevice,
            instances_[i].config.physicalDevice);
        exportType = pairCaps.recommendedType;
        break;  // Use first peer's recommendation
    }
    
    auto exportInfo = master.pool.exportMemory(*masterAlloc, exportType);
    if (!exportInfo) {
        master.pool.deallocate(std::move(*masterAlloc));
        results[0] = std::nullopt;
        return results;
    }
    
    // 3. Import on each peer
    for (size_t i = 1; i < instances_.size(); ++i) {
        auto& peer = instances_[i];
        
        // Determine import type for this peer
        auto pairCaps = getCrossVendorCaps(
            master.config.physicalDevice,
            peer.config.physicalDevice);
        
        ExternalMemoryInfo importInfo = *exportInfo;
        importInfo.type = pairCaps.recommendedType;
        
        // Handle type conversion for NVIDIA<->AMD/Intel on Windows
        #ifdef VVM_PLATFORM_WINDOWS
        auto masterVendor = getVendorProperties(master.config.physicalDevice);
        auto peerVendor = getVendorProperties(peer.config.physicalDevice);
        bool masterNvidia = (masterVendor.vendorID == 0x10DE);
        bool peerNvidia = (peerVendor.vendorID == 0x10DE);
        
        if (masterNvidia && !peerNvidia) {
            // Master is NVIDIA (exported D3D12_HEAP), peer imports as OPAQUE_WIN32
            importInfo.type = ExternalHandleType::OpaqueWin32;
        } else if (!masterNvidia && peerNvidia) {
            // Master is AMD/Intel (exported OPAQUE_WIN32), peer imports as D3D12_HEAP
            importInfo.type = ExternalHandleType::D3D12Heap;
        }
        #endif
        
        auto peerAlloc = peer.pool.importMemory(importInfo, usage);
        results[i] = peerAlloc;
        
        if (!peerAlloc) {
            // Cleanup previous imports
            for (size_t j = 1; j < i; ++j) {
                if (results[j]) {
                    instances_[j].pool.deallocate(std::move(*results[j]));
                    results[j] = std::nullopt;
                }
            }
            master.pool.deallocate(std::move(*masterAlloc));
            results[0] = std::nullopt;
            break;
        }
    }
    
    // Close exported handle (master keeps its allocation)
    #ifdef VVM_PLATFORM_LINUX
    if (exportInfo->fd >= 0) close(exportInfo->fd);
    #elif defined(VVM_PLATFORM_WINDOWS)
    if (exportInfo->handle) CloseHandle(exportInfo->handle);
    #endif
    
    return results;
}

void MultiGPUPoolManager::submitMigrationBarrier(
    const std::vector<MigrationOperation>& ops) {
    
    if (ops.empty()) return;
    
    // Signal timeline semaphore on master
    timelineValue_++;
    
    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &timelineValue_;
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineInfo;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &timelineSemaphore_;
    submitInfo.commandBufferCount = 0;
    
    // Submit to master's transfer queue
    auto& master = instances_[0];
    VkQueue queue = master.config.transferQueue != VK_NULL_HANDLE 
        ? master.config.transferQueue 
        : master.config.graphicsQueue;
    
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    
    // Wait on all peer devices
    for (size_t i = 1; i < instances_.size(); ++i) {
        VkTimelineSemaphoreSubmitInfo waitTimelineInfo{};
        waitTimelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        waitTimelineInfo.waitSemaphoreValueCount = 1;
        waitTimelineInfo.pWaitSemaphoreValues = &timelineValue_;
        
        VkSubmitInfo waitSubmitInfo{};
        waitSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        waitSubmitInfo.pNext = &waitTimelineInfo;
        waitSubmitInfo.waitSemaphoreCount = 1;
        waitSubmitInfo.pWaitSemaphores = &timelineSemaphore_;
        waitSubmitInfo.commandBufferCount = 0;
        
        VkQueue peerQueue = instances_[i].config.transferQueue != VK_NULL_HANDLE
            ? instances_[i].config.transferQueue
            : instances_[i].config.graphicsQueue;
        
        vkQueueSubmit(peerQueue, 1, &waitSubmitInfo, VK_NULL_HANDLE);
    }
}

void MultiGPUPoolManager::waitAllIdle() {
    for (auto& instance : instances_) {
        VkQueue queue = instance.config.graphicsQueue;
        if (queue != VK_NULL_HANDLE) {
            vkQueueWaitIdle(queue);
        }
    }
}

// ============================================================================
// Helper: Find compatible memory type for import
// ============================================================================

uint32_t findImportMemoryType(VkPhysicalDevice physicalDevice,
                               VkMemoryPropertyFlags required,
                               uint32_t exportedTypeIndex,
                               const ExternalMemoryCaps& caps) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    
    // First try the exported type index
    if (exportedTypeIndex < memProps.memoryTypeCount) {
        VkMemoryPropertyFlags flags = memProps.memoryTypes[exportedTypeIndex].propertyFlags;
        if ((flags & required) == required) {
            return exportedTypeIndex;
        }
    }
    
    // Fallback: find any compatible type
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memProps.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    
    return UINT32_MAX;
}

} // namespace vvm