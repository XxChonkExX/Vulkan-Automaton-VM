#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/buddy_allocator.hpp"
#include "vulkan_vm/utils.hpp"
#include "external_memory.hpp"

#include <algorithm>

namespace vvm {

// ============================================================================
// Vendor-specific P2P copy optimization
// ============================================================================

struct VendorP2PCaps {
    bool supportsDirectP2P = false;
    VkExternalMemoryHandleTypeFlagBits optimalExportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    VkExternalMemoryHandleTypeFlagBits optimalImportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    bool requiresDedicatedAllocation = true;
    std::string notes;
};

static VendorP2PCaps getVendorP2PCaps(uint32_t srcVendorId, uint32_t dstVendorId) {
    VendorP2PCaps caps;
    
    // NVIDIA (0x10DE) -> Any
    if (srcVendorId == 0x10DE) {
        caps.supportsDirectP2P = true;
        #ifdef VVM_PLATFORM_WINDOWS
        caps.optimalExportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
        caps.optimalImportHandleType = (dstVendorId == 0x10DE) 
            ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT
            : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        #else
        caps.optimalExportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        caps.optimalImportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        #endif
        caps.notes = "NVIDIA source: D3D12_HEAP (Win) / DMA-BUF (Linux) export";
    }
    // AMD (0x1002) -> Any
    else if (srcVendorId == 0x1002) {
        caps.supportsDirectP2P = true;
        #ifdef VVM_PLATFORM_WINDOWS
        caps.optimalExportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        caps.optimalImportHandleType = (dstVendorId == 0x10DE)
            ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT
            : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        #else
        caps.optimalExportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        caps.optimalImportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        #endif
        caps.notes = "AMD source: OPAQUE_WIN32 (Win) / DMA-BUF (Linux) export";
    }
    // Intel (0x8086) -> Any
    else if (srcVendorId == 0x8086) {
        caps.supportsDirectP2P = true;
        #ifdef VVM_PLATFORM_WINDOWS
        caps.optimalExportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        caps.optimalImportHandleType = (dstVendorId == 0x10DE)
            ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT
            : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        #else
        caps.optimalExportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        caps.optimalImportHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        #endif
        caps.notes = "Intel source: OPAQUE_WIN32 (Win) / DMA-BUF (Linux) export";
    }
    // Unknown vendor - try standard path
    else {
        caps.supportsDirectP2P = false;
        caps.notes = "Unknown vendor, using fallback";
    }
    
    return caps;
}

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
        
        // Create import info from export info. Each peer needs its OWN dup'ed
        // handle because a successful import consumes (transfers) one handle.
        auto importInfo = duplicateForImport(*exportInfo);
        importInfo.type = pairCaps.recommendedType;
        importInfo.size = exportInfo->size;
        importInfo.memoryTypeIndex = exportInfo->memoryTypeIndex;
        importInfo.dedicatedAllocation = exportInfo->dedicatedAllocation;
        
        // Handle type conversion for NVIDIA<->AMD/Intel on Windows
        #ifdef VVM_PLATFORM_WINDOWS
        auto masterVendor = getVendorProperties(master.config.physicalDevice);
        auto peerVendor = getVendorProperties(peer.config.physicalDevice);
        bool masterNvidia = (masterVendor.vendorID == 0x10DE);
        bool peerNvidia = (peerVendor.vendorID == 0x10DE);
        
        if (masterNvidia && !peerNvidia) {
            // Master is NVIDIA (exported D3D12_HEAP), peer imports as OPAQUE_WIN32
            importInfo.type = vvm::ExternalHandleType::OpaqueWin32;
        } else if (!masterNvidia && peerNvidia) {
            // Master is AMD/Intel (exported OPAQUE_WIN32), peer imports as D3D12_HEAP
            importInfo.type = vvm::ExternalHandleType::D3D12Heap;
        }
        #endif
        
        auto peerAlloc = peer.pool.importMemory(std::move(importInfo), usage);
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
    
    // exportInfo still owns the ORIGINAL handle; it is not moved/closed here.
    // Its destructor closes it once, after all peers have dup'ed their own.
    // (dup / DuplicateHandle created independent copies for each import.)
    (void)exportInfo;
    
    return results;
}

vvm::PeerAccessInfo MultiGPUPoolManager::queryPeerAccess(
    uint32_t srcDeviceIndex, uint32_t dstDeviceIndex) const {

    PeerAccessInfo info;
    if (srcDeviceIndex >= instances_.size() || dstDeviceIndex >= instances_.size() ||
        srcDeviceIndex == dstDeviceIndex) {
        info.notes = "invalid or identical device indices";
        return info;
    }

    auto srcCaps = queryExternalMemoryCaps(instances_[srcDeviceIndex].config.physicalDevice);
    auto dstCaps = queryExternalMemoryCaps(instances_[dstDeviceIndex].config.physicalDevice);
    auto pair = getCrossVendorCaps(instances_[srcDeviceIndex].config.physicalDevice,
                                   instances_[dstDeviceIndex].config.physicalDevice);

    info.externalMemorySupported = srcCaps.supportedHandleTypes != 0 &&
                                   dstCaps.supportedHandleTypes != 0;
#if defined(VVM_PLATFORM_WINDOWS)
    bool srcWin = srcCaps.supportsOpaqueWin32 || srcCaps.supportsD3D12Heap;
    bool dstWin = dstCaps.supportsOpaqueWin32 || dstCaps.supportsD3D12Heap;
    info.externalMemorySupported = info.externalMemorySupported && srcWin && dstWin;
#endif

    info.recommendedType = pair.recommendedType;
    info.canDirectCopy = info.externalMemorySupported &&
                         pair.recommendedType != ExternalHandleType::OpaqueFd;
    // Do not hard-require a specific type; fall back to OpaqueFd->OpaqueWin32
    // mapping handled by exportMemory. "canDirectCopy" means we have a viable
    // handle type for both sides.
    info.canDirectCopy = info.canDirectCopy || (info.externalMemorySupported &&
                                                pair.sameVendor);
    info.notes = info.canDirectCopy
            ? "direct GPU->GPU copy path available (external memory + device copy)"
            : "external memory not sufficient for direct copy on this pair";
    return info;
}

bool MultiGPUPoolManager::copyDeviceToDevice(
    uint32_t srcDeviceIndex, uint32_t dstDeviceIndex,
    const Allocation& src, const Allocation& dst,
    VkDeviceSize srcOffset, VkDeviceSize dstOffset,
    VkDeviceSize size, VkFence fence) {

    if (srcDeviceIndex >= instances_.size() || dstDeviceIndex >= instances_.size() ||
        srcDeviceIndex == dstDeviceIndex) {
        VVM_LOG_ERROR("copyDeviceToDevice: invalid device indices");
        return false;
    }
    if (!src.buffer || !dst.buffer || src.memory == VK_NULL_HANDLE) {
        VVM_LOG_ERROR("copyDeviceToDevice: invalid source/destination allocation");
        return false;
    }
    // The fast path (export/import) requires a dedicated src allocation.
    // If the src is sub-allocated, skip straight to the host-staged fallback.
    const bool srcIsDedicated = (src.blockIndex == UINT32_MAX);
    if (!srcIsDedicated) {
        VVM_LOG_INFO("copyDeviceToDevice: src is sub-allocated, using host-staged peer copy");
        return copyDeviceToDeviceHostStaged(srcDeviceIndex, dstDeviceIndex,
                                            src, dst, srcOffset, dstOffset, size, fence);
    }

    auto& srcPool = instances_[srcDeviceIndex].pool;
    auto& dstPool = instances_[dstDeviceIndex].pool;

    if (size == VK_WHOLE_SIZE) {
        size = std::min(src.size, dst.size);
    }
    if (srcOffset + size > src.size || dstOffset + size > dst.size) {
        VVM_LOG_ERROR("copyDeviceToDevice: range exceeds allocation size");
        return false;
    }

    // Get vendor-specific P2P capabilities
    auto srcVendorProps = getVendorProperties(instances_[srcDeviceIndex].config.physicalDevice);
    auto dstVendorProps = getVendorProperties(instances_[dstDeviceIndex].config.physicalDevice);
    auto p2pCaps = getVendorP2PCaps(srcVendorProps.vendorID, dstVendorProps.vendorID);
    
    if (!p2pCaps.supportsDirectP2P) {
        VVM_LOG_WARN("copyDeviceToDevice: vendor P2P not supported, falling back to host-staged");
        return copyDeviceToDeviceHostStaged(srcDeviceIndex, dstDeviceIndex,
                                            src, dst, srcOffset, dstOffset, size, fence);
    }

    VVM_LOG_INFO("copyDeviceToDevice: using vendor P2P path: {}", p2pCaps.notes);

    // 1. Export source memory from src device using vendor-optimal handle type.
    auto exportInfo = srcPool.exportMemory(src, 
        static_cast<ExternalHandleType>(p2pCaps.optimalExportHandleType));
    if (!exportInfo) {
        VVM_LOG_WARN("copyDeviceToDevice: exportMemory failed, "
                     "falling back to host-staged peer copy");
        return copyDeviceToDeviceHostStaged(srcDeviceIndex, dstDeviceIndex,
                                            src, dst, srcOffset, dstOffset, size, fence);
    }

    // 2. Import (alias) on the dst device using vendor-optimal import handle type.
    auto importInfo = duplicateForImport(*exportInfo);
    importInfo.type = static_cast<ExternalHandleType>(p2pCaps.optimalImportHandleType);
    auto remote = dstPool.importMemory(std::move(importInfo),
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!remote) {
        VVM_LOG_WARN("copyDeviceToDevice: importMemory failed on dst device, "
                     "falling back to host-staged peer copy (driver/hardware limitation)");
        return copyDeviceToDeviceHostStaged(srcDeviceIndex, dstDeviceIndex,
                                            src, dst, srcOffset, dstOffset, size, fence);
    }

    // 3. Copy remote-alias -> dst allocation on the dst device's queue.
    auto& dev = dstPool.getDeviceConfig();
    VkQueue queue = dev.transferQueue != VK_NULL_HANDLE ? dev.transferQueue
                       : dev.graphicsQueue;
    if (queue == VK_NULL_HANDLE) {
        VVM_LOG_ERROR("copyDeviceToDevice: dst device has no transfer/graphics queue");
        dstPool.deallocate(std::move(*remote));
        return false;
    }

    uint32_t queueFamily = dev.transferQueueFamily != UINT32_MAX
                               ? dev.transferQueueFamily
                               : dev.graphicsQueueFamily;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpInfo.queueFamilyIndex = queueFamily;
    if (vkCreateCommandPool(dev.device, &cpInfo, nullptr, &cmdPool) != VK_SUCCESS) {
        VVM_LOG_ERROR("copyDeviceToDevice: vkCreateCommandPool failed");
        dstPool.deallocate(std::move(*remote));
        return false;
    }

    VkCommandBufferAllocateInfo cba{};
    cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = cmdPool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(dev.device, &cba, &cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(dev.device, cmdPool, nullptr);
        dstPool.deallocate(std::move(*remote));
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkBufferCopy region{};
    region.srcOffset = srcOffset;
    region.dstOffset = dstOffset;
    region.size = size;
    VkResult rc = VK_SUCCESS;
    if ((rc = vkBeginCommandBuffer(cmd, &beginInfo)) != VK_SUCCESS ||
        (vkCmdCopyBuffer(cmd, remote->buffer, dst.buffer, 1, &region),
         (rc = vkEndCommandBuffer(cmd)) != VK_SUCCESS)) {
        VVM_LOG_ERROR("copyDeviceToDevice: command buffer failure rc={}", static_cast<int>(rc));
        vkDestroyCommandPool(dev.device, cmdPool, nullptr);
        dstPool.deallocate(std::move(*remote));
        return false;
    }

    bool waitInternal = (fence == VK_NULL_HANDLE);
    VkFence done = fence;
    VkFence internalFence = VK_NULL_HANDLE;
    if (waitInternal) {
        VkFenceCreateInfo fInfo{};
        fInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(dev.device, &fInfo, nullptr, &internalFence) != VK_SUCCESS) {
            vkDestroyCommandPool(dev.device, cmdPool, nullptr);
            dstPool.deallocate(std::move(*remote));
            return false;
        }
        done = internalFence;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    rc = vkQueueSubmit(queue, 1, &submitInfo, done);

    if (waitInternal && rc == VK_SUCCESS) {
        rc = vkWaitForFences(dev.device, 1, &internalFence, VK_TRUE, UINT64_MAX);
    }

    if (rc != VK_SUCCESS) {
        VVM_LOG_ERROR("copyDeviceToDevice: queue submit/wait failed rc={}", static_cast<int>(rc));
    }

    if (internalFence) vkDestroyFence(dev.device, internalFence, nullptr);
    vkDestroyCommandPool(dev.device, cmdPool, nullptr);
    dstPool.deallocate(std::move(*remote));
    return rc == VK_SUCCESS;
}

// ============================================================================
// Host-staged fallback for copyDeviceToDevice (Spark-style chunked transfer)
// ============================================================================
//
// When the fast export/import path fails (e.g. cross-vendor or dGPU->iGPU on
// Windows where the driver refuses the imported handle), fall back to reading
// the src allocation into a host-visible staging buffer on the src device,
// memcpy'ing it chunk-by-chunk across into a host-visible staging buffer on
// the dst device, and finally copying from that staging buffer into the dst
// allocation. This is the same data layout as the network migrate path, but
// confined to a single process and avoiding the TCP framing overhead. This
// is the "Spark shuffle" path when unified pooling is unavailable.

namespace {
constexpr VkDeviceSize kHostStagedChunkSize = 4ull * 1024 * 1024;  // 4 MiB
}  // namespace

bool MultiGPUPoolManager::copyDeviceToDeviceHostStaged(
    uint32_t srcDeviceIndex, uint32_t dstDeviceIndex,
    const Allocation& src, const Allocation& dst,
    VkDeviceSize srcOffset, VkDeviceSize dstOffset,
    VkDeviceSize size, VkFence fence) {

    if (srcDeviceIndex >= instances_.size() || dstDeviceIndex >= instances_.size() ||
        srcDeviceIndex == dstDeviceIndex) {
        return false;
    }
    if (!src.buffer || !dst.buffer) return false;

    auto& srcPool = instances_[srcDeviceIndex].pool;
    auto& dstPool = instances_[dstDeviceIndex].pool;

    if (size == VK_WHOLE_SIZE) {
        size = std::min(src.size - srcOffset, dst.size - dstOffset);
    }
    if (srcOffset + size > src.size || dstOffset + size > dst.size) {
        VVM_LOG_ERROR("copyDeviceToDeviceHostStaged: range exceeds allocation size");
        return false;
    }

    const VkBufferUsageFlags kStagingUsage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkMemoryPropertyFlags kStagingFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    // Allocate chunk-sized staging buffers on each device. Using chunk size keeps
    // peak host memory bounded regardless of total transfer size.
    const VkDeviceSize chunkSize = std::min(size, kHostStagedChunkSize);

    auto srcStage = srcPool.allocate(chunkSize, kStagingUsage, kStagingFlags);
    auto dstStage = dstPool.allocate(chunkSize, kStagingUsage, kStagingFlags);
    if (!srcStage || !srcStage->hostPtr || !dstStage || !dstStage->hostPtr) {
        VVM_LOG_ERROR("copyDeviceToDeviceHostStaged: failed to allocate host-visible staging buffers");
        if (srcStage) srcPool.deallocate(std::move(*srcStage));
        if (dstStage) dstPool.deallocate(std::move(*dstStage));
        return false;
    }

    bool ok = true;
    VkDeviceSize remaining = size;
    VkDeviceSize srcOff = srcOffset;
    VkDeviceSize dstOff = dstOffset;

    while (remaining > 0 && ok) {
        const VkDeviceSize thisChunk = std::min(remaining, chunkSize);

        // 1. Copy device->host on the source device (src alloc -> src staging).
        if (!srcPool.copyBuffer(src, *srcStage, srcOff, 0, thisChunk, VK_NULL_HANDLE)) {
            VVM_LOG_ERROR("copyDeviceToDeviceHostStaged: src device->host copy failed at offset {}",
                          static_cast<unsigned long long>(srcOff));
            ok = false;
            break;
        }

        // 2. memcpy host buffer -> host buffer (cross-process-safe "TCP" pivot).
        std::memcpy(dstStage->hostPtr, srcStage->hostPtr, static_cast<size_t>(thisChunk));

        // 3. Copy host->device on the destination device (dst staging -> dst alloc).
        if (!dstPool.copyBuffer(*dstStage, dst, 0, dstOff, thisChunk, fence)) {
            VVM_LOG_ERROR("copyDeviceToDeviceHostStaged: host->dst device copy failed at offset {}",
                          static_cast<unsigned long long>(dstOff));
            ok = false;
            break;
        }

        srcOff += thisChunk;
        dstOff += thisChunk;
        remaining -= thisChunk;
    }

    srcPool.deallocate(std::move(*srcStage));
    dstPool.deallocate(std::move(*dstStage));

    if (ok) {
        VVM_LOG_INFO("copyDeviceToDeviceHostStaged: transferred {} bytes via host staging (chunk={})",
                     static_cast<unsigned long long>(size),
                     static_cast<unsigned long long>(chunkSize));
    }
    return ok;
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