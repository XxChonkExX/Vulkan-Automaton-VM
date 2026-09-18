#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/buddy_allocator.hpp"
#include "vulkan_vm/utils.hpp"
#include "vulkan_vm/cross_gpu/external_memory.hpp"
#include <algorithm>
#include <cstdlib>

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

// Map a Vulkan handle-type bit to the library enum. NOTE: ExternalHandleType
// is a sequential enum whose values DO NOT match the VkExternalMemoryHandle-
// TypeFlagBits bitmask values (e.g. DMA_BUF_BIT_EXT == 0x200, DmaBuf == 4).
// A static_cast between them silently produces an out-of-range enum on Linux
// (DMA-BUF) and only works by numeric coincidence for OpaqueWin32.
static ExternalHandleType toExternalHandleType(VkExternalMemoryHandleTypeFlagBits bit) {
    switch (bit) {
        case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT:          return ExternalHandleType::OpaqueFd;
        case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT:       return ExternalHandleType::OpaqueWin32;
        case VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT:         return ExternalHandleType::D3D12Heap;
        case VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:        return ExternalHandleType::DmaBuf;
        case VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID: return ExternalHandleType::AndroidHardwareBuffer;
        default:                                                    return ExternalHandleType::None;
    }
}

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

// ---------------------------------------------------------------------------
// Shared host arena (VK_EXT_external_memory_host): the zero-copy cross-vendor
// medium. One pinned host allocation imported as VkDeviceMemory on every
// instance pool; copyDeviceToDeviceArena moves data in two DMA legs with no
// CPU memcpy. Probe-validated on XTX<->1080 Ti (3.4 GiB/s, tests/
// shared_arena_test).
// ---------------------------------------------------------------------------

bool MultiGPUPoolManager::createSharedArena(void* hostPtr, VkDeviceSize size,
                                            VkBufferUsageFlags usage) {
    if (!hostPtr || size == 0 || instances_.empty()) return false;
    if (hasSharedArena()) {
        VVM_LOG_WARN("createSharedArena: arena already exists ({} bytes); ignoring",
                     arenaSize_ / (1024 * 1024));
        return false;
    }
    arena_.clear();
    arenaPointer_ = hostPtr;
    arenaSize_ = size;
    int imported = 0;
    for (size_t i = 0; i < instances_.size(); ++i) {
        ArenaDeviceEntry entry{};
        entry.instanceIndex = static_cast<uint32_t>(i);
        auto a = instances_[i].pool.importMemoryHostPointer(hostPtr, size, usage);
        if (a.has_value()) {
            entry.import = *a;
            entry.imported = true;
            ++imported;
        } else {
            VVM_LOG_WARN("createSharedArena: instance {} did not import "
                         "(VK_EXT_external_memory_host enabled?)", i);
        }
        arena_.push_back(std::move(entry));
    }
    VVM_LOG_INFO("SharedArena created: {} MB imported on {}/{} instance(s)",
                 size / (1024 * 1024), imported, instances_.size());
    return imported > 0;
}

#ifdef _WIN32
bool MultiGPUPoolManager::createSharedArena(VkDeviceSize size,
                                            VkBufferUsageFlags usage) {
    if (hasSharedArena()) return false;
    // 64 KB-aligned VirtualAlloc (satisfies any sane import alignment).
    void* arena = VirtualAlloc(nullptr, static_cast<SIZE_T>(size),
                               MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!arena) {
        VVM_LOG_ERROR("createSharedArena: VirtualAlloc failed ({} MB)",
                      size / (1024 * 1024));
        return false;
    }
    if (!createSharedArena(arena, size, usage)) {
        VirtualFree(arena, 0, MEM_RELEASE);
        return false;
    }
    // Ownership: freed AFTER the pools (declared before instances_ =
    // reverse-destruction order frees it last).
    arenaOwner_ = std::shared_ptr<void>(arena, [](void* p) {
        if (p) VirtualFree(p, 0, MEM_RELEASE);
    });
    return true;
}
#endif

VkBuffer MultiGPUPoolManager::arenaBuffer(uint32_t instanceIndex) const {
    const Allocation* a = arenaImport(instanceIndex);
    return a ? a->buffer : VK_NULL_HANDLE;
}

const Allocation* MultiGPUPoolManager::arenaImport(uint32_t instanceIndex) const {
    for (const auto& e : arena_) {
        if (e.instanceIndex == instanceIndex && e.imported) {
            return &e.import;
        }
    }
    return nullptr;
}

bool MultiGPUPoolManager::copyDeviceToDeviceArena(
    uint32_t srcDeviceIndex, uint32_t dstDeviceIndex,
    const Allocation& src, const Allocation& dst,
    VkDeviceSize srcOffset, VkDeviceSize dstOffset, VkDeviceSize size) {
    if (srcDeviceIndex >= instances_.size() || dstDeviceIndex >= instances_.size() ||
        srcDeviceIndex == dstDeviceIndex) {
        VVM_LOG_ERROR("copyDeviceToDeviceArena: invalid device indices");
        return false;
    }
    const Allocation* srcArena = arenaImport(srcDeviceIndex);
    const Allocation* dstArena = arenaImport(dstDeviceIndex);
    if (!srcArena || !dstArena) {
        VVM_LOG_ERROR("copyDeviceToDeviceArena: missing arena import on src or dst");
        return false;
    }
    if (!src.buffer || !dst.buffer) return false;

    auto& srcPool = instances_[srcDeviceIndex].pool;
    auto& dstPool = instances_[dstDeviceIndex].pool;

    if (size == VK_WHOLE_SIZE) {
        size = std::min({src.size - srcOffset, dst.size - dstOffset, arenaSize_});
    }
    if (srcOffset + size > src.size || dstOffset + size > dst.size ||
        size > arenaSize_) {
        VVM_LOG_ERROR("copyDeviceToDeviceArena: range exceeds allocation/arena size");
        return false;
    }

    // Two DMA legs through the same physical pages: the src GPU writes its
    // arena view, the dst GPU reads its own. No CPU memcpy, one sync per
    // leg (copyBuffer fences internally).
    if (!srcPool.copyBuffer(src, *srcArena, srcOffset, 0, size)) {
        VVM_LOG_ERROR("copyDeviceToDeviceArena: src -> arena leg failed");
        return false;
    }
    if (!dstPool.copyBuffer(*dstArena, dst, 0, dstOffset, size)) {
        VVM_LOG_ERROR("copyDeviceToDeviceArena: arena -> dst leg failed");
        return false;
    }
    return true;
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
#if defined(VVM_PLATFORM_WINDOWS)
        // Cross-vendor import is refused on Windows - sometimes as a driver
        // crash (AMD master -> Intel peer AVs in vkAllocateMemory). Same-
        // vendor only on Windows; Linux dma-buf cross-vendor stays enabled.
        {
            const auto masterV = getVendorProperties(master.config.physicalDevice);
            const auto peerV = getVendorProperties(peer.config.physicalDevice);
            if (masterV.vendorID != peerV.vendorID) {
                VVM_LOG_WARN("allocateDistributed: peer {} is cross-vendor; "
                             "direct import unsupported on Windows, skipping", i);
                continue;   // results[i] stays nullopt - caller handles partial
            }
        }
#endif
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
    // Opaque external handles are driver-private memory: a cross-vendor
    // import is refused by the destination driver (measured on Windows:
    // AMD->NVIDIA import OOMs, NVIDIA reports no exportable type for the
    // same usage). Gate the fast path on same-vendor; the copy entry point
    // still falls back to host-staged automatically. (No handle-type
    // hard requirement beyond that; OpaqueFd->OpaqueWin32 mapping is
    // handled by exportMemory.)
    info.canDirectCopy = info.externalMemorySupported && pair.sameVendor;
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

#if defined(VVM_PLATFORM_WINDOWS)
    // Cross-vendor opaque-handle import is refused on Windows: gracefully by
    // some drivers (AMD->NVIDIA returns VkResult -13), by a driver CRASH on
    // others (AMD->Intel AVs inside vkAllocateMemory - found via p2p_xn_test
    // on the Arc Pro B70). The verified cross-vendor paths are host-staged
    // copies and the shared host arena; direct import stays same-vendor on
    // Windows. Linux dma-buf cross-vendor is real and unaffected
    // (docs/LINUX_TEST_RESULTS_2026-08-25.md).
    if (srcVendorProps.vendorID != dstVendorProps.vendorID) {
        VVM_LOG_INFO("copyDeviceToDevice: cross-vendor direct import unsupported "
                     "on Windows, using host-staged");
        return copyDeviceToDeviceHostStaged(srcDeviceIndex, dstDeviceIndex,
                                            src, dst, srcOffset, dstOffset, size, fence);
    }
#endif

    VVM_LOG_INFO("copyDeviceToDevice: using vendor P2P path: {}", p2pCaps.notes);

    // 1. Export source memory from src device using vendor-optimal handle type.
    auto exportInfo = srcPool.exportMemory(src,
        toExternalHandleType(p2pCaps.optimalExportHandleType));
    if (!exportInfo) {
        VVM_LOG_WARN("copyDeviceToDevice: exportMemory failed, "
                     "falling back to host-staged peer copy");
        return copyDeviceToDeviceHostStaged(srcDeviceIndex, dstDeviceIndex,
                                            src, dst, srcOffset, dstOffset, size, fence);
    }

    // 2. Import (alias) on the dst device using vendor-optimal import handle type.
    auto importInfo = duplicateForImport(*exportInfo);
    importInfo.type = toExternalHandleType(p2pCaps.optimalImportHandleType);
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

    VkFence done = fence;
    VkFence internalFence = VK_NULL_HANDLE;
    if (fence == VK_NULL_HANDLE) {
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

    // Always wait before tearing down the command pool and the imported
    // alias: with a caller-provided fence, skipping an internal wait would
    // destroy the command buffer and the remote-alias memory while the GPU
    // copy is still reading it.
    if (rc == VK_SUCCESS) {
        rc = vkWaitForFences(dev.device, 1, &done, VK_TRUE, UINT64_MAX);
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
// Staging chunk size: each chunk costs two full submit+wait round trips
// (copyBuffer is transient per call), so small chunks cap large transfers
// (measured XN: 4 MiB -> 1.9, 16 MiB -> 2.8, 64 MiB -> 3.3 GiB/s).
// Default 16 MiB (32 MiB peak host); override via VVM_STAGED_CHUNK_MB.
VkDeviceSize stagedChunkSize() {
    static const VkDeviceSize v = [] {
        if (const char* e = std::getenv("VVM_STAGED_CHUNK_MB")) {
            const unsigned long long mb = std::strtoull(e, nullptr, 0);
            if (mb >= 1 && mb <= 1024) return static_cast<VkDeviceSize>(mb) << 20;
        }
        return 16ull * 1024 * 1024;
    }();
    return v;
}
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
    // Staging must be HOST_CACHED: the CPU reads/writes every chunk via
    // memcpy, and write-combined VRAM (ReBAR types, which first-match
    // selection prefers) reads at ~40 MB/s. Cached system RAM (GART)
    // memcpys at GB/s; the GPU DMA legs are unaffected. Measured 41 MiB/s
    // -> cached on XTX<->Ti legs (2026-09-17 probe).
    const VkMemoryPropertyFlags kStagingFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    // Allocate chunk-sized staging buffers on each device. Using chunk size keeps
    // peak host memory bounded regardless of total transfer size.
    const VkDeviceSize chunkSize = std::min(size, stagedChunkSize());

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
    
    if (instances_.empty()) return;
    (void)ops;
    
    // Signal timeline semaphore on master
    timelineValue_++;
    
    // A VkSemaphore is device-scoped: submitting waits for the master's
    // semaphore on peer-device queues is invalid (VUID-VkSubmitInfo-
    // waitSemaphore-00061). Sharing it across devices would require external
    // semaphore import, which is not wired here. Implement the barrier
    // host-side instead: every device must be idle before the barrier
    // returns, which is a strict superset of the intended ordering.
    for (auto& instance : instances_) {
        VkQueue queue = instance.config.transferQueue != VK_NULL_HANDLE
            ? instance.config.transferQueue
            : instance.config.graphicsQueue;
        if (queue != VK_NULL_HANDLE) {
            vkQueueWaitIdle(queue);
        }
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