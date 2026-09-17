#pragma once

// Cross-GPU Memory Sharing
// Includes: MultiGPUPoolManager, ExternalHandleType, ExternalHandle, external memory utilities

#include "vulkan_vm/core.hpp"
#include "vulkan_vm/utils.hpp"

namespace vvm {

// ============================================================================
// External Memory Export/Import Info
// ============================================================================

struct ExternalMemoryExportInfo {
    VkExternalMemoryHandleTypeFlagBits handleType;
    bool exportable = true;
    bool importable = true;
};

// ============================================================================
// Dedicated Allocation Info
// ============================================================================

struct DedicatedAllocationInfo {
    bool requiresDedicatedAllocation = false;
    bool prefersDedicatedAllocation = false;
};

// ============================================================================
// Peer Access
// ============================================================================

struct PeerAccessInfo {
    bool canDirectCopy = false;   // export (src) + import (dst) + GPU copy viable
    bool externalMemorySupported = false;
    ExternalHandleType recommendedType = ExternalHandleType::OpaqueFd;
    std::string notes;
};

// ============================================================================
// Multi-GPU Pool Manager
// ============================================================================

class VVM_API MultiGPUPoolManager {
public:
    static std::optional<MultiGPUPoolManager> create(
        const std::vector<DeviceConfig>& devices,
        const PoolConfig& config,
        uint32_t masterIndex = 0);

    MultiGPUPoolManager() = default;
    MultiGPUPoolManager(const MultiGPUPoolManager&) = delete;
    MultiGPUPoolManager& operator=(const MultiGPUPoolManager&) = delete;
    MultiGPUPoolManager(MultiGPUPoolManager&&) noexcept = default;
    MultiGPUPoolManager& operator=(MultiGPUPoolManager&&) noexcept = default;
    ~MultiGPUPoolManager() = default;
    
    // One call: allocates on master, imports on all peers
    // Returns vector of allocations (one per device), all aliasing same memory
    std::vector<std::optional<Allocation>> allocateDistributed(
        VkDeviceSize size,
        VkBufferUsageFlags usage);
    
    // Peer capability query
    PeerAccessInfo queryPeerAccess(uint32_t srcDeviceIndex, uint32_t dstDeviceIndex) const;
    
    // Direct GPU->GPU copy WITHOUT host staging when possible
    // Falls back to chunked host-staged copy if cross-GPU import fails
    bool copyDeviceToDevice(uint32_t srcDeviceIndex, uint32_t dstDeviceIndex,
                            const Allocation& src, const Allocation& dst,
                            VkDeviceSize srcOffset = 0,
                            VkDeviceSize dstOffset = 0,
                            VkDeviceSize size = VK_WHOLE_SIZE,
                            VkFence fence = VK_NULL_HANDLE);
    
private:
    // Host-staged fallback for copyDeviceToDevice. Used when the cross-GPU
    // external memory import fails (driver/hardware limitation, e.g. dGPU->iGPU
    // on Windows). Copies data through host-visible staging buffers on both
    // devices, chunked at kHostStagedChunkSize to bound peak host memory use.
    // Supports arbitrary 'src' allocation (dedicated OR sub-allocated), which
    // is a relaxation not available to the export/import fast path.
    bool copyDeviceToDeviceHostStaged(uint32_t srcDeviceIndex, uint32_t dstDeviceIndex,
                                      const Allocation& src, const Allocation& dst,
                                      VkDeviceSize srcOffset, VkDeviceSize dstOffset,
                                      VkDeviceSize size, VkFence fence);

public:

    // Synchronize across GPUs
    void submitMigrationBarrier(const std::vector<MigrationOperation>& ops);
    void waitAllIdle();
    
    UnifiedMemoryPool& getPool(uint32_t index) { return instances_[index].pool; }
    const std::vector<GPUInstance>& getInstances() const { return instances_; }

    // ---- Shared host arena (VK_EXT_external_memory_host) ----------------
    // One pinned host allocation imported as VkDeviceMemory on EVERY
    // instance pool: the zero-copy cross-vendor medium (every GPU DMAes
    // into the same RAM; measured 3.4 GiB/s XTX<->1080 Ti with NO CPU
    // memcpy). Requires VK_EXT_external_memory_host enabled on the
    // devices at creation; pools on devices without it are skipped
    // (report in arena().devices with imported=false).
    //
    // copyDeviceToDeviceArena moves src -> arena -> dst in TWO DMA legs
    // (no CPU memcpy, no per-chunk sync floor): the src GPU writes the
    // arena slab, the dst GPU reads it. Falls back to false when either
    // device lacks an import.
    struct ArenaDeviceEntry {
        uint32_t instanceIndex = UINT32_MAX;
        Allocation import;        // dedicated-style tracked import
        bool imported = false;
    };
    // hostPtr must stay alive (VirtualAlloc'd by the caller, or the
    // manager's own arena below) while the arena exists.
    bool createSharedArena(void* hostPtr, VkDeviceSize size,
                           VkBufferUsageFlags usage =
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // Manager-owned arena: VirtualAlloc'd here (Windows), freed AFTER the
    // pools release their imports (declared before instances_ below:
    // reverse-destruction order = the owner dies last).
#ifdef _WIN32
    bool createSharedArena(VkDeviceSize size,
                           VkBufferUsageFlags usage =
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
#endif
    bool hasSharedArena() const { return !arena_.empty(); }
    void* arenaPointer() const { return arenaPointer_; }
    VkDeviceSize arenaSize() const { return arenaSize_; }
    // Arena buffer (TRANSFER|STORAGE) bound on instance i; nullptr when
    // that device did not import.
    VkBuffer arenaBuffer(uint32_t instanceIndex) const;
    // Full import Allocation for instance i (dedicated-style); nullptr when
    // that device did not import.
    const Allocation* arenaImport(uint32_t instanceIndex) const;
    // Zero-copy move: src -> arena slab -> dst. One sync per leg.
    bool copyDeviceToDeviceArena(uint32_t srcDeviceIndex, uint32_t dstDeviceIndex,
                                 const Allocation& src, const Allocation& dst,
                                 VkDeviceSize srcOffset = 0,
                                 VkDeviceSize dstOffset = 0,
                                 VkDeviceSize size = VK_WHOLE_SIZE);

private:
    // Arena lifetime owner: declared BEFORE instances_ so reverse-destruction
    // order frees the host pages AFTER the pools release their imports
    // (vkFreeMemory on an import whose host pages are already freed would
    // have the driver unmapping dead pages). Null = caller-owned arena.
    std::shared_ptr<void> arenaOwner_;
    std::vector<GPUInstance> instances_;
    VkSemaphore timelineSemaphore_ = VK_NULL_HANDLE;
    uint64_t timelineValue_ = 0;
    // Shared arena state (see createSharedArena).
    std::vector<ArenaDeviceEntry> arena_;
    void* arenaPointer_ = nullptr;
    VkDeviceSize arenaSize_ = 0;
};

// ============================================================================
// External Memory Functions
// ============================================================================

// Export Vulkan memory as external handle
// Requires dedicated allocation (allocateDedicatedExportable)
ExternalHandle exportMemory(
    UnifiedMemoryPool& pool,
    const Allocation& allocation,
    ExternalHandleType type);

// Import external handle as Vulkan memory
// Consumes the handle on success
std::optional<Allocation> importMemory(
    UnifiedMemoryPool& pool,
    ExternalHandle&& handle,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

// Duplicate handle for multi-GPU import (each peer needs own handle)
ExternalHandle duplicateForImport(const ExternalHandle& handle);

// Vendor-specific handle type recommendations
ExternalHandleType getRecommendedHandleType(uint32_t srcVendorId, uint32_t dstVendorId, bool isLinux);

} // namespace vvm
