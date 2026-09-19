#pragma once

// Cross-GPU Memory Sharing
// Includes: MultiGPUPoolManager, ExternalHandleType, ExternalHandle, external memory utilities

#include "vulkan_vm/core.hpp"
#include "vulkan_vm/utils.hpp"
#include "vulkan_vm/cross_gpu/external_semaphore.hpp"

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
    // hostPtr must stay alive (page-aligned host pages owned by the
    // caller, or the manager's own arena below) while the arena exists.
    bool createSharedArena(void* hostPtr, VkDeviceSize size,
                           VkBufferUsageFlags usage =
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // Manager-owned arena: VirtualAlloc'd on Windows, mmap'd on Linux, freed
    // AFTER the pools release their imports (declared before instances_
    // below: reverse-destruction order = the owner dies last).
#if defined(_WIN32) || defined(__linux__)
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

    // ---- Runtime rebalancing ------------------------------------------------
    // Per-instance pressure snapshot for policy brains (placement / tensor
    // layers): pool-used bytes vs device-local heap budget. pressure is
    // used/budget (0 when the budget is unknown). driverUsedBytes is the
    // live driver-side consumption (includes external users like the
    // display compositor, which the pool cannot see).
    struct PoolPressure {
        uint32_t instanceIndex = UINT32_MAX;
        VkDeviceSize usedBytes = 0;
        VkDeviceSize budgetBytes = 0;
        VkDeviceSize driverUsedBytes = 0;
        float pressure = 0.0f;
    };
    std::vector<PoolPressure> poolPressures() const;

    // Move one live allocation src->dst at runtime: allocate on dst, copy,
    // retire the src (Ready token - the synchronous copy is complete on
    // return), and hand back the dst allocation. The caller swaps its
    // handle and pumps collect() on the src pool eventually (also swept
    // opportunistically on entry). All-or-nothing: dst-alloc or copy
    // failure rolls back and the src allocation is untouched (still
    // valid, still owned). A rejected retire (stale handle - caller bug)
    // consumes the handle; pass live allocations.
    // usage/memFlags describe the DST allocation; memFlags == 0 mirrors
    // the src's flags. Only same-process manager pools participate.
    std::optional<Allocation> migrateAllocation(uint32_t srcDeviceIndex,
                                                uint32_t dstDeviceIndex,
                                                Allocation&& alloc,
                                                VkBufferUsageFlags usage,
                                                VkMemoryPropertyFlags memFlags = 0);

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
[[nodiscard]] std::optional<Allocation> importMemory(
    UnifiedMemoryPool& pool,
    ExternalHandle&& handle,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

// Duplicate handle for multi-GPU import (each peer needs own handle)
ExternalHandle duplicateForImport(const ExternalHandle& handle);

// Vendor-specific handle type recommendations
ExternalHandleType getRecommendedHandleType(uint32_t srcVendorId, uint32_t dstVendorId, bool isLinux);

} // namespace vvm
