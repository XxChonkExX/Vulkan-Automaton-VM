#pragma once

// Offload/Swap - Host shadow buffer, migration engine, offload manager
// Includes core.hpp for Allocation, MigrationOperation, etc.

#include "vulkan_vm/core.hpp"
#include <mutex>

namespace vvm {

// ============================================================================
// Offload Configuration
// ============================================================================

struct OffloadConfig {
    VkDeviceSize hostShadowSize = 4 * 1024 * 1024 * 1024;  // 4GB host shadow
    [[deprecated("unsafe on vkMapMemory memory; use only on user-allocated mmap regions")]]
    bool useMadvise = false;
    [[deprecated("unsafe on vkMapMemory memory; use only on user-allocated mmap regions")]]
    bool useMprotect = false;
    VkQueue transferQueue = VK_NULL_HANDLE;
    uint32_t transferQueueFamily = UINT32_MAX;
    // Mapping lifetime management
    bool persistentMapping = true;  // Keep mapped for coherent access
    bool useCoherentMapping = true; // Use HOST_COHERENT for mapped regions
};

// ============================================================================
// Host Shadow Buffer (for offload/swap)
// ============================================================================

struct HostShadowBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mappedPtr = nullptr;
    VkDeviceSize used = 0;
    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> freeRanges;  // offset, size
    
    // madvise/mprotect support
    #ifdef VVM_PLATFORM_LINUX
    std::vector<int> madviseFds;  // for MADV_DONTNEED tracking
    #endif
};

// Thread Safety: HostShadowManager is NOT thread-safe. All methods must be
// externally synchronized.

class HostShadowManager {
public:
    HostShadowManager(VkPhysicalDevice physicalDevice, VkDevice device, const OffloadConfig& config);
    ~HostShadowManager();
    
    HostShadowManager(const HostShadowManager&) = delete;
    HostShadowManager& operator=(const HostShadowManager&) = delete;
    
    // Allocate region in host shadow
    std::optional<VkDeviceSize> allocateRegion(VkDeviceSize size);
    void freeRegion(VkDeviceSize offset, VkDeviceSize size);
    
    // Map/unmap for CPU access
    void* mapRegion(VkDeviceSize offset, VkDeviceSize size);
    void unmapRegion(VkDeviceSize offset, VkDeviceSize size);
    
    // Advise kernel (Linux ONLY -- unsafe on memory mapped via vkMapMemory).
    // These helpers exist for potential use on user-allocated mmap'd regions
    // only, and are NOT called by the standard offload/reload flow. Calling
    // madvise() on Vulkan driver-owned mappings can corrupt GPU-side data.
    [[deprecated("unsafe on vkMapMemory memory; use only on user-allocated mmap regions")]]
    void adviseDontNeed(VkDeviceSize offset, VkDeviceSize size);
    [[deprecated("unsafe on vkMapMemory memory; use only on user-allocated mmap regions")]]
    void adviseWillNeed(VkDeviceSize offset, VkDeviceSize size);
    [[deprecated("unsafe on vkMapMemory memory; use only on user-allocated mmap regions")]]
    void adviseFree(VkDeviceSize offset, VkDeviceSize size);  // MADV_FREE
    
    // Protect/unprotect (mprotect). Linux ONLY -- unsafe on memory mapped
    // via vkMapMemory, may SIGSEGV the driver. NOT called by offload/reload.
    [[deprecated("unsafe on vkMapMemory memory; use only on user-allocated mmap regions")]]
    void protectRegion(VkDeviceSize offset, VkDeviceSize size);   // PROT_NONE
    [[deprecated("unsafe on vkMapMemory memory; use only on user-allocated mmap regions")]]
    void unprotectRegion(VkDeviceSize offset, VkDeviceSize size); // PROT_READ|WRITE
    
    // Get buffer for copy operations
    VkBuffer getBuffer() const { return shadowBuffer_.buffer; }
    VkDeviceSize getSize() const { return shadowBuffer_.size; }
    VkDeviceSize getUsed() const { return shadowBuffer_.used; }

private:
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    OffloadConfig config_;
    HostShadowBuffer shadowBuffer_;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    
    bool createShadowBuffer();
    void destroyShadowBuffer();
    VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment);
};

// ============================================================================
// Migration Engine (async device <-> host transfers)
// ============================================================================

struct MigrationContext {
    VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore signalSemaphore = VK_NULL_HANDLE;
    VkSemaphore waitSemaphore = VK_NULL_HANDLE;
    uint64_t timelineValue = 0;
    bool inUse = false;
};

// Thread Safety: MigrationEngine is NOT thread-safe. All methods must be
// externally synchronized. The maxConcurrent parameter controls internal
// command buffer/fence pool size, not thread concurrency.

class MigrationEngine {
public:
    MigrationEngine(VkDevice device, VkQueue transferQueue, uint32_t queueFamily, 
                    uint32_t maxConcurrent = 4);
    ~MigrationEngine();
    
    MigrationEngine(const MigrationEngine&) = delete;
    MigrationEngine& operator=(const MigrationEngine&) = delete;
    
    // Submit migration: device -> host (offload) or host -> device (reload)
    struct MigrationRequest {
        Allocation* allocation = nullptr;
        VkDeviceSize srcOffset = 0;
        VkDeviceSize dstOffset = 0;  // in host shadow
        VkDeviceSize size = 0;
        bool toHost = true;  // true = device->host, false = host->device
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkPipelineStageFlags signalStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkBuffer hostShadowBuffer = VK_NULL_HANDLE;  // Host shadow buffer for copy
    };
    
    std::optional<MigrationOperation> submitMigration(const MigrationRequest& req);
    void waitMigration(const MigrationOperation& op);
    bool pollMigration(const MigrationOperation& op);
    
    // Flush pending migrations
    void flush();
    void waitIdle();
    
    uint32_t getPendingCount() const;

private:
    VkDevice device_;
    VkQueue transferQueue_;
    uint32_t queueFamily_;
    uint32_t maxConcurrent_;
    
    std::vector<MigrationContext> contexts_;
    std::vector<MigrationOperation> pendingOps_;
    uint32_t nextContext_ = 0;
    uint64_t timelineValue_ = 0;
    VkSemaphore timelineSemaphore_ = VK_NULL_HANDLE;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkFence contextFence_ = VK_NULL_HANDLE;
    
    std::optional<MigrationContext*> acquireContext();
    void releaseContext(MigrationContext* ctx);
    void submitCopy(MigrationContext* ctx, const MigrationRequest& req);
};

// ============================================================================
// High-Level Offload Manager
// ============================================================================

// Thread Safety: OffloadManager is now thread-safe. All public methods are
// internally mutex-guarded. The internal MigrationEngine and HostShadowManager
// are protected by this mutex.

class UnifiedMemoryPool;  // forward

class OffloadManager {
public:
    OffloadManager(UnifiedMemoryPool* pool, const OffloadConfig& config);
    ~OffloadManager();
    
    // Offload allocation to host (async)
    std::optional<MigrationOperation> offload(Allocation& alloc);
    
    // Reload allocation from host to device (async)
    std::optional<MigrationOperation> reload(Allocation& alloc);
    
    // Synchronous versions
    bool offloadSync(Allocation& alloc, uint64_t timeoutNs = UINT64_MAX);
    bool reloadSync(Allocation& alloc, uint64_t timeoutNs = UINT64_MAX);
    
    // Wait for any pending operation
    void waitAll();
    
    // Stats
    struct Stats {
        uint64_t bytesOffloaded = 0;
        uint64_t bytesReloaded = 0;
        uint32_t activeMigrations = 0;
        uint32_t completedMigrations = 0;
    };
    Stats getStats() const;
    void resetStats();

private:
    UnifiedMemoryPool* pool_;
    OffloadConfig config_;
    std::unique_ptr<HostShadowManager> shadowManager_;
    std::unique_ptr<MigrationEngine> migrationEngine_;
    
    // Protects all mutable state
    mutable std::mutex mutex_;
    
    Stats stats_;
    mutable std::mutex statsMutex_;
};

} // namespace vvm