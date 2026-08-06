#pragma once

#ifdef VVM_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif VVM_PLATFORM_MACOS
#include <TargetConditionals.h>
#endif

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>
#include <optional>
#include <string>
#include <span>
#include <memory>
#include <mutex>

#include "vulkan_vm/buddy_allocator.hpp"
#include "vulkan_vm/utils.hpp"

namespace vvm {

// ============================================================================
// Configuration
// ============================================================================

struct PoolConfig {
    VkDeviceSize blockSize = 512 * 1024 * 1024;           // 512MB per block
    VkDeviceSize minAlignment = 256 * 1024;                // 256KB for tensor cores
    bool enableHostVisible = true;                         // shadow buffer for swap
    bool enableExternal = true;                            // cross-GPU sharing
    bool enableDeviceAddress = true;                       // bindless access
    uint32_t maxBlocks = 16;                               // max blocks per pool
    VkMemoryPropertyFlags preferredFlags = 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
};

struct DeviceConfig {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily = UINT32_MAX;
    uint32_t computeQueueFamily = UINT32_MAX;
    uint32_t transferQueueFamily = UINT32_MAX;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkQueue transferQueue = VK_NULL_HANDLE;
};

// ============================================================================
// Allocation Types
// ============================================================================

struct Allocation {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;  // shared block memory
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    VkDeviceAddress deviceAddress = 0;       // for bindless
    void* hostPtr = nullptr;                 // mapped if HOST_VISIBLE
    uint32_t blockIndex = UINT32_MAX;        // which block owns this
    bool isHostVisible = false;
    bool isExternal = false;
    bool isMapped = false;
    bool isCoherent = false;
    VkMemoryPropertyFlags memoryFlags = 0;
    // Offload tracking: when the allocation is offloaded to the host shadow
    // buffer, this holds the offset within that shadow buffer. UINT64_MAX means
    // the allocation is not currently offloaded.
    VkDeviceSize shadowOffset = static_cast<VkDeviceSize>(-1);

    // Saved from the original pool mapping so reload can restore it.
    void* savedHostPtr = nullptr;
};

// Forward declaration for RAII wrapper
class UnifiedMemoryPool;

// RAII wrapper for Allocation - automatically returns to pool on destruction.
// Usage: auto alloc = pool->allocate(...); UniqueAllocation ua(pool, std::move(alloc));
// Note: Not thread-safe; pool operations must be externally synchronized.
class UniqueAllocation {
public:
    using Deleter = void(*)(UnifiedMemoryPool*, Allocation&&);
    
    UniqueAllocation() = default;
    // NOTE: Use make() factory instead of constructor for proper deleter setup
    UniqueAllocation(UnifiedMemoryPool* pool, Allocation&& alloc)
        : pool_(pool), alloc_(std::move(alloc)), deleter_(nullptr) {}
    
    UniqueAllocation(UniqueAllocation&& other) noexcept
        : pool_(other.pool_), alloc_(std::move(other.alloc_)), deleter_(other.deleter_) {
        other.pool_ = nullptr;
        other.deleter_ = nullptr;
    }
    
    UniqueAllocation& operator=(UniqueAllocation&& other) noexcept {
        if (this != &other) {
            reset();
            pool_ = other.pool_;
            alloc_ = std::move(other.alloc_);
            deleter_ = other.deleter_;
            other.pool_ = nullptr;
            other.deleter_ = nullptr;
        }
        return *this;
    }
    
    ~UniqueAllocation() { reset(); }
    
    void reset() {
        if (pool_ && alloc_.buffer != VK_NULL_HANDLE && deleter_) {
            deleter_(pool_, std::move(alloc_));
            alloc_ = {};
        }
    }
    
    Allocation* get() { return alloc_.buffer != VK_NULL_HANDLE ? &alloc_ : nullptr; }
    const Allocation* get() const { return alloc_.buffer != VK_NULL_HANDLE ? &alloc_ : nullptr; }
    Allocation* operator->() { return get(); }
    const Allocation* operator->() const { return get(); }
    explicit operator bool() const { return alloc_.buffer != VK_NULL_HANDLE; }
    
    // Release ownership without deallocating
    Allocation release() {
        Allocation tmp = std::move(alloc_);
        alloc_ = {};
        pool_ = nullptr;
        deleter_ = nullptr;
        return tmp;
    }
    
    // Factory to create with proper deleter
    static UniqueAllocation make(UnifiedMemoryPool* pool, Allocation&& alloc);
    
private:
    UnifiedMemoryPool* pool_ = nullptr;
    Allocation alloc_;
    Deleter deleter_ = nullptr;
};

struct BlockInfo {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize used = 0;
    VkDeviceSize offset = 0;  // for host-visible shadow
    void* hostPtr = nullptr;
    // NOTE: Pool blocks are NEVER exportable. Cross-GPU sharing must go
    // through allocateDedicatedExportable() -> exportMemory() so every
    // exported allocation has its own dedicated VkDeviceMemory.
    // Buddy allocator for this block (replaces freeRanges)
    std::unique_ptr<class BuddyAllocator> buddy;
    VkMemoryPropertyFlags memoryFlags = 0;
    bool isHostVisible = false;
    bool isCoherent = false;
};

// ============================================================================
// Cross-GPU Sharing
// ============================================================================

// Forward declare ExternalMemoryInfo (defined in utils.hpp)
struct ExternalMemoryInfo;

// Dedicated allocation info for external memory
struct DedicatedAllocationInfo {
    bool requiresDedicatedAllocation = false;
    bool prefersDedicatedAllocation = false;
};

// ============================================================================
// Offload/Swap
// ============================================================================

struct OffloadConfig {
    VkDeviceSize hostShadowSize = 4 * 1024 * 1024 * 1024;  // 4GB host shadow
    bool useMadvise = false;                                // unsafe on vkMapMemory memory; kept only for future user-mmap regions
    bool useMprotect = false;                               // unsafe on vkMapMemory memory; enables page-fault detection only
    VkQueue transferQueue = VK_NULL_HANDLE;
    uint32_t transferQueueFamily = UINT32_MAX;
    // Mapping lifetime management
    bool persistentMapping = true;  // Keep mapped for coherent access
    bool useCoherentMapping = true; // Use HOST_COHERENT for mapped regions
};

struct MigrationOperation {
    Allocation* allocation = nullptr;
    bool toHost = true;          // true = device->host, false = host->device
    VkFence completionFence = VK_NULL_HANDLE;
    VkSemaphore signalSemaphore = VK_NULL_HANDLE;
    VkSemaphore waitSemaphore = VK_NULL_HANDLE;
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags signalStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    // Opaque pointer to the owning MigrationContext. The engine releases this
    // context back to the pool when waitMigration/pollMigration observes the
    // fence signaled. Kept opaque here to avoid a circular include with
    // offload.hpp (which defines MigrationContext).
    void* owningContext = nullptr;
};

// Forward declaration
class OffloadManager;

// ============================================================================
// Statistics
// ============================================================================

struct PoolStats {
    VkDeviceSize totalAllocated = 0;
    VkDeviceSize totalUsed = 0;
    VkDeviceSize totalFree = 0;
    VkDeviceSize largestFreeBlock = 0;
    uint32_t allocationCount = 0;
    uint32_t blockCount = 0;
    float fragmentationRatio = 0.0f;  // free / (free + used) in worst block
};

struct DeviceMemoryInfo {
    VkPhysicalDeviceMemoryProperties memProps;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    std::vector<VkDeviceSize> heapSizes;
    std::vector<VkDeviceSize> heapUsed;
};

// ============================================================================
// Core Interface
// ============================================================================
//
// Thread Safety: UnifiedMemoryPool is NOT thread-safe. All public methods
// must be externally synchronized by the caller. Concurrent calls to
// allocate/deallocate/exportMemory/importMemory/offloadToHost/reloadToDevice
// from multiple threads will result in data races. Use a mutex or other
// synchronization primitive if multi-threaded access is required.

class UnifiedMemoryPool {
public:
    // Factory
    static std::optional<UnifiedMemoryPool> create(const DeviceConfig& device, 
                                                    const PoolConfig& config);
    
    // Non-copyable, movable
    UnifiedMemoryPool(const UnifiedMemoryPool&) = delete;
    UnifiedMemoryPool& operator=(const UnifiedMemoryPool&) = delete;
    UnifiedMemoryPool(UnifiedMemoryPool&&) noexcept;
    UnifiedMemoryPool& operator=(UnifiedMemoryPool&&) noexcept;
    ~UnifiedMemoryPool();

// Allocation
    std::optional<Allocation> allocate(VkDeviceSize size,
                                       VkBufferUsageFlags usage,
                                       VkMemoryPropertyFlags flags = 0);
    
    // Allocate a dedicated VkDeviceMemory for a single exportable buffer.
    // Each exportable allocation gets its own dedicated memory (not sub-allocated).
    // This is required by the Vulkan spec for reliable external memory import.
    std::optional<Allocation> allocateDedicatedExportable(
        VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags flags = 0);
    
    std::optional<Allocation> allocateTensor(VkDeviceSize size,
                                             VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    
    void deallocate(Allocation&& alloc);

    // Cross-GPU: Export/Import
    // NOTE: exportMemory only supports dedicated allocations (created via
    // allocateDedicatedExportable). Sub-allocated blocks are NOT supported
    // for cross-GPU export because Vulkan external memory import requires
    // dedicated allocations for reliable cross-device import.
    std::optional<ExternalMemoryInfo> exportMemory(const Allocation& alloc,
                                                   ExternalHandleType type);
    
    std::optional<Allocation> importMemory(const ExternalMemoryInfo& info,
                                           VkBufferUsageFlags usage);

    // Offload/Swap
    std::optional<MigrationOperation> offloadToHost(Allocation& alloc);
    std::optional<MigrationOperation> reloadToDevice(Allocation& alloc);
    void waitMigration(const MigrationOperation& op);

    // Stats & Info
    PoolStats getStats() const;
    DeviceMemoryInfo getDeviceMemoryInfo() const;
    const PoolConfig& getConfig() const { return config_; }
    const DeviceConfig& getDeviceConfig() const { return deviceConfig_; }
    VkDevice getDevice() const { return device_; }
    VkPhysicalDevice getPhysicalDevice() const { return deviceConfig_.physicalDevice; }

    // Maintenance
    // NOTE: buddy coalescing already merges adjacent free ranges on
    // deallocate; defragment() releases idle blocks (in-place compaction of
    // live sub-allocations is not possible without user-buffer rebinding).
    void defragment();  // release idle blocks; compaction not supported
    void trim();        // release empty blocks back to OS (optional)

private:
    friend class MultiGPUPoolManager;
    friend struct GPUInstance;
    UnifiedMemoryPool() = default;
    bool initialize(const DeviceConfig& device, const PoolConfig& config);
    
    // Verify required Vulkan features/extensions were enabled at device creation.
    bool validateDeviceCapabilities() const;
    
    std::optional<uint32_t> findMemoryType(VkMemoryPropertyFlags required,
                                           VkMemoryPropertyFlags preferred);
    std::optional<VkDeviceMemory> allocateBlock(VkDeviceSize size,
                                                 uint32_t memoryTypeIndex);
    std::optional<Allocation> subAllocate(VkDeviceSize size,
                                          VkDeviceSize alignment,
                                          uint32_t blockIndex,
                                          VkBufferUsageFlags usage);
    void subDeallocate(Allocation&& alloc);
    
    // Buddy allocator helpers
    VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment);
    
    // Thread safety: all public methods are guarded by this mutex
    mutable std::mutex mutex_;
    
    DeviceConfig deviceConfig_;
    PoolConfig config_;
    VkDevice device_ = VK_NULL_HANDLE;
    std::vector<BlockInfo> blocks_;
    // Dedicated allocations (exportable/imported) tracked separately from blocks
    std::vector<Allocation> dedicatedAllocations_;
    uint32_t deviceLocalMemoryType_ = UINT32_MAX;
    uint32_t hostVisibleMemoryType_ = UINT32_MAX;
    VkCommandPool transferCmdPool_ = VK_NULL_HANDLE;
    
    // Offload manager for host swap
    std::unique_ptr<OffloadManager> offloadManager_;
};

// ============================================================================
// Multi-GPU Pool Manager
// ============================================================================

struct GPUInstance {
    UnifiedMemoryPool pool;
    DeviceConfig config;
    uint32_t deviceIndex = 0;
    bool isMaster = false;  // allocates and exports
};

class MultiGPUPoolManager {
public:
    static std::optional<MultiGPUPoolManager> create(
        const std::vector<DeviceConfig>& devices,
        const PoolConfig& config,
        uint32_t masterIndex = 0);

    // Allocate on master, import on others
    std::vector<std::optional<Allocation>> allocateDistributed(VkDeviceSize size,
                                                                VkBufferUsageFlags usage);
    
    // Synchronize across GPUs
    void submitMigrationBarrier(const std::vector<MigrationOperation>& ops);
    void waitAllIdle();

    UnifiedMemoryPool& getPool(uint32_t index) { return instances_[index].pool; }
    const std::vector<GPUInstance>& getInstances() const { return instances_; }

private:
    std::vector<GPUInstance> instances_;
    VkSemaphore timelineSemaphore_ = VK_NULL_HANDLE;
    uint64_t timelineValue_ = 0;
};

// ============================================================================
// Error Handling
// ============================================================================

enum class ErrorCode {
    Success = 0,
    VulkanError,
    OutOfMemory,
    InvalidConfig,
    UnsupportedFeature,
    AllocationFailed,
    ExportFailed,
    ImportFailed,
    MigrationFailed,
    SyncFailed
};

struct Result {
    ErrorCode code = ErrorCode::Success;
    VkResult vkResult = VK_SUCCESS;
    std::string message;
    
    explicit operator bool() const { return code == ErrorCode::Success; }
    static Result success() { return {}; }
    static Result error(ErrorCode c, const std::string& msg, VkResult vk = VK_SUCCESS) {
        return {c, vk, msg};
    }
};

} // namespace vvm

// Inline implementation of UniqueAllocation::make (requires UnifiedMemoryPool to be fully defined)
namespace vvm {
inline void uniqueAllocationDeleter(UnifiedMemoryPool* pool, Allocation&& alloc) {
    if (pool && alloc.buffer != VK_NULL_HANDLE) {
        pool->deallocate(std::move(alloc));
    }
}

inline UniqueAllocation UniqueAllocation::make(UnifiedMemoryPool* pool, Allocation&& alloc) {
    UniqueAllocation ua;
    ua.pool_ = pool;
    ua.alloc_ = std::move(alloc);
    ua.deleter_ = &uniqueAllocationDeleter;
    return ua;
}
}

#include "vulkan_vm/offload.hpp"
#include "vulkan_vm/network.hpp"