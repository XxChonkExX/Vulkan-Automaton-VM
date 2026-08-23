#pragma once

// Core VulkanVM types and configurations
// Includes: PoolConfig, DeviceConfig, AllocDesc, MemoryUsage, MemoryTopology,
//           Allocation, BlockInfo, UnifiedMemoryPool (forward), BuddyAllocator

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

// Export macros for shared library
#if defined(VVM_BUILD_SHARED) && defined(VVM_EXPORT)
#  if defined(_MSC_VER)
#    define VVM_API __declspec(dllexport)
#  else
#    define VVM_API __attribute__((visibility("default")))
#  endif
#elif defined(VVM_BUILD_SHARED) && defined(_MSC_VER)
// Consumer of the shared lib on MSVC: import symbols
#  define VVM_API __declspec(dllimport)
#else
#  define VVM_API
#endif

#include <cstdint>
#include <vector>
#include <optional>
#include <string>
#include <span>
#include <memory>
#include <mutex>
#include <functional>
#include <cassert>
#include <unordered_set>

#include "vulkan_vm/buddy_allocator.hpp"
#include "vulkan_vm/utils.hpp"

namespace vvm {

// ============================================================================
// Configuration
// ============================================================================

struct VVM_API PoolConfig {
    // Block size for the buddy allocator (power of two). Default 512MB.
    // For high-VRAM cards (RTX 4090 24GB, RTX 6000 48GB), consider 1GB or 2GB.
    VkDeviceSize blockSize = 512ull * 1024 * 1024;  // 512MB per block
    
    // Optional: multiple block sizes for flexible routing.
    // If set, the allocator picks the smallest block size >= request size.
    // Falls back to blockSize if empty.
    std::vector<VkDeviceSize> blockSizes;
    
    // Minimum allocation alignment (power of two). 256KB for tensor cores.
    VkDeviceSize minAlignment = 256 * 1024;
    
    // Enable host-visible memory for staging/offload
    bool enableHostVisible = true;                         // shadow buffer for swap
    
    // Enable external memory handle types for cross-GPU sharing
    bool enableExternal = true;                            // cross-GPU sharing
    
    // Enable VK_BUFFER_DEVICE_ADDRESS for bindless access
    bool enableDeviceAddress = true;                       // bindless access
    
    // Max number of blocks to allocate (0 = unlimited, limited by maxPoolBytes/heapFraction)
    uint32_t maxBlocks = 16;                               // max blocks per pool
    
    // Preferred memory flags for allocations
    VkMemoryPropertyFlags preferredFlags = 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    
    // Budget: never pre-allocate past this fraction of the heap (0 = disabled).
    // Checked against VK_EXT_memory_budget when available, otherwise the
    // static heap size. When the budget would be exceeded, allocate() fails
    // soft (falls back to offload/host memory) instead of stealing VRAM.
    float maxHeapFraction = 0.0f;

    // Memory priority for block/dedicated allocations (VK_EXT_memory_priority).
    // 0 = default (driver may degrade/evict allocations under heap pressure).
    // >0 chains VkMemoryPriorityAllocateInfoEXT on every vkAllocateMemory.
    // Requires the extension to be ENABLED on the VkDevice at creation time.
    // Inference workloads filling most of VRAM should use 1.0f.
    float memoryPriority = 0.0f;

    // Exclude HOST_VISIBLE types from the device-local selection (ReBAR-mapped
    // VRAM). Some drivers place/map ReBAR memory differently than pure
    // DEVICE_LOCAL; benchmark both when chasing bandwidth parity.
    bool preferPureDeviceLocal = false;
    
    // Hard cap on total pool memory in bytes (0 = no explicit cap).
    VkDeviceSize maxPoolBytes = 0;

    // Host shadow buffer configuration (for offload/staging).
    // Shadow size = blockSize * hostShadowMultiplier, clamped to maxHostShadowBytes.
    // Default multiplier is 4.0 (shadow is 4x block size).
    float hostShadowMultiplier = 4.0f;
    // Hard cap on host shadow size in bytes (0 = no explicit cap, only multiplier applies).
    VkDeviceSize maxHostShadowBytes = 0;
    
    // Build a config tuned for a physical device (Discrete/Hybrid/Unified)
    static PoolConfig forDevice(VkPhysicalDevice physicalDevice);
    
    // APU-specific tuned config
    static PoolConfig forAPU(VkDeviceSize totalSystemRAM);
    
    // High-VRAM tuned config (for 24GB+ cards like RTX 4090, RTX 6000 Ada)
    static PoolConfig forHighVRAM(VkPhysicalDevice physicalDevice);
};

struct VVM_API DeviceConfig {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily = 0;
    uint32_t computeQueueFamily = 0;
    uint32_t transferQueueFamily = 0;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkQueue transferQueue = VK_NULL_HANDLE;
};

// Memory usage intent (hides raw VkMemoryPropertyFlags)
enum class MemoryUsage {
    GpuOnly,      // DEVICE_LOCAL, not mapped
    CpuToGpu,     // HOST_VISIBLE | HOST_COHERENT, mapped, upload
    GpuToCpu,     // HOST_VISIBLE | HOST_COHERENT, mapped, readback
    CpuCopy,      // HOST_VISIBLE, mapped, CPU<->CPU
    Auto          // Auto-select based on usage flags
};

// Allocation descriptor - intent-based API
struct VVM_API AllocDesc {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    MemoryUsage memoryUsage = MemoryUsage::Auto;
    bool exportable = false;
    bool mapped = false;
    std::string name;
};

// Memory topology classification (OpenUMA-style device awareness).
enum class MemoryTopologyType {
    Discrete,  // separate VRAM + system memory
    Unified,   // single heap, DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
    Hybrid,    // APU with a small dedicated carve-out
};

// Classify a physical device's memory layout.
VVM_API MemoryTopologyType detectMemoryTopology(VkPhysicalDevice physicalDevice);

// ============================================================================
// Allocation
// ============================================================================

struct VVM_API Allocation {
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

    // Generation counter for handle validation (prevents stale handle use).
    // Incremented on deallocate; allocation must match current generation to be valid.
    uint64_t generation = 0;
};

// Block metadata (internal)
struct VVM_API BlockInfo {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize used = 0;
    VkDeviceSize offset = 0;
    void* hostPtr = nullptr;
    std::unique_ptr<class BuddyAllocator> buddy;
    VkMemoryPropertyFlags memoryFlags = 0;
    bool isHostVisible = false;
    bool isCoherent = false;
};

// ============================================================================
// Pool Statistics
// ============================================================================

struct VVM_API PoolStats {
    VkDeviceSize totalAllocated = 0;
    VkDeviceSize totalUsed = 0;
    VkDeviceSize totalFree = 0;
    VkDeviceSize largestFreeBlock = 0;
    float fragmentationRatio = 0.0f;
    uint32_t blockCount = 0;
    uint32_t allocationCount = 0;
    uint32_t dedicatedCount = 0;
    VkDeviceSize totalCapacity = 0;
};

struct VVM_API DeviceMemoryInfo {
    VkPhysicalDeviceMemoryProperties memProps;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    std::vector<VkDeviceSize> heapSizes;
    std::vector<VkDeviceSize> heapUsed;
};

using MigrationId = uint64_t;

// ============================================================================
// Migration Operation
// ============================================================================

struct VVM_API MigrationOperation {
    MigrationId id = 0;
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
    // Optional callback invoked after the operation completes (fence signaled).
    // Used for cleanup actions like freeing shadow regions after reload.
    std::function<void()> onComplete;
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

struct VVM_API Result {
    ErrorCode code = ErrorCode::Success;
    VkResult vkResult = VK_SUCCESS;
    std::string message;
    
    explicit operator bool() const { return code == ErrorCode::Success; }
    static Result success() { return {}; }
    static Result error(ErrorCode c, const std::string& msg, VkResult vk = VK_SUCCESS) {
        return {c, vk, msg};
    }
};

// Templated Result for operations that return a value
template<typename T>
struct VVM_API ResultT {
    ErrorCode code = ErrorCode::Success;
    VkResult vkResult = VK_SUCCESS;
    std::string message;
    T value;
    
    explicit operator bool() const { return code == ErrorCode::Success; }
    static ResultT<T> success(T val) { return {ErrorCode::Success, VK_SUCCESS, "", std::move(val)}; }
    static ResultT<T> error(ErrorCode c, const std::string& msg, VkResult vk = VK_SUCCESS) {
        return {c, vk, msg, T{}};
    }
};

class VVM_API OffloadManager;  // forward declaration for UnifiedMemoryPool

// ============================================================================
// Unified Memory Pool (core definition)
// ============================================================================
//
// Thread Safety: All public methods are thread-safe (guarded by internal mutex).
//                Not thread-safe for concurrent move operations (move before use).
//                Use UniqueAllocation for RAII ownership; avoid raw Allocation handles.

class UniqueAllocation;  // forward declaration for deallocate(UniqueAllocation&&)

class VVM_API UnifiedMemoryPool {
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
    
    // Rich allocation descriptor. exportable=true routes to a dedicated
    // exportable allocation; memoryUsage maps to memory types internally.
    std::optional<Allocation> allocate(const AllocDesc& desc);
    
    // Allocate a dedicated VkDeviceMemory for a single exportable buffer.
    // Each exportable allocation gets its own dedicated memory (not sub-allocated).
    // This is required by the Vulkan spec for reliable external memory import.
    std::optional<Allocation> allocateDedicatedExportable(
        VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags flags = 0);
    
    // Allocate a dedicated VkDeviceMemory for a single NON-exportable buffer.
    // Used as the oversized-allocation fallback in allocate(): requests larger
    // than config_.blockSize cannot be served by the buddy blocks, so they get
    // their own VkDeviceMemory instead of failing.
    std::optional<Allocation> allocateDedicated(
        VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags flags = 0);
    
    std::optional<Allocation> allocateTensor(VkDeviceSize size,
                                             VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    
    void deallocate(Allocation&& alloc);
    void deallocate(UniqueAllocation&& alloc);

    // Cross-GPU: Export/Import
    // NOTE: exportMemory only supports dedicated allocations (created via
    // allocateDedicatedExportable). Sub-allocated blocks are NOT supported
    // for cross-GPU export because Vulkan external memory import requires
    // dedicated allocations for reliable cross-device import.
    [[nodiscard]] std::optional<ExternalMemoryInfo> exportMemory(const Allocation& alloc,
                                                       ExternalHandleType type);
    
    // Ownership contract: on SUCCESS the OS handle (FD/HANDLE) is transferred
    // to the driver and info is emptied (importMemory consumes it). Pass an
    // ExternalMemoryInfo that owns a handle you no longer need afterwards,
    // e.g. from exportMemory or duplicateForImport. Use duplicateForImport
    // when the same memory must be imported on multiple peers.
    // On FAILURE, info still owns the handle and its destructor closes it.
    [[nodiscard]] std::optional<Allocation> importMemory(ExternalMemoryInfo&& info,
                                               VkBufferUsageFlags usage);

    // Offload/Swap
    std::optional<MigrationOperation> offloadToHost(Allocation& alloc);
    std::optional<MigrationOperation> reloadToDevice(Allocation& alloc);
    void waitMigration(const MigrationOperation& op);

    // One-shot copy between two allocations on the same device via transfer
    // queue. Creates a transient command pool/buffer, submits, waits, destroys.
    bool copyBuffer(const Allocation& src, const Allocation& dst,
                    VkDeviceSize srcOffset, VkDeviceSize dstOffset,
                    VkDeviceSize size, VkFence fence = VK_NULL_HANDLE);

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
    
    // Generation tracking for handle validation.
    //
    // Generations are monotonic, but validity is tracked via a LIVE SET, not
    // by comparing against the current counter. Comparing against the counter
    // only ever matches the most-recently-created allocation, so freeing an
    // older allocation (out of order) was wrongly rejected and leaked memory.
    uint64_t nextGeneration() {
        uint64_t g = ++generationCounter_;
        liveGenerations_.insert(g);
        return g;
    }
    uint64_t getCurrentGeneration() const { return generationCounter_; }
    bool isValidGeneration(uint64_t generation) const {
        return liveGenerations_.count(generation) != 0;
    }
    void retireGeneration(uint64_t generation) {
        liveGenerations_.erase(generation);
    }
    
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
    
    // Mapping/wiring helpers
    VkMemoryPropertyFlags usageToFlags(MemoryUsage usage) const;
    bool wouldExceedBudget(VkDeviceSize additionalBytes) const;
    void setDebugName(VkObjectType objectType, uint64_t objectHandle,
                      const char* name) const;
    
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
    uint32_t deviceLocalHeapIndex_ = UINT32_MAX;
    bool memoryBudgetAvailable_ = false;
    VkCommandPool transferCmdPool_ = VK_NULL_HANDLE;
    bool debugUtilsEnabled_ = false;
    PFN_vkSetDebugUtilsObjectNameEXT fnSetDebugName_ = nullptr;
    
    // Offload manager for host swap
    std::unique_ptr<OffloadManager> offloadManager_;
    
    // Generation counter for handle validation (prevents stale handle use)
    uint64_t generationCounter_ = 0;
    // Live (not-yet-freed) generations. An allocation's generation is valid
    // iff it is present here; deallocate() retires it.
    std::unordered_set<uint64_t> liveGenerations_;
};

// ============================================================================
// GPU Instance (for MultiGPUPoolManager)
// ============================================================================

struct VVM_API GPUInstance {
    UnifiedMemoryPool pool;
    DeviceConfig config;
    uint32_t deviceIndex = 0;
    bool isMaster = false;  // allocates and exports
};

// ============================================================================
// Forward Declarations
// ============================================================================

// Deleter function for UniqueAllocation
inline void uniqueAllocationDeleter(UnifiedMemoryPool* pool, Allocation&& alloc) {
    if (pool && alloc.buffer != VK_NULL_HANDLE) {
        pool->deallocate(std::move(alloc));
    }
}

// RAII wrapper for Allocation - only create via UniqueAllocation::make()
class VVM_API UniqueAllocation {
public:
    using Deleter = void(*)(UnifiedMemoryPool*, Allocation&&);
    
private:
    UniqueAllocation() = default;
    UniqueAllocation(UnifiedMemoryPool* pool, Allocation&& alloc)
        : pool_(pool), alloc_(std::move(alloc)), deleter_(&uniqueAllocationDeleter) {
        assert(deleter_ != nullptr && "UniqueAllocation: deleter must not be null");
    }
    
public:
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
        } else if (pool_ && alloc_.buffer != VK_NULL_HANDLE && !deleter_) {
            VVM_LOG_ERROR("UniqueAllocation: null deleter with live allocation - possible misuse (use make() factory)");
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
    
    // Factory to create with proper deleter (only way to construct)
    static UniqueAllocation make(UnifiedMemoryPool* pool, Allocation&& alloc);
    
private:
    UnifiedMemoryPool* pool_ = nullptr;
    Allocation alloc_;
    Deleter deleter_ = nullptr;
};

} // namespace vvm