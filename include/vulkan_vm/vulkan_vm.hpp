#pragma once

#ifdef VVM_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>
#include <optional>
#include <string>
#include <span>
#include <memory>

#include "vulkan_vm/buddy_allocator.hpp"

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
};

struct BlockInfo {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize used = 0;
    VkDeviceSize offset = 0;  // for host-visible shadow
    void* hostPtr = nullptr;
    int exportFd = -1;        // Linux external handle
#ifdef VVM_PLATFORM_WINDOWS
    HANDLE exportHandle = nullptr;  // Windows external handle
#endif
    // Buddy allocator for this block (replaces freeRanges)
    std::unique_ptr<class BuddyAllocator> buddy;
    VkMemoryPropertyFlags memoryFlags = 0;
    bool isHostVisible = false;
    bool isCoherent = false;
};

// ============================================================================
// Cross-GPU Sharing
// ============================================================================

enum class ExternalHandleType {
    OpaqueFd,      // Linux: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    OpaqueWin32,   // Windows: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
    D3D12Heap,     // Windows: VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT
    DmaBuf         // Linux: VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
};

struct ExternalMemoryInfo {
    ExternalHandleType type = ExternalHandleType::OpaqueFd;
    int fd = -1;           // Linux
#ifdef VVM_PLATFORM_WINDOWS
    HANDLE handle = nullptr;  // Windows
#endif
    VkDeviceSize size = 0;
    uint32_t memoryTypeIndex = UINT32_MAX;
    bool dedicatedAllocation = false;
};

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
    bool useMadvise = true;                                 // MADV_DONTNEED/FREE
    bool useMprotect = true;                                // PROT_NONE on offload
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
    
    std::optional<Allocation> allocateTensor(VkDeviceSize size,
                                             VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    
    void deallocate(Allocation&& alloc);

    // Cross-GPU: Export/Import
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
    void defragment();  // coalesce free ranges, migrate if needed
    void trim();        // release empty blocks back to OS (optional)

private:
    friend class MultiGPUPoolManager;
    friend struct GPUInstance;
    UnifiedMemoryPool() = default;
    bool initialize(const DeviceConfig& device, const PoolConfig& config);
    
    std::optional<uint32_t> findMemoryType(VkMemoryPropertyFlags required,
                                           VkMemoryPropertyFlags preferred);
    std::optional<VkDeviceMemory> allocateBlock(VkDeviceSize size,
                                                 uint32_t memoryTypeIndex,
                                                 bool exportable);
    std::optional<Allocation> subAllocate(VkDeviceSize size,
                                          VkDeviceSize alignment,
                                          uint32_t blockIndex);
    void subDeallocate(Allocation&& alloc);
    
    // Buddy allocator helpers
    VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment);
    
    DeviceConfig deviceConfig_;
    PoolConfig config_;
    VkDevice device_ = VK_NULL_HANDLE;
    std::vector<BlockInfo> blocks_;
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

#include "vulkan_vm/offload.hpp"
#include "vulkan_vm/network.hpp"