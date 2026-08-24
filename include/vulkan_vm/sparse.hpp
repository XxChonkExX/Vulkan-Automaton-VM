#pragma once

#include "vulkan_vm/core.hpp"

#include <vector>
#include <optional>
#include <cstdint>

namespace vvm {

// ============================================================================
// Sparse / Residency Support for Virtual Memory
// ============================================================================
//
// Creates a buffer with VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
// VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT: a virtual address space that is much
// larger than its physical backing, with memory committed on demand.
//
// - The virtual buffer reserves a large address range (e.g. 16 GiB) but
//   allocates physical VkDeviceMemory pages lazily.
// - Pages are committed (makeResident) and released (makeUnresident)
//   individually, so a workload only ramps up VRAM for pages that are actually
//   touched — the rest read as zero.
// - Sparse residency requires VK_QUEUE_SPARSE_BINDING_BIT on a queue and the
//   sparseBinding + sparseResidencyBuffer device features.
//
// Thread safety: like UnifiedMemoryPool, this type is NOT thread-safe. Use an
// external mutex if residency is modified from multiple threads.

struct SparseMemoryConfig {
    // Total virtual size of the sparse buffer (bytes). The address range is
    // reserved up front; physical pages are only allocated as they are bound.
    VkDeviceSize virtualSize = 0;

    // Suggested page size. The implementation rounds this up to the device's
    // sparse residency granularity (a power of two >= 1 MiB). If zero, the
    // device granularity is used.
    VkDeviceSize pageSize = 0;

    // Usage flags for the virtual buffer.
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // Enable device address + capacity GLSL access.
    bool enableDeviceAddress = true;
};

// One virtual page: its physical backing memory and residency state.
struct SparsePage {
    VkDeviceMemory memory = VK_NULL_HANDLE;   // physical backing (when bound)
    VkDeviceSize memoryOffset = 0;            // offset within the physical page
    bool bound = false;
};

class SparseVirtualMemoryPool {
public:
    static std::optional<SparseVirtualMemoryPool> create(
        const DeviceConfig& device,
        const SparseMemoryConfig& config);

    // Non-copyable, movable (move = detach Vulkan objects).
    SparseVirtualMemoryPool(const SparseVirtualMemoryPool&) = delete;
    SparseVirtualMemoryPool& operator=(const SparseVirtualMemoryPool&) = delete;
    SparseVirtualMemoryPool(SparseVirtualMemoryPool&&) noexcept;
    SparseVirtualMemoryPool& operator=(SparseVirtualMemoryPool&&) noexcept;
    ~SparseVirtualMemoryPool();

    // ========================================================================
    // Residency (commit / release virtual pages)
    // ========================================================================

    // Bind physical pages for [offset, offset + size) inside the virtual
    // buffer. The range is rounded up to whole pages. Returns false if the
    // device ran out of memory or the range was invalid.
    bool makeResident(VkDeviceSize offset, VkDeviceSize size);

    // Unbind physical pages for [offset, offset+size). The pages' memory is
    // returned to the pool (not the OS) so it can be reused by makeResident.
    bool makeUnresident(VkDeviceSize offset, VkDeviceSize size);

    // Returns true if the page containing 'offset' currently has its
    // physical backing memory bound.
    bool isResident(VkDeviceSize offset) const;

    // ========================================================================
    // Stats
    // ========================================================================

    VkDeviceSize getVirtualSize() const { return config_.virtualSize; }
    VkDeviceSize getPageSize() const { return pageSize_; }
    uint64_t getPageCount() const { return pages_.size(); }
    uint64_t getResidentPageCount() const;     // pages with physical memory bound
    VkDeviceSize getResidentBytes() const { return getResidentPageCount() * pageSize_; }
    uint32_t getResidencyGranularity() const { return residencyGranularity_; }

    // ========================================================================
    // Access
    // ========================================================================

    VkBuffer getBuffer() const { return buffer_; }
    VkDeviceAddress getDeviceAddress() const { return deviceAddress_; }
    VkDeviceMemory getPageMemory(uint32_t page) const;  // bound page memory
    void* getResidentHostPtr(uint32_t page) const;      // mapped ptr if page host-visible & bound

    // Device + queue access for external synchronization (copy/compute via the
    // sparse-capable queue we were created with).
    VkDevice getDevice() const { return device_; }
    VkQueue getSparseQueue() const { return sparseQueue_; }

private:
    SparseVirtualMemoryPool() = default;
    bool initialize(const DeviceConfig& device, const SparseMemoryConfig& config);

    // Page table: index = page number. Size = ceil(virtualSize / pageSize).
    std::vector<SparsePage> pages_;
    // Physical page cache: free list to reuse page-sized VkDeviceMemory.
    struct PagedMemory {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkMemoryPropertyFlags properties = 0;
        void* hostPtr = nullptr;  // set only if the memory type is HOST_VISIBLE
    };
    std::vector<PagedMemory> memoryPool_;   // owned memory blocks (page-sized)
    std::vector<uint32_t> freeMemorySlots_; // indices into memoryPool_ available for reuse

    // Device/config state
    DeviceConfig deviceConfig_;
    SparseMemoryConfig config_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceAddress deviceAddress_ = 0;
    VkQueue sparseQueue_ = VK_NULL_HANDLE;
    uint32_t sparseQueueFamily_ = UINT32_MAX;
    VkDeviceSize granularity_ = 0;
    VkDeviceSize pageSize_ = 0;
    uint32_t residencyGranularity_ = 0;
    bool isHostVisiblePool_ = false;

    // Command pool + fence used for vkQueueBindSparse + any internal copy.
    VkCommandPool internalCmdPool_ = VK_NULL_HANDLE;
    VkFence bindFence_ = VK_NULL_HANDLE;
    VkFence copyFence_ = VK_NULL_HANDLE;

    bool allocatePageMemory(uint32_t memoryTypeIndex, SparsePage& out);
    static VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment);
    void destroy();
};

} // namespace vvm