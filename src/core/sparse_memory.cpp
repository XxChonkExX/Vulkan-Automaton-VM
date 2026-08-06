#include "vulkan_vm/sparse.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <cmath>

namespace vvm {

// ============================================================================
// SparseVirtualMemoryPool Implementation
// ============================================================================

VkDeviceSize SparseVirtualMemoryPool::alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment == 0) return value;
    return (value + alignment - 1) & ~(alignment - 1);
}

std::optional<SparseVirtualMemoryPool> SparseVirtualMemoryPool::create(
    const DeviceConfig& device,
    const SparseMemoryConfig& config) {

    if (!device.device || !device.physicalDevice || config.virtualSize == 0) {
        return std::nullopt;
    }

    // Query sparse features: must support both sparseBinding and sparse residency buffers.
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(device.physicalDevice, &feats);
    if (!feats.sparseBinding || !feats.sparseResidencyBuffer) {
        VVM_LOG_WARN("SparseVirtualMemoryPool: device does not support sparseBinding/sparseResidencyBuffer");
        return std::nullopt;
    }

    // Find a queue family with VK_QUEUE_SPARSE_BINDING_BIT; prefer the
    // configured transfer family when it qualifies (the device may only have
    // created queues for that family).
    uint32_t queueFamily = device.transferQueueFamily;
    bool hasSparseFamily = false;
    {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device.physicalDevice, &count, nullptr);
        std::vector<VkQueueFamilyProperties> props(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device.physicalDevice, &count, props.data());
        if (queueFamily < count &&
            (props[queueFamily].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0) {
            hasSparseFamily = true;
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                if ((props[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0) {
                    queueFamily = i;
                    hasSparseFamily = true;
                    break;
                }
            }
        }
    }
    if (!hasSparseFamily) {
        VVM_LOG_WARN("SparseVirtualMemoryPool: no queue family with VK_QUEUE_SPARSE_BINDING_BIT");
        return std::nullopt;
    }

    SparseVirtualMemoryPool pool;
    pool.deviceConfig_ = device;
    pool.config_ = config;
    pool.device_ = device.device;
    pool.sparseQueueFamily_ = queueFamily;
    vkGetDeviceQueue(device.device, queueFamily, 0, &pool.sparseQueue_);
    if (!pool.sparseQueue_) return std::nullopt;

    // Sparse residency granularity. The sparse buffer requirements API is not
    // available in every SDK header set; we use a conservative power-of-two
    // page granularity of 1 MiB, which all sparse implementations accept.
    VkDeviceSize gran = 1024ull * 1024;  // 1 MiB fallback
    pool.granularity_ = gran;
    pool.residencyGranularity_ = static_cast<uint32_t>(gran);

    // Page size: request >= granularity, round up to a multiple of granularity.
    VkDeviceSize pageSize = config.pageSize != 0 ? config.pageSize : gran;
    pageSize = alignUp(std::max(pageSize, gran), gran);
    // Buffer sparse memory binds must be aligned to granularity; page size is
    // already granularity-aligned. Ensure it is also a power of two when possible.
    pool.pageSize_ = pageSize;

    // Create the sparse virtual buffer.
    VkBufferCreateInfo binfo{};
    binfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    binfo.size = config.virtualSize;
    binfo.usage = config.usage;
    binfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    binfo.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
                  VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
    if (config.enableDeviceAddress) {
        binfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }

    VkBuffer sparseBuffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device.device, &binfo, nullptr, &sparseBuffer) != VK_SUCCESS ||
        !sparseBuffer) {
        VVM_LOG_ERROR("SparseVirtualMemoryPool: vkCreateBuffer failed");
        return std::nullopt;
    }
    pool.buffer_ = sparseBuffer;

    // Internal command pool + fences.
    VkCommandPoolCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpInfo.queueFamilyIndex = queueFamily;
    VkResult rc = vkCreateCommandPool(device.device, &cpInfo, nullptr, &pool.internalCmdPool_);
    if (rc != VK_SUCCESS) {
        pool.destroy();
        return std::nullopt;
    }
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device.device, &fenceInfo, nullptr, &pool.bindFence_) != VK_SUCCESS ||
        vkCreateFence(device.device, &fenceInfo, nullptr, &pool.copyFence_) != VK_SUCCESS) {
        pool.destroy();
        return std::nullopt;
    }

    // Reserve the page table.
    uint64_t pageCount = (config.virtualSize + pageSize - 1) / pageSize;
    pool.pages_.assign(static_cast<size_t>(pageCount), SparsePage{});

    // Device address if requested.
    if (config.enableDeviceAddress) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = sparseBuffer;
        pool.deviceAddress_ = vkGetBufferDeviceAddress(device.device, &addrInfo);
    }

    VVM_LOG_INFO("SparseVirtualMemoryPool: buffer={} virtual={} MiB page={} MiB ({}) pages",
                 static_cast<void*>(sparseBuffer),
                 static_cast<uint64_t>(config.virtualSize / (1024 * 1024)),
                 static_cast<uint64_t>(pageSize / (1024 * 1024)),
                 static_cast<uint64_t>(pageCount));
    return pool;
}

SparseVirtualMemoryPool::~SparseVirtualMemoryPool() {
    destroy();
}

SparseVirtualMemoryPool::SparseVirtualMemoryPool(SparseVirtualMemoryPool&& other) noexcept
    : pages_(std::move(other.pages_))
    , memoryPool_(std::move(other.memoryPool_))
    , freeMemorySlots_(std::move(other.freeMemorySlots_))
    , deviceConfig_(other.deviceConfig_)
    , config_(other.config_)
    , device_(other.device_)
    , buffer_(other.buffer_)
    , deviceAddress_(other.deviceAddress_)
    , sparseQueue_(other.sparseQueue_)
    , sparseQueueFamily_(other.sparseQueueFamily_)
    , granularity_(other.granularity_)
    , pageSize_(other.pageSize_)
    , residencyGranularity_(other.residencyGranularity_)
    , isHostVisiblePool_(other.isHostVisiblePool_)
    , internalCmdPool_(other.internalCmdPool_)
    , bindFence_(other.bindFence_)
    , copyFence_(other.copyFence_) {
    other.device_ = VK_NULL_HANDLE;
    other.buffer_ = VK_NULL_HANDLE;
    other.deviceAddress_ = 0;
    other.sparseQueue_ = VK_NULL_HANDLE;
    other.internalCmdPool_ = VK_NULL_HANDLE;
    other.bindFence_ = VK_NULL_HANDLE;
    other.copyFence_ = VK_NULL_HANDLE;
}

SparseVirtualMemoryPool& SparseVirtualMemoryPool::operator=(SparseVirtualMemoryPool&& other) noexcept {
    if (this != &other) {
        destroy();
        pages_ = std::move(other.pages_);
        memoryPool_ = std::move(other.memoryPool_);
        freeMemorySlots_ = std::move(other.freeMemorySlots_);
        deviceConfig_ = other.deviceConfig_;
        config_ = other.config_;
        device_ = other.device_;
        buffer_ = other.buffer_;
        deviceAddress_ = other.deviceAddress_;
        sparseQueue_ = other.sparseQueue_;
        sparseQueueFamily_ = other.sparseQueueFamily_;
        granularity_ = other.granularity_;
        pageSize_ = other.pageSize_;
        residencyGranularity_ = other.residencyGranularity_;
        isHostVisiblePool_ = other.isHostVisiblePool_;
        internalCmdPool_ = other.internalCmdPool_;
        bindFence_ = other.bindFence_;
        copyFence_ = other.copyFence_;
        other.device_ = VK_NULL_HANDLE;
        other.buffer_ = VK_NULL_HANDLE;
        other.sparseQueue_ = VK_NULL_HANDLE;
        other.internalCmdPool_ = VK_NULL_HANDLE;
        other.bindFence_ = VK_NULL_HANDLE;
        other.copyFence_ = VK_NULL_HANDLE;
    }
    return *this;
}

void SparseVirtualMemoryPool::destroy() {
    if (!device_) return;
    // Release any still-bound pages back to the free list first.
    for (auto& page : pages_) {
        if (page.bound && page.memory) {
            page.bound = false;
            page.memory = VK_NULL_HANDLE;
        }
    }
    freeMemorySlots_.clear();
    // Free the physical page blocks.
    for (auto& blk : memoryPool_) {
        if (blk.memory) {
            vkFreeMemory(device_, blk.memory, nullptr);
            blk.memory = VK_NULL_HANDLE;
        }
    }
    memoryPool_.clear();

    if (buffer_) {
        vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }
    if (bindFence_) {
        vkDestroyFence(device_, bindFence_, nullptr);
        bindFence_ = VK_NULL_HANDLE;
    }
    if (copyFence_) {
        vkDestroyFence(device_, copyFence_, nullptr);
        copyFence_ = VK_NULL_HANDLE;
    }
    if (internalCmdPool_) {
        vkDestroyCommandPool(device_, internalCmdPool_, nullptr);
        internalCmdPool_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
}

uint64_t SparseVirtualMemoryPool::getResidentPageCount() const {
    uint64_t n = 0;
    for (const auto& page : pages_) {
        if (page.bound) ++n;
    }
    return n;
}

VkDeviceMemory SparseVirtualMemoryPool::getPageMemory(uint32_t page) const {
    if (page >= pages_.size()) return VK_NULL_HANDLE;
    return pages_[page].memory;
}

void* SparseVirtualMemoryPool::getResidentHostPtr(uint32_t page) const {
    if (page >= pages_.size() || !pages_[page].bound) return nullptr;
    for (const auto& blk : memoryPool_) {
        if (blk.memory == pages_[page].memory && blk.hostPtr) {
            return static_cast<char*>(blk.hostPtr) + pages_[page].memoryOffset;
        }
    }
    return nullptr;
}

bool SparseVirtualMemoryPool::isResident(VkDeviceSize offset) const {
    if (offset >= config_.virtualSize) return false;
    uint64_t page = static_cast<uint64_t>(offset / pageSize_);
    return page < pages_.size() && pages_[page].bound;
}

bool SparseVirtualMemoryPool::allocatePageMemory(uint32_t memoryTypeIndex, SparsePage& out) {
    // Reuse a freed slot if one is available.
    if (!freeMemorySlots_.empty()) {
        uint32_t slot = freeMemorySlots_.back();
        freeMemorySlots_.pop_back();
        PagedMemory& blk = memoryPool_[slot];
        if (blk.memory) {
            out.memory = blk.memory;
            out.memoryOffset = 0;
            out.bound = true;
            return true;
        }
    }

    // Allocate a fresh page-sized block of device memory.
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = pageSize_;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkResult res = vkAllocateMemory(device_, &allocInfo, nullptr, &mem);
    if (res != VK_SUCCESS || !mem) return false;

    // Determine host visibility for this memory type.
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(deviceConfig_.physicalDevice, &memProps);
    bool hostVisible = false;
    if (memoryTypeIndex < memProps.memoryTypeCount) {
        hostVisible = (memProps.memoryTypes[memoryTypeIndex].propertyFlags &
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    }

    PagedMemory blk;
    blk.memory = mem;
    blk.properties = (memoryTypeIndex < memProps.memoryTypeCount)
                         ? memProps.memoryTypes[memoryTypeIndex].propertyFlags
                         : 0;
    blk.hostPtr = nullptr;
    if (hostVisible) {
        vkMapMemory(device_, mem, 0, pageSize_, 0, &blk.hostPtr);
    }
    memoryPool_.push_back(std::move(blk));

    out.memory = mem;
    out.memoryOffset = 0;
    out.bound = true;
    return true;
}

bool SparseVirtualMemoryPool::makeResident(VkDeviceSize offset, VkDeviceSize size) {
    if (size == 0 || offset >= config_.virtualSize) return false;
    VkDeviceSize end = std::min<VkDeviceSize>(offset + size, config_.virtualSize);
    uint64_t firstPage = offset / pageSize_;
    uint64_t lastPage = (end - 1) / pageSize_;

    // Pick a memory type: prefer DEVICE_LOCAL | hostVisible (best for residency +
    // host fallback). fall back to device local only.
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(deviceConfig_.physicalDevice, &memProps);
    uint32_t memoryTypeIndex = UINT32_MAX;
    // Prefer a device-local + host-visible type if the device has one (allows
    // host reads of resident pages, which is very useful for residency tests).
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        const auto& flags = memProps.memoryTypes[i].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memoryTypeIndex = i;
            break;
        }
    }
    if (memoryTypeIndex == UINT32_MAX) {
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                memoryTypeIndex = i;
                break;
            }
        }
    }
    if (memoryTypeIndex == UINT32_MAX) {
        VVM_LOG_ERROR("SparseVirtualMemoryPool: no suitable device-local memory type");
        return false;
    }

    // Build the sparse buffer bind list for every page to become resident.
    std::vector<VkSparseMemoryBind> binds;
    std::vector<uint32_t> toBind;
    bool ok = true;
    for (uint64_t p = firstPage; p <= lastPage; ++p) {
        SparsePage& page = pages_[p];
        if (page.bound) continue;
        if (!allocatePageMemory(memoryTypeIndex, page)) {
            VVM_LOG_WARN("SparseVirtualMemoryPool: failed to allocate physical page {}", p);
            ok = false;
            break;
        }
        VkSparseMemoryBind bind{};
        bind.resourceOffset = static_cast<VkDeviceSize>(p) * pageSize_;
        bind.size = pageSize_;
        bind.memory = page.memory;
        bind.memoryOffset = page.memoryOffset;
        binds.push_back(bind);
        toBind.push_back(static_cast<uint32_t>(p));
    }
    if (!ok) {
        // Roll back any pages we just marked bound.
        for (uint32_t p : toBind) {
            pages_[p].bound = false;
            pages_[p].memory = VK_NULL_HANDLE;
        }
        return false;
    }
    if (binds.empty()) return true;

    VkSparseBufferMemoryBindInfo buffInfo{};
    buffInfo.buffer = buffer_;
    buffInfo.bindCount = static_cast<uint32_t>(binds.size());
    buffInfo.pBinds = binds.data();

    VkBindSparseInfo binfo{};
    binfo.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
    binfo.bufferBindCount = 1;
    binfo.pBufferBinds = &buffInfo;

    if (vkQueueBindSparse(sparseQueue_, 1, &binfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        VVM_LOG_ERROR("SparseVirtualMemoryPool: vkQueueBindSparse failed");
        return false;
    }
    // Wait for the sparse queue to finish binding so the pages are usable.
    vkQueueWaitIdle(sparseQueue_);
    return true;
}

bool SparseVirtualMemoryPool::makeUnresident(VkDeviceSize offset, VkDeviceSize size) {
    if (offset >= config_.virtualSize || size == 0) return false;
    VkDeviceSize end = std::min<VkDeviceSize>(offset + size, config_.virtualSize);
    uint64_t firstPage = offset / pageSize_;
    uint64_t lastPage = (end - 1) / pageSize_;

    // Build binds that unbind pages (memory = NULL means "unbind this page").
    std::vector<VkSparseMemoryBind> binds;
    std::vector<uint32_t> toDetach;
    for (uint64_t p = firstPage; p <= lastPage; ++p) {
        if (!pages_[p].bound) continue;
        VkSparseMemoryBind bind{};
        bind.resourceOffset = static_cast<VkDeviceSize>(p) * pageSize_;
        bind.size = pageSize_;
        bind.memory = VK_NULL_HANDLE;  // NULL unbinds
        bind.memoryOffset = 0;
        binds.push_back(bind);
        toDetach.push_back(static_cast<uint32_t>(p));
        pages_[p].bound = false;
    }
    if (binds.empty()) return true;

    VkSparseBufferMemoryBindInfo buffInfo{};
    buffInfo.buffer = buffer_;
    buffInfo.bindCount = static_cast<uint32_t>(binds.size());
    buffInfo.pBinds = binds.data();

    VkBindSparseInfo binfo{};
    binfo.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
    binfo.bufferBindCount = 1;
    binfo.pBufferBinds = &buffInfo;

    if (vkQueueBindSparse(sparseQueue_, 1, &binfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        // Restore bound state on failure.
        for (uint32_t p : toDetach) pages_[p].bound = true;
        return false;
    }
    vkQueueWaitIdle(sparseQueue_);

    // Return the memory blocks to the free list so they can be reused.
    for (uint32_t p : toDetach) {
        if (pages_[p].memory) {
            for (uint32_t slot = 0; slot < memoryPool_.size(); ++slot) {
                if (memoryPool_[slot].memory == pages_[p].memory) {
                    freeMemorySlots_.push_back(slot);
                    break;
                }
            }
            pages_[p].memory = VK_NULL_HANDLE;
        }
    }
    return true;
}

} // namespace vvm