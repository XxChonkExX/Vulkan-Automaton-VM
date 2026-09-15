// ============================================================================
// VulkanMemoryBackend - the first IDeviceMemoryBackend implementation.
//
// Wraps the exact Vulkan semantics the Chonk Buffer pool has always used:
//   - allocate: VkMemoryAllocateInfo + optional pNext chain
//     (DEVICE_ADDRESS flags -> memory priority -> dedicated hint -> export)
//   - buffers:  VkBufferCreateInfo (+ VkExternalMemoryBufferCreateInfo when
//     exportable), bound via vkBindBufferMemory2 when created with memory
//   - budget:   vkGetPhysicalDeviceMemoryProperties2 + budget properties
//
// pNext chain order matches the pool's historical comment (RADV gfx1151
// crash): dedicated info first, then device-address flags, then priority.
// ============================================================================

#include "vulkan_vm/vulkan_mem_backend.hpp"
#include "vulkan_vm/hip_mem_backend.hpp"
#include "vulkan_vm/l0_mem_backend.hpp"

#include <cstring>

namespace vvm {

namespace {

// Backend-native error code for the Vulkan backend is the VkResult value,
// packed via the shared convention in mem_backend.hpp.
int asError(VkResult r) { return encode_backend_error(static_cast<int>(r)); }

} // namespace

VulkanMemoryBackend::VulkanMemoryBackend(const DeviceConfig& cfg) {
    device_ = cfg.device;
    physical_ = cfg.physicalDevice;
    if (physical_ != VK_NULL_HANDLE) {
        vkGetPhysicalDeviceMemoryProperties(physical_, &memProps_);
    }
    pfnGetBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
        vkGetDeviceProcAddr(device_, "vkGetBufferDeviceAddress"));

    // bufferDeviceAddress lives in Vulkan-1.2 features, not the core struct.
    if (physical_ != VK_NULL_HANDLE) {
        VkPhysicalDeviceVulkan12Features features12{};
        features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &features12;
        vkGetPhysicalDeviceFeatures2(physical_, &features2);
        deviceAddressFeature_ = features12.bufferDeviceAddress != VK_FALSE;
    }
}

const char* VulkanMemoryBackend::name() const { return "vulkan"; }

std::vector<BackendHeap> VulkanMemoryBackend::heaps() const {
    std::vector<BackendHeap> out;
    out.reserve(memProps_.memoryHeapCount);
    for (uint32_t i = 0; i < memProps_.memoryHeapCount; ++i) {
        out.push_back({memProps_.memoryHeaps[i].size,
                       static_cast<uint32_t>(memProps_.memoryHeaps[i].flags)});
    }
    return out;
}

std::vector<BackendMemType> VulkanMemoryBackend::memoryTypes() const {
    std::vector<BackendMemType> out;
    out.reserve(memProps_.memoryTypeCount);
    for (uint32_t i = 0; i < memProps_.memoryTypeCount; ++i) {
        out.push_back({memProps_.memoryTypes[i].heapIndex,
                       static_cast<uint32_t>(memProps_.memoryTypes[i].propertyFlags)});
    }
    return out;
}

BackendBudget VulkanMemoryBackend::heapBudget(uint32_t heapIndex) const {
    BackendBudget out{};
    VkPhysicalDeviceMemoryProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    props2.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(physical_, &props2);
    if (heapIndex < VK_MAX_MEMORY_HEAPS) {
        out.budgetBytes = budget.heapBudget[heapIndex];
        out.usedBytes   = budget.heapUsage[heapIndex];
        out.valid       = out.budgetBytes != 0;
    }
    return out;
}

bool VulkanMemoryBackend::type_is_host_visible(uint32_t typeIndex) const {
    if (typeIndex >= memProps_.memoryTypeCount) return false;
    return (memProps_.memoryTypes[typeIndex].propertyFlags &
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
}

BackendMemory VulkanMemoryBackend::allocate(const BackendAllocRequest& req,
                                            int* resultError) {
    if (resultError) *resultError = 0;

    // These MUST outlive the vkAllocateMemory call (see pool comment: a
    // dangling pNext crashed RADV on gfx1151).
    VkMemoryAllocateFlagsInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryPriorityAllocateInfoEXT priorityInfo{};
    priorityInfo.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT;

    VkMemoryDedicatedAllocateInfo dedicatedInfo{};
    dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedInfo.buffer = reinterpret_cast<VkBuffer>(req.dedicatedFor);

#ifdef VVM_PLATFORM_WINDOWS
    VkExportMemoryAllocateInfoKHR exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    VkExportMemoryAllocateInfoKHR exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = req.size;
    allocInfo.memoryTypeIndex = req.typeIndex;

    // Chain (tail -> head): priority, then device-address flags, then
    // dedicated hint, then export. Export must be the head-most because an
    // exported dedicated allocation carries both requirements.
    void* tail = nullptr;
    if (req.priority > 0.0f) {
        priorityInfo.priority = req.priority;
        priorityInfo.pNext = tail;
        tail = &priorityInfo;
    }
    void* noDedicatedHead = tail;
    if (req.deviceAddress) {
        flagsInfo.pNext = tail;
        noDedicatedHead = &flagsInfo;
    }
    if (req.dedicatedFor != 0) {
        dedicatedInfo.pNext = noDedicatedHead;
        allocInfo.pNext = &dedicatedInfo;
    } else {
        allocInfo.pNext = noDedicatedHead;
    }
    if (req.exportable) {
        exportInfo.pNext = allocInfo.pNext;
        allocInfo.pNext = &exportInfo;
    }

    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkResult result = vkAllocateMemory(device_, &allocInfo, nullptr, &memory);
    if (result != VK_SUCCESS) {
        if (resultError) *resultError = asError(result);
        return 0;
    }
    return reinterpret_cast<uint64_t>(memory);
}

void VulkanMemoryBackend::free(BackendMemory mem) {
    if (mem != 0) {
        vkFreeMemory(device_, reinterpret_cast<VkDeviceMemory>(mem), nullptr);
    }
}

void* VulkanMemoryBackend::map(BackendMemory mem, bool* ok) {
    if (ok) *ok = false;
    if (mem == 0) return nullptr;
    void* ptr = nullptr;
    VkResult result = vkMapMemory(device_, reinterpret_cast<VkDeviceMemory>(mem),
                                  0, VK_WHOLE_SIZE, 0, &ptr);
    if (result != VK_SUCCESS) {
        return nullptr;
    }
    if (ok) *ok = true;
    return ptr;
}

void VulkanMemoryBackend::unmap(BackendMemory mem) {
    if (mem != 0) {
        vkUnmapMemory(device_, reinterpret_cast<VkDeviceMemory>(mem));
    }
}

BackendBuffer VulkanMemoryBackend::create_buffer(BackendMemory mem, uint64_t offset,
                                                 uint64_t size, uint64_t usageBits,
                                                 bool exportable, int* resultError) {
    if (resultError) *resultError = 0;

    VkExternalMemoryBufferCreateInfo extInfo{};
    extInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
#ifdef VVM_PLATFORM_WINDOWS
    extInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    extInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = static_cast<VkBufferUsageFlags>(usageBits);
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (exportable) {
        info.pNext = &extInfo;
    }

    VkBuffer buffer = VK_NULL_HANDLE;
    VkResult result = vkCreateBuffer(device_, &info, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        if (resultError) *resultError = asError(result);
        return 0;
    }
    if (mem != 0) {
        VkBindBufferMemoryInfo bind{};
        bind.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
        bind.buffer = buffer;
        bind.memory = reinterpret_cast<VkDeviceMemory>(mem);
        bind.memoryOffset = offset;
        result = vkBindBufferMemory2(device_, 1, &bind);
        if (result != VK_SUCCESS) {
            if (resultError) *resultError = asError(result);
            vkDestroyBuffer(device_, buffer, nullptr);
            return 0;
        }
    }
    return reinterpret_cast<uint64_t>(buffer);
}

bool VulkanMemoryBackend::bind_buffer(BackendBuffer buf, BackendMemory mem) {
    if (buf == 0 || mem == 0) return false;
    VkBindBufferMemoryInfo bind{};
    bind.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
    bind.buffer = reinterpret_cast<VkBuffer>(buf);
    bind.memory = reinterpret_cast<VkDeviceMemory>(mem);
    bind.memoryOffset = 0;
    return vkBindBufferMemory2(device_, 1, &bind) == VK_SUCCESS;
}

void VulkanMemoryBackend::destroy_buffer(BackendBuffer buf) {
    if (buf != 0) {
        vkDestroyBuffer(device_, reinterpret_cast<VkBuffer>(buf), nullptr);
    }
}

uint64_t VulkanMemoryBackend::buffer_device_address(BackendBuffer buf) const {
    if (buf == 0 || !pfnGetBufferDeviceAddress) return 0;
    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = reinterpret_cast<VkBuffer>(buf);
    return pfnGetBufferDeviceAddress(device_, &info);
}

bool VulkanMemoryBackend::supports_export(ExternalHandleType type) const {
#ifdef VVM_PLATFORM_WINDOWS
    return type == ExternalHandleType::OpaqueWin32 ||
           type == ExternalHandleType::D3D12Heap ||
           type == ExternalHandleType::D3D12Resource;
#else
    return type == ExternalHandleType::OpaqueFd ||
           type == ExternalHandleType::DmaBuf;
#endif
}

std::unique_ptr<IDeviceMemoryBackend> create_memory_backend(MemBackendKind kind,
                                                            const DeviceConfig& cfg) {
    switch (kind) {
        case MemBackendKind::Vulkan:
            return std::make_unique<VulkanMemoryBackend>(cfg);
        case MemBackendKind::Hip:
            // Dynamic amdhip64 loader; nullptr when HIP is absent. The
            // DeviceConfig's Vulkan handles are irrelevant here - the
            // backendDeviceIndex selects the HIP device.
            return HipMemoryBackend::create(cfg.backendDeviceIndex);
        case MemBackendKind::Level0:
            // Dynamic ze_loader (ships with Intel GPU drivers); nullptr when
            // Level Zero is absent. backendDeviceIndex selects the device.
            return L0MemoryBackend::create(cfg.backendDeviceIndex);
    }
    return nullptr;
}

} // namespace vvm
