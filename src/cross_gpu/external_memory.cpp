#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"
#include "external_memory.hpp"

#include <algorithm>

namespace vvm {

// ============================================================================
// External Memory Capabilities Detection
// ============================================================================

ExternalMemoryCaps queryExternalMemoryCaps(VkPhysicalDevice physicalDevice) {
    ExternalMemoryCaps caps;
    
    // Check each handle type
    const struct {
        VkExternalMemoryHandleTypeFlagBits handleType;
        const char* name;
    } handleTypes[] = {
        {VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, "OPAQUE_FD"},
        {VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT, "OPAQUE_WIN32"},
        {VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT, "OPAQUE_WIN32_KMT"},
        {VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT, "D3D12_HEAP"},
        {VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT, "D3D12_RESOURCE"},
        {VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, "DMA_BUF"},
#ifdef VK_USE_PLATFORM_FUCHSIA
        {VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_HANDLE_BIT_FUCHSIA, "ZIRCON"},
#endif
        {VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, "HOST_ALLOCATION"},
        {VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT, "HOST_MAPPED_FOREIGN"},
    };
    
    for (const auto& ht : handleTypes) {
        VkPhysicalDeviceExternalBufferInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
        info.handleType = ht.handleType;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        
        VkExternalBufferProperties props{};
        props.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
        
        vkGetPhysicalDeviceExternalBufferProperties(physicalDevice, &info, &props);
        const auto& memProps = props.externalMemoryProperties;
        
        if (memProps.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) {
            caps.supportedHandleTypes |= ht.handleType;
            caps.supportedFeatures |= memProps.externalMemoryFeatures;
            
            switch (ht.handleType) {
                case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT:
                    caps.supportsOpaqueFd = true; break;
                case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT:
                case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT:
                    caps.supportsOpaqueWin32 = true; break;
                case VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT:
                    caps.supportsD3D12Heap = true; break;
                case VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:
                    caps.supportsDmaBuf = true; break;
#ifdef VK_USE_PLATFORM_FUCHSIA
                case VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_HANDLE_BIT_FUCHSIA:
                    caps.supportsZirconHandle = true; break;
#endif
            }
        }
    }
    
    return caps;
}

// ============================================================================
// Platform-Optimal Handle Type Selection
// ============================================================================

ExternalHandleType selectOptimalHandleType(const ExternalMemoryCaps& caps,
                                            const VkPhysicalDeviceProperties& vendor) {
    // NVIDIA vendor ID: 0x10DE
    // AMD vendor ID: 0x1002
    // Intel vendor ID: 0x8086
    const bool isNvidia = (vendor.vendorID == 0x10DE);
    const bool isAmd = (vendor.vendorID == 0x1002);
    const bool isIntel = (vendor.vendorID == 0x8086);
    
    #ifdef VVM_PLATFORM_LINUX
    // Linux: Prefer DMA-BUF for cross-vendor, OPAQUE_FD for same-vendor
    if (caps.supportsDmaBuf && (isNvidia || isAmd || isIntel)) {
        return ExternalHandleType::DmaBuf;  // Best for cross-vendor (NVIDIA<->AMD<->Intel)
    }
    if (caps.supportsOpaqueFd) {
        return ExternalHandleType::OpaqueFd;  // Good for same-vendor
    }
    #elif defined(VVM_PLATFORM_WINDOWS)
    // Windows: NVIDIA prefers D3D12_HEAP, AMD/Intel use OPAQUE_WIN32
    if (isNvidia && caps.supportsD3D12Heap) {
        return ExternalHandleType::D3D12Heap;  // NVIDIA optimal path
    }
    if (caps.supportsOpaqueWin32) {
        return ExternalHandleType::OpaqueWin32;  // AMD/Intel standard
    }
    #endif
    
    // Fallback
    #ifdef VVM_PLATFORM_LINUX
    if (caps.supportsOpaqueFd) return ExternalHandleType::OpaqueFd;
    #elif defined(VVM_PLATFORM_WINDOWS)
    if (caps.supportsOpaqueWin32) return ExternalHandleType::OpaqueWin32;
    #endif
    
    return ExternalHandleType::OpaqueFd;  // Default
}

VkPhysicalDeviceProperties getVendorProperties(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceProperties vendor{};
    vkGetPhysicalDeviceProperties(physicalDevice, &vendor);
    return vendor;
}

// ============================================================================
// Export Memory (Source GPU)
// ============================================================================

std::optional<ExternalMemoryInfo> exportMemory(VkDevice device,
                                                VkDeviceMemory memory,
                                                ExternalHandleType type,
                                                VkDeviceSize size,
                                                uint32_t memoryTypeIndex) {
    ExternalMemoryInfo info;
    info.type = type;
    info.size = size;
    info.memoryTypeIndex = memoryTypeIndex;
    
    #ifdef VVM_PLATFORM_LINUX
    if (type == ExternalHandleType::OpaqueFd || type == ExternalHandleType::DmaBuf) {
        VkMemoryGetFdInfoKHR fdInfo{};
        fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fdInfo.memory = memory;
        
        switch (type) {
            case ExternalHandleType::OpaqueFd:
                fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
                break;
            case ExternalHandleType::DmaBuf:
                fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
                break;
            default:
                return std::nullopt;
        }
        
        PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = 
            (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
        if (!vkGetMemoryFdKHR) return std::nullopt;
        
        int fd = -1;
        if (vkGetMemoryFdKHR(device, &fdInfo, &fd) != VK_SUCCESS) return std::nullopt;
        info.fd = fd;
        return info;
    }
    #elif defined(VVM_PLATFORM_WINDOWS)
    if (type == ExternalHandleType::OpaqueWin32 || type == ExternalHandleType::D3D12Heap) {
        VkMemoryGetWin32HandleInfoKHR handleInfo{};
        handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handleInfo.memory = memory;
        
        switch (type) {
            case ExternalHandleType::OpaqueWin32:
                handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
                break;
            case ExternalHandleType::D3D12Heap:
                handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
                break;
            default:
                return std::nullopt;
        }
        
        PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = 
            (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
        if (!vkGetMemoryWin32HandleKHR) return std::nullopt;
        
        HANDLE handle = nullptr;
        if (vkGetMemoryWin32HandleKHR(device, &handleInfo, &handle) != VK_SUCCESS) return std::nullopt;
        info.handle = handle;
        return info;
    }
    #endif
    
    return std::nullopt;
}

// ============================================================================
// Import Memory (Destination GPU)
// ============================================================================

struct ImportMemoryParams {
    VkDevice device;
    ExternalMemoryInfo info;
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags requiredFlags;
    VkMemoryPropertyFlags preferredFlags;
};

std::optional<Allocation> importMemory(const ImportMemoryParams& params) {
    VkImportMemoryFdInfoKHR importFdInfo{};
    VkImportMemoryWin32HandleInfoKHR importWin32Info{};
    VkMemoryDedicatedAllocateInfo dedicatedInfo{};
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = params.info.size;
    allocInfo.memoryTypeIndex = params.info.memoryTypeIndex;
    
    // Build pNext chain for import
    void* pNext = nullptr;
    
    #ifdef VVM_PLATFORM_LINUX
    if (params.info.type == ExternalHandleType::OpaqueFd && params.info.fd >= 0) {
        importFdInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        importFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        importFdInfo.fd = params.info.fd;
        importFdInfo.pNext = pNext;
        pNext = &importFdInfo;
    } else if (params.info.type == ExternalHandleType::DmaBuf && params.info.fd >= 0) {
        importFdInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        importFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        importFdInfo.fd = params.info.fd;
        importFdInfo.pNext = pNext;
        pNext = &importFdInfo;
    }
    #elif defined(VVM_PLATFORM_WINDOWS)
    if (params.info.type == ExternalHandleType::OpaqueWin32 && params.info.handle) {
        importWin32Info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        importWin32Info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        importWin32Info.handle = params.info.handle;
        importWin32Info.pNext = pNext;
        pNext = &importWin32Info;
    } else if (params.info.type == ExternalHandleType::D3D12Heap && params.info.handle) {
        importWin32Info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        importWin32Info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
        importWin32Info.handle = params.info.handle;
        importWin32Info.pNext = pNext;
        pNext = &importWin32Info;
    }
    #endif
    
    // Device address support
    VkMemoryAllocateFlagsInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    flagsInfo.pNext = pNext;
    pNext = &flagsInfo;
    
    allocInfo.pNext = pNext;
    
    VkDeviceMemory memory;
    if (vkAllocateMemory(params.device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        return std::nullopt;
    }
    
    // Create buffer
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = params.info.size;
    bufferInfo.usage = params.usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkBuffer buffer;
    if (vkCreateBuffer(params.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        vkFreeMemory(params.device, memory, nullptr);
        return std::nullopt;
    }
    
    if (vkBindBufferMemory(params.device, buffer, memory, 0) != VK_SUCCESS) {
        vkDestroyBuffer(params.device, buffer, nullptr);
        vkFreeMemory(params.device, memory, nullptr);
        return std::nullopt;
    }
    
    Allocation alloc;
    alloc.buffer = buffer;
    alloc.memory = memory;
    alloc.offset = 0;
    alloc.size = params.info.size;
    alloc.isExternal = true;
    
    // Get device address for bindless
    VkBufferDeviceAddressInfo addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addrInfo.buffer = buffer;
    alloc.deviceAddress = vkGetBufferDeviceAddress(params.device, &addrInfo);
    
    // Map if host visible
    VkMemoryPropertyFlags memFlags;
    // Would need to query memory type properties here
    // For now, assume not host visible unless specified
    
    return alloc;
}

// ============================================================================
// Cross-Vendor Compatibility Matrix
// ============================================================================

VendorPairCaps getCrossVendorCaps(VkPhysicalDevice src, VkPhysicalDevice dst) {
    VendorPairCaps caps;
    
    auto srcVendor = getVendorProperties(src);
    auto dstVendor = getVendorProperties(dst);
    
    bool srcNvidia = (srcVendor.vendorID == 0x10DE);
    bool srcAmd = (srcVendor.vendorID == 0x1002);
    bool srcIntel = (srcVendor.vendorID == 0x8086);
    bool dstNvidia = (dstVendor.vendorID == 0x10DE);
    bool dstAmd = (dstVendor.vendorID == 0x1002);
    bool dstIntel = (dstVendor.vendorID == 0x8086);
    
    auto srcCaps = queryExternalMemoryCaps(src);
    auto dstCaps = queryExternalMemoryCaps(dst);
    
    #ifdef VVM_PLATFORM_LINUX
    // Linux: DMA-BUF is universal for cross-vendor
    if (srcCaps.supportsDmaBuf && dstCaps.supportsDmaBuf) {
        caps.recommendedType = ExternalHandleType::DmaBuf;
        caps.nvidiaToAmd = srcNvidia && dstAmd;
        caps.nvidiaToIntel = srcNvidia && dstIntel;
        caps.amdToIntel = srcAmd && dstIntel;
        caps.notes = "DMA-BUF: Best cross-vendor compatibility on Linux";
    } else if (srcCaps.supportsOpaqueFd && dstCaps.supportsOpaqueFd) {
        caps.recommendedType = ExternalHandleType::OpaqueFd;
        caps.notes = "OPAQUE_FD: Same-vendor or limited cross-vendor";
    }
    #elif defined(VVM_PLATFORM_WINDOWS)
    // Windows: NVIDIA uses D3D12_HEAP, others use OPAQUE_WIN32
    // Cross-vendor needs D3D12_HEAP on NVIDIA side + OPAQUE_WIN32 on other
    if (srcNvidia && srcCaps.supportsD3D12Heap && (dstAmd || dstIntel) && dstCaps.supportsOpaqueWin32) {
        caps.recommendedType = ExternalHandleType::D3D12Heap;  // Export as D3D12_HEAP from NVIDIA
        caps.nvidiaToAmd = true;
        caps.nvidiaToIntel = true;
        caps.notes = "NVIDIA(D3D12_HEAP) -> AMD/Intel(OPAQUE_WIN32)";
    } else if ((srcAmd || srcIntel) && srcCaps.supportsOpaqueWin32 && dstNvidia && dstCaps.supportsD3D12Heap) {
        caps.recommendedType = ExternalHandleType::OpaqueWin32;  // Export as OPAQUE_WIN32 from AMD/Intel
        caps.nvidiaToAmd = true;
        caps.nvidiaToIntel = true;
        caps.notes = "AMD/Intel(OPAQUE_WIN32) -> NVIDIA(D3D12_HEAP)";
    } else if (srcCaps.supportsOpaqueWin32 && dstCaps.supportsOpaqueWin32) {
        caps.recommendedType = ExternalHandleType::OpaqueWin32;
        caps.notes = "OPAQUE_WIN32: Same vendor or NVIDIA<->NVIDIA";
    }
    #endif
    
    return caps;
}

} // namespace vvm