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

VkPhysicalDeviceProperties getVendorProperties(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceProperties vendor{};
    vkGetPhysicalDeviceProperties(physicalDevice, &vendor);
    return vendor;
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
    // Windows: For cross-vendor compatibility, both sides MUST use the SAME handle type.
    // NVIDIA supports both D3D12_HEAP and OPAQUE_WIN32; AMD/Intel typically only OPAQUE_WIN32.
    // Therefore, OPAQUE_WIN32 is the only universally compatible handle type for cross-vendor.
    if (srcCaps.supportsOpaqueWin32 && dstCaps.supportsOpaqueWin32) {
        caps.recommendedType = ExternalHandleType::OpaqueWin32;
        caps.nvidiaToAmd = srcNvidia && (dstAmd || dstIntel);
        caps.nvidiaToIntel = srcNvidia && dstIntel;
        caps.amdToNvidia = (srcAmd || srcIntel) && dstNvidia;
        caps.notes = "OPAQUE_WIN32: Universal cross-vendor on Windows (both sides use same handle type)";
    }
    // NVIDIA-to-NVIDIA can optionally use D3D12_HEAP for better performance
    else if (srcNvidia && dstNvidia && srcCaps.supportsD3D12Heap && dstCaps.supportsD3D12Heap) {
        caps.recommendedType = ExternalHandleType::D3D12Heap;
        caps.notes = "D3D12_HEAP: NVIDIA-to-NVIDIA only (better performance)";
    }
    #endif
    
    return caps;
}

} // namespace vvm