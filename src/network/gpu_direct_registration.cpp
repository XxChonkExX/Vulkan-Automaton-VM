#include "vulkan_vm/network/gpu_direct_registration.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <vector>
#include <memory>
#include <algorithm>

namespace vvm {
namespace network {

// ============================================================================
// Helper: Find memory type index for imported memory (cross-device)
// ============================================================================
static std::optional<uint32_t> findImportMemoryTypeIndex(
    VkPhysicalDevice physicalDevice,
    VkMemoryPropertyFlags requiredFlags,
    VkExternalMemoryHandleTypeFlagBits handleType) {
    
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    
    VkPhysicalDeviceExternalBufferInfo extInfo{};
    extInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    extInfo.handleType = handleType;
    extInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    
    VkExternalBufferProperties extProps{};
    extProps.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(physicalDevice, &extInfo, &extProps);
    
    VkExternalMemoryHandleTypeFlags compatibleHandleTypes = extProps.externalMemoryProperties.compatibleHandleTypes;
    if ((compatibleHandleTypes & handleType) == 0) {
        VVM_LOG_WARN("findImportMemoryTypeIndex: handle type %u not supported on destination device", handleType);
        return std::nullopt;
    }
    
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memProps.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags) {
            return i;
        }
    }
    
    VVM_LOG_WARN("findImportMemoryTypeIndex: no memory type with required flags 0x%x", requiredFlags);
    return std::nullopt;
}

// ============================================================================
// Helper: Allocate and map host staging buffer for fallback
// ============================================================================
static std::optional<std::pair<VkBuffer, void*>> createHostStagingBuffer(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceSize size,
    VkQueue transferQueue,
    uint32_t transferQueueFamily) {
    
    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = size;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkBuffer buffer;
    if (vkCreateBuffer(device, &bufInfo, nullptr, &buffer) != VK_SUCCESS) {
        return std::nullopt;
    }
    
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);
    
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    
    uint32_t memType = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & 
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memType = i;
            break;
        }
    }
    
    if (memType == UINT32_MAX) {
        vkDestroyBuffer(device, buffer, nullptr);
        return std::nullopt;
    }
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memType;
    
    VkDeviceMemory memory;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        return std::nullopt;
    }
    
    if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        return std::nullopt;
    }
    
    void* mappedPtr = nullptr;
    if (vkMapMemory(device, memory, 0, size, 0, &mappedPtr) != VK_SUCCESS) {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        return std::nullopt;
    }
    
    return std::make_pair(buffer, mappedPtr);
}

// ============================================================================
// NVIDIA GPUDirect RDMA Registration
// ============================================================================

std::optional<GpuDirectRegistration> registerGpuMemoryForRdmaNvidia(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceMemory memory,
    VkDeviceSize offset,
    VkDeviceSize size,
    const std::string& nicName) {
    
    (void)nicName;  // Not used for NVIDIA path
    
    PFN_vkGetMemoryRemoteAddressNV vkGetMemoryRemoteAddressNV = 
        (PFN_vkGetMemoryRemoteAddressNV)vkGetDeviceProcAddr(device, "vkGetMemoryRemoteAddressNV");
    
    if (!vkGetMemoryRemoteAddressNV) {
        VVM_LOG_ERROR("NVIDIA GPUDirect: VK_NV_external_memory_rdma not supported on device");
        return std::nullopt;
    }
    
    VkMemoryGetRemoteAddressInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_REMOTE_ADDRESS_INFO_NV;
    info.memory = memory;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_RDMA_ADDRESS_BIT_NV;
    
    VkRemoteAddressNV remoteAddr{};
    VkResult result = vkGetMemoryRemoteAddressNV(device, &info, &remoteAddr);
    if (result != VK_SUCCESS) {
        VVM_LOG_ERROR("NVIDIA GPUDirect: vkGetMemoryRemoteAddressNV failed: %s", vkResultToString(result).c_str());
        return std::nullopt;
    }
    
    GpuDirectRegistration reg;
    reg.remoteAddress = remoteAddr;
    reg.rkey = 0;  // Will be set by ibv_reg_mr on Linux, or via Windows NDKPI
    reg.mr = nullptr;
    reg.valid = true;
    
    VVM_LOG_INFO("NVIDIA GPUDirect registration: remoteAddr=0x%llx", 
                 static_cast<unsigned long long>(remoteAddr.address));
    
    return reg;
}

void unregisterGpuMemoryForRdmaNvidia(const GpuDirectRegistration& reg) {
    if (reg.mr) {
        // ibv_dereg_mr(reg.mr);  // Linux
        // Or Windows NDKPI deregistration
        reg.mr = nullptr;
    }
}

// ============================================================================
// AMD GPU-Direct Registration (Windows)
// ============================================================================

#ifdef VVM_PLATFORM_WINDOWS
#include <windows.h>
#include <winternl.h>

// Map a Win32 handle (from VkExportMemoryWin32HandleInfoKHR) to CPU VA
static void* mapWin32HandleForDma(HANDLE handle, size_t size) {
    // Use MapViewOfFile to get a CPU-visible mapping of the GPU memory
    // This requires the handle to have FILE_MAP_READ | FILE_MAP_WRITE
    HANDLE mapHandle = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), handle, GetCurrentProcess(), &mapHandle,
                         FILE_MAP_READ | FILE_MAP_WRITE, FALSE, 0)) {
        VVM_LOG_ERROR("AMD GPUDirect: DuplicateHandle failed: %lu", GetLastError());
        return nullptr;
    }
    
    void* ptr = MapViewOfFile(mapHandle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, size);
    CloseHandle(mapHandle);
    
    if (!ptr) {
        VVM_LOG_ERROR("AMD GPUDirect: MapViewOfFile failed: %lu", GetLastError());
    }
    return ptr;
}

static void unmapWin32HandleForDma(void* ptr) {
    if (ptr) UnmapViewOfFile(ptr);
}
#endif

std::optional<DmaBufRegistration> registerDmaBufForRdmaAmd(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceMemory memory,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkQueue transferQueue,
    uint32_t transferQueueFamily,
    const std::string& nicName) {
    
    (void)nicName;
    
    // Step 1: Export as OPAQUE_WIN32 handle
    VkMemoryGetWin32HandleInfoKHR getHandleInfo{};
    getHandleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    getHandleInfo.memory = memory;
    getHandleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
    
    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = 
        (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
    if (!vkGetMemoryWin32HandleKHR) {
        VVM_LOG_ERROR("AMD GPUDirect: vkGetMemoryWin32HandleKHR not available");
        return std::nullopt;
    }
    
    HANDLE win32Handle = nullptr;
    VkResult result = vkGetMemoryWin32HandleKHR(device, &getHandleInfo, &win32Handle);
    if (result != VK_SUCCESS || !win32Handle) {
        VVM_LOG_ERROR("AMD GPUDirect: vkGetMemoryWin32HandleKHR failed: %s", vkResultToString(result).c_str());
        return std::nullopt;
    }
    
    // Step 2: Map the handle to CPU VA (on Windows, MapViewOfFile)
    void* cpuVa = nullptr;
#ifdef VVM_PLATFORM_WINDOWS
    cpuVa = mapWin32HandleForDma(win32Handle, static_cast<size_t>(size));
#else
    (void)win32Handle;
    (void)device;
    (void)physicalDevice;
    (void)offset;
    (void)size;
    (void)transferQueue;
    (void)transferQueueFamily;
    VVM_LOG_WARN("AMD GPUDirect: DMA-BUF path not implemented on Linux yet");
    return std::nullopt;
#endif
    
    if (!cpuVa) {
        CloseHandle(win32Handle);
        return std::nullopt;
    }
    
    DmaBufRegistration reg;
    reg.fd = reinterpret_cast<intptr_t>(win32Handle);  // Store handle in fd field on Windows
    reg.mr = nullptr;  // Will be set by ibv_reg_mr on Linux, or Windows NDKPI equivalent
    reg.rkey = 0;
    reg.valid = true;
    
    VVM_LOG_INFO("AMD GPUDirect registration: handle=%p, mapped CPU VA=%p, size=%llu",
                 win32Handle, cpuVa, static_cast<unsigned long long>(size));
    
    return reg;
}

void unregisterDmaBufForRdmaAmd(const DmaBufRegistration& reg) {
    if (reg.mr) {
        // ibv_dereg_mr(reg.mr);  // Linux
        reg.mr = nullptr;
    }
#ifdef VVM_PLATFORM_WINDOWS
    if (reg.fd != -1) {
        HANDLE handle = reinterpret_cast<HANDLE>(reg.fd);
        // Note: we don't CloseHandle here because the Vulkan export handle 
        // is owned by the caller (ExternalMemoryInfo RAII wrapper)
    }
#endif
}

// ============================================================================
// Intel GPU-Direct Registration (Windows)
// ============================================================================

std::optional<DmaBufRegistration> registerDmaBufForRdmaIntel(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceMemory memory,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkQueue transferQueue,
    uint32_t transferQueueFamily,
    const std::string& nicName) {
    
    // Intel Arc on Windows also exports as OPAQUE_WIN32 handle
    // Same implementation as AMD
    return registerDmaBufForRdmaAmd(device, physicalDevice, memory, offset, size,
                                    transferQueue, transferQueueFamily, nicName);
}

void unregisterDmaBufForRdmaIntel(const DmaBufRegistration& reg) {
    unregisterDmaBufForRdmaAmd(reg);
}

// ============================================================================
// Unified Vendor Dispatcher
// ============================================================================

std::optional<GpuDirectRegistration> registerGpuMemoryForRdmaVendor(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceMemory memory,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkQueue transferQueue,
    uint32_t transferQueueFamily,
    const std::string& nicName,
    uint32_t vendorId) {
    
    switch (vendorId) {
        case 0x10DE:  // NVIDIA
            return registerGpuMemoryForRdmaNvidia(device, physicalDevice, memory, offset, size, nicName);
        case 0x1002:  // AMD
            return registerGpuMemoryForRdmaNvidia(device, physicalDevice, memory, offset, size, nicName);
        case 0x8086:  // Intel
            return registerGpuMemoryForRdmaNvidia(device, physicalDevice, memory, offset, size, nicName);
        default:
            VVM_LOG_WARN("registerGpuMemoryForRdmaVendor: unknown vendor 0x%x, trying NVIDIA path", vendorId);
            return registerGpuMemoryForRdmaNvidia(device, physicalDevice, memory, offset, size, nicName);
    }
}

std::optional<DmaBufRegistration> registerDmaBufForRdmaVendor(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceMemory memory,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkQueue transferQueue,
    uint32_t transferQueueFamily,
    const std::string& nicName,
    uint32_t vendorId) {
    
    switch (vendorId) {
        case 0x10DE:  // NVIDIA
            // NVIDIA uses GpuDirectRegistration, not DmaBufRegistration
            VVM_LOG_WARN("NVIDIA uses registerGpuMemoryForRdmaVendor, not DmaBuf path");
            return std::nullopt;
        case 0x1002:  // AMD
            return registerDmaBufForRdmaAmd(device, physicalDevice, memory, offset, size,
                                           transferQueue, transferQueueFamily, nicName);
        case 0x8086:  // Intel
            return registerDmaBufForRdmaIntel(device, physicalDevice, memory, offset, size,
                                             transferQueue, transferQueueFamily, nicName);
        default:
            VVM_LOG_WARN("registerDmaBufForRdmaVendor: unknown vendor 0x%x, trying AMD path", vendorId);
            return registerDmaBufForRdmaAmd(device, physicalDevice, memory, offset, size,
                                           transferQueue, transferQueueFamily, nicName);
    }
}

void unregisterVendorGpuMemory(const GpuDirectRegistration& reg, uint32_t vendorId) {
    switch (vendorId) {
        case 0x10DE:
            unregisterGpuMemoryForRdmaNvidia(reg);
            break;
        case 0x1002:
            // AMD uses DmaBufRegistration path
            break;
        case 0x8086:
            // Intel uses DmaBufRegistration path
            break;
    }
}

void unregisterVendorDmaBuf(const DmaBufRegistration& reg, uint32_t vendorId) {
    switch (vendorId) {
        case 0x10DE:
            break;
        case 0x1002:
            unregisterDmaBufForRdmaAmd(reg);
            break;
        case 0x8086:
            unregisterDmaBufForRdmaIntel(reg);
            break;
    }
}

} // namespace network
} // namespace vvm