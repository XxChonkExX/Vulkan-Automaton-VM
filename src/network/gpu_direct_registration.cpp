#include "vulkan_vm/network/gpu_direct_registration.hpp"
#include "vulkan_vm/utils.hpp"

#include <infiniband/verbs.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <sstream>

namespace vvm {
namespace network {

// ============================================================================
// NVIDIA GPUDirect Registration (VK_NV_external_memory_rdma + nvidia-peermem)
// ============================================================================

static bool isPeermemLoaded() {
    return std::filesystem::exists("/sys/module/nvidia_peermem");
}

static std::optional<uint64_t> findGpuBarAddress(VkPhysicalDevice physicalDevice) {
    // Try to find the GPU's PCI BAR address from sysfs
    // This is a simplified approach - in production you'd match by vendor/device ID
    for (const auto& entry : std::filesystem::directory_iterator("/sys/bus/pci/devices")) {
        if (!entry.is_directory()) continue;
        
        std::string path = entry.path().string() + "/resource";
        std::ifstream file(path);
        if (!file.is_open()) continue;
        
        std::string line;
        if (std::getline(file, line)) {
            // Parse "start end flags" format
            std::istringstream iss(line);
            uint64_t start, end, flags;
            if (iss >> std::hex >> start >> end >> flags) {
                // Check if this is a memory BAR (flags & IORESOURCE_MEM)
                if (flags & 0x20000000) { // IORESOURCE_MEM
                    // This is a memory BAR - could be the GPU's VRAM
                    // For now, return the first memory BAR found
                    // In production, match by vendor/device ID from VkPhysicalDeviceProperties
                    return start;
                }
            }
        }
    }
    return std::nullopt;
}

std::optional<GpuDirectRegistration> registerGpuMemoryForRdma(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceMemory memory,
    VkDeviceSize offset,
    VkDeviceSize size,
    const std::string& nicName) {

    GpuDirectRegistration reg;

    // Get the remote address via VK_NV_external_memory_rdma
    PFN_vkGetMemoryRemoteAddressNV vkGetMemoryRemoteAddressNV =
        (PFN_vkGetMemoryRemoteAddressNV)vkGetDeviceProcAddr(device, "vkGetMemoryRemoteAddressNV");
    if (!vkGetMemoryRemoteAddressNV) {
        VVM_LOG_ERROR("vkGetMemoryRemoteAddressNV not available");
        return std::nullopt;
    }

    VkMemoryGetRemoteAddressInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_REMOTE_ADDRESS_INFO_NV;
    info.memory = memory;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_RDMA_ADDRESS_BIT_NV;

    VkResult result = vkGetMemoryRemoteAddressNV(device, &info, &reg.remoteAddress);
    if (result != VK_SUCCESS) {
        VVM_LOG_ERROR("vkGetMemoryRemoteAddressNV failed: {}", vkResultToString(result));
        return std::nullopt;
    }

    // Try to register the GPU's PCI BAR for a functional ibv_mr
    // This requires nvidia-peermem kernel module
    if (isPeermemLoaded()) {
        VVM_LOG_INFO("nvidia-peermem kernel module detected, attempting BAR registration");
        
        // Find the GPU's PCI BAR address
        auto barAddr = findGpuBarAddress(physicalDevice);
        if (barAddr) {
            // The remote address from VK_NV_external_memory_rdma IS the PCI BAR address
            // We can try to register it with ibv_reg_mr
            // Note: This requires the PD from the verbs context, which we don't have here
            // The actual MR registration happens in VerbsRdmaTransport::registerGpuMemory
            // where we have the PD. We just provide the remote address here.
            VVM_LOG_INFO("Found GPU PCI BAR at {:#x}", *barAddr);
        }
    } else {
        VVM_LOG_WARN("nvidia-peermem kernel module not loaded. GPU-direct RDMA will not have local MR.");
        VVM_LOG_WARN("Install nvidia-peermem (from nvidia-peermem GitHub) for full GPU-direct support.");
    }

    reg.valid = true;
    reg.rkey = 0; // Set if/when we register an MR
    VVM_LOG_INFO("NVIDIA GPUDirect: remote address obtained (0x{})",
                 reinterpret_cast<uint64_t>(reg.remoteAddress));

    return reg;
}

void unregisterVendorGpuMemory(const GpuDirectRegistration& reg, uint32_t vendorId) {
    if (vendorId == 0x10DE && reg.mr) {
        ibv_dereg_mr(reg.mr);
    }
}

// ============================================================================
// Vendor-specific dispatch
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
        case 0x10DE: // NVIDIA
            return registerGpuMemoryForRdma(device, physicalDevice, memory, offset, size, nicName);

        case 0x1002: // AMD
        case 0x8086: // Intel
            // AMD/Intel use DMA-BUF path, which is handled in VerbsRdmaTransport
            // directly since it needs the PD from the verbs context.
            VVM_LOG_WARN("AMD/Intel GPUDirect: use DMA-BUF path in VerbsRdmaTransport");
            return std::nullopt;

        default:
            VVM_LOG_WARN("Unknown GPU vendor {:#x} for GPUDirect", vendorId);
            return std::nullopt;
    }
}

} // namespace network
} // namespace vvm