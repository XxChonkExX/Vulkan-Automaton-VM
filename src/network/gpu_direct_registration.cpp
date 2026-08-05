#include "vulkan_vm/network/rdma_transport.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <infiniband/verbs.h>
#include <dlfcn.h>
#include <cstring>

namespace vvm {
namespace network {

// ============================================================================
// NVIDIA GPUDirect RDMA Registration
// ============================================================================

std::optional<GpuDirectRegistration> registerGpuMemoryForRdma(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceMemory memory,
    VkDeviceSize offset,
    VkDeviceSize size,
    const std::string& nicName) {
    
    // This requires:
    // 1. VK_NV_external_memory_rdma extension
    // 2. NVIDIA GPUDirect RDMA kernel module (nvidia-peermem)
    // 3. ibv_reg_mr on the GPU memory
    
    // Get function pointers
    PFN_vkGetMemoryRemoteAddressNV vkGetMemoryRemoteAddressNV = 
        (PFN_vkGetMemoryRemoteAddressNV)vkGetDeviceProcAddr(device, "vkGetMemoryRemoteAddressNV");
    
    if (!vkGetMemoryRemoteAddressNV) {
        VVM_LOG_ERROR("VK_NV_external_memory_rdma not supported");
        return std::nullopt;
    }
    
    // Get remote address
    VkMemoryGetRemoteAddressInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_REMOTE_ADDRESS_INFO_NV;
    info.memory = memory;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_RDMA_ADDRESS_BIT_NV;
    
    VkRemoteAddressNV remoteAddr{};
    VkResult result = vkGetMemoryRemoteAddressNV(device, &info, &remoteAddr);
    if (result != VK_SUCCESS) {
        VVM_LOG_ERROR("vkGetMemoryRemoteAddressNV failed: {}", vkResultToString(result));
        return std::nullopt;
    }
    
    // For actual ibv_reg_mr, we need the GPU memory pointer
    // This is vendor-specific and requires nvidia-peermem module
    // The remote address from Vulkan is used by the peer for RDMA operations
    
    GpuDirectRegistration reg;
    reg.remoteAddress = remoteAddr;
    reg.rkey = 0;  // Would be set after ibv_reg_mr
    reg.mr = nullptr;
    reg.valid = true;
    
    VVM_LOG_INFO("GPU-direct registration: remoteAddr=0x{:x}", remoteAddr.address);
    
    return reg;
}

void unregisterGpuMemoryForRdma(const GpuDirectRegistration& reg) {
    if (reg.mr) {
        ibv_dereg_mr(reg.mr);
    }
}

// ============================================================================
// DMA-BUF Registration (AMD/Intel path)
// ============================================================================

std::optional<DmaBufRegistration> registerDmaBufForRdma(
    int dmaBufFd,
    size_t size,
    const std::string& nicName) {
    
    // This requires:
    // 1. DMA-BUF fd from VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
    // 2. ibv_reg_mr on the DMA-BUF fd
    // 3. AMD ROCm RDMA or Intel equivalent kernel support
    
    // Open RDMA device
    int numDevices = 0;
    struct ibv_device** devList = ibv_get_device_list(&numDevices);
    if (!devList || numDevices == 0) return std::nullopt;
    
    struct ibv_device* chosenDev = devList[0];
    if (!nicName.empty()) {
        for (int i = 0; i < numDevices; ++i) {
            if (std::string(ibv_get_device_name(devList[i])) == nicName) {
                chosenDev = devList[i];
                break;
            }
        }
    }
    
    struct ibv_context* ctx = ibv_open_device(chosenDev);
    ibv_free_device_list(devList);
    
    if (!ctx) return std::nullopt;
    
    struct ibv_pd* pd = ibv_alloc_pd(ctx);
    if (!pd) {
        ibv_close_device(ctx);
        return std::nullopt;
    }
    
    // Register DMA-BUF
    // This requires kernel support for DMA-BUF registration
    // struct ibv_reg_dmabuf_mr_attr attr{};
    // attr.pd = pd;
    // attr.fd = dmaBufFd;
    // attr.length = size;
    // ... 
    
    // Note: ibv_reg_dmabuf_mr is a newer API, may not be available everywhere
    // Fallback: use regular ibv_reg_mr if we can get the CPU pointer
    
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    
    VVM_LOG_WARN("DMA-BUF registration not fully implemented");
    return std::nullopt;
}

void unregisterDmaBufForRdma(const DmaBufRegistration& reg) {
    if (reg.mr) {
        ibv_dereg_mr(reg.mr);
    }
}

} // namespace network
} // namespace vvm