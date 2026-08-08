#pragma once

// VkRemoteAddressNV (VK_KHR_external_memory_rdma) is a beta extension type and
// is only emitted by vulkan_core.h when this macro is set. Keep it local so any
// TU including this header (on any platform) gets the type.
#ifndef VK_ENABLE_BETA_EXTENSIONS
#define VK_ENABLE_BETA_EXTENSIONS 1
#endif
#include <vulkan/vulkan.h>

#include <optional>
#include <string>
#include <cstdint>

#if defined(VVM_NETWORK_HAS_VERBS)
#include <infiniband/verbs.h>
#else
struct ibv_mr;  // opaque; only used as an incomplete pointer type on non-verbs builds
#endif

namespace vvm {
namespace network {

// ============================================================================
// GPU-direct memory registration helper (NVIDIA path)
// ============================================================================

struct GpuDirectRegistration {
    VkRemoteAddressNV remoteAddress = {0};
    uint32_t rkey = 0;
    struct ibv_mr* mr = nullptr;  // verbs memory region
    bool valid = false;
};

// Register VkDeviceMemory for NVIDIA GPUDirect RDMA
std::optional<GpuDirectRegistration> registerGpuMemoryForRdma(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceMemory memory,
    VkDeviceSize offset,
    VkDeviceSize size,
    const std::string& nicName = "");

// ============================================================================
// Vendor-specific GPU-direct registration
// ============================================================================

// Vendor-agnostic dispatch: returns the appropriate registration struct
// based on GPU vendor ID.
std::optional<GpuDirectRegistration> registerGpuMemoryForRdmaVendor(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkDeviceMemory memory,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkQueue transferQueue,
    uint32_t transferQueueFamily,
    const std::string& nicName,
    uint32_t vendorId);

void unregisterVendorGpuMemory(const GpuDirectRegistration& reg, uint32_t vendorId);

} // namespace network
} // namespace vvm