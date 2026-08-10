#pragma once

#ifndef VK_ENABLE_BETA_EXTENSIONS
#define VK_ENABLE_BETA_EXTENSIONS 1
#endif
#include <vulkan/vulkan.h>

#include <optional>
#include <string>
#include <cstdint>
#include <memory>

#if defined(VVM_NETWORK_HAS_VERBS)
#include <infiniband/verbs.h>
#else
struct ibv_mr;
#endif

#if defined(_WIN32) && defined(VVM_HAS_LEVEL_ZERO)
#define VVM_LEVEL_ZERO_AVAILABLE 1
#include <level_zero/ze_api.h>
#else
typedef struct _ze_device_handle_t* ze_device_handle_t;
typedef struct _ze_context_handle_t* ze_context_handle_t;
typedef void* ze_ipc_mem_handle_t;
#endif

namespace vvm {
namespace network {

struct GpuDirectRegistration {
    VkRemoteAddressNV remoteAddress = {0};
    uint32_t rkey = 0;
    struct ibv_mr* mr = nullptr;
    bool valid = false;

    // Level Zero (Intel) - Windows
#if defined(VVM_LEVEL_ZERO_AVAILABLE)
    ze_ipc_mem_handle_t zeHandle = nullptr;
    HANDLE win32Handle = nullptr;
    ze_external_memory_type_flags_t zeExportFlags = 0;
#endif

    // ROCm (AMD) - Linux
#if defined(__linux__)
    int dmaBufFd = -1;
    void* hipExternalMemory = nullptr;
    void* hipMappedPtr = nullptr;
#endif

    // Vendor identification
    uint32_t vendorId = 0;
};

struct GpuDirectConfig {
    VkDevice vkDevice = VK_NULL_HANDLE;
    VkPhysicalDevice vkPhysicalDevice = VK_NULL_HANDLE;
    VkDeviceMemory vkMemory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    VkQueue transferQueue = VK_NULL_HANDLE;
    uint32_t transferQueueFamily = 0;
    std::string nicName;
    uint32_t vendorId = 0;

    // Level Zero handles (for Intel path)
    ze_context_handle_t zeContext = nullptr;
    ze_device_handle_t zeDevice = nullptr;

    // ROCm handles (for AMD path)
    void* hipDevice = nullptr;
    void* hipContext = nullptr;
};

std::optional<GpuDirectRegistration> registerGpuMemoryForRdmaVendor(
    const GpuDirectConfig& config);

void unregisterVendorGpuMemory(const GpuDirectRegistration& reg);

namespace detail {
    std::optional<GpuDirectRegistration> registerIntelLevelZero(const GpuDirectConfig& config);
    std::optional<GpuDirectRegistration> registerAmdRocm(const GpuDirectConfig& config);
    std::optional<GpuDirectRegistration> registerNvidiaVulkan(const GpuDirectConfig& config);
    void unregisterIntelLevelZero(const GpuDirectRegistration& reg);
    void unregisterAmdRocm(const GpuDirectRegistration& reg);
    void unregisterNvidiaVulkan(const GpuDirectRegistration& reg);
}

} // namespace network
} // namespace vvm