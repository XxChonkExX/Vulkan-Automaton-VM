#include "vulkan_vm/network/gpu_direct_registration.hpp"
#include "vulkan_vm/utils.hpp"

#include <mutex>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <handleapi.h>
#endif

namespace vvm {
namespace network {

namespace detail {

// ============================================================================
// Intel Level Zero path (Windows) - OPAQUE_WIN32 export
// ============================================================================

static std::once_flag s_zeInitFlag;
static bool s_zeAvailable = false;
static ze_driver_handle_t s_zeDriver = nullptr;

static bool initLevelZero() {
    std::call_once(s_zeInitFlag, []() {
        ze_result_t res = zeInit(ZE_INIT_FLAG_GPU_ONLY);
        if (res != ZE_RESULT_SUCCESS) {
            VVM_LOG_WARN("Level Zero init failed: 0x{:x}", res);
            return;
        }
        uint32_t count = 0;
        res = zeDriverGet(&count, nullptr);
        if (res != ZE_RESULT_SUCCESS || count == 0) {
            VVM_LOG_WARN("No Level Zero drivers found");
            return;
        }
        std::vector<ze_driver_handle_t> drivers(count);
        res = zeDriverGet(&count, drivers.data());
        if (res == ZE_RESULT_SUCCESS && count > 0) {
            s_zeDriver = drivers[0];
            s_zeAvailable = true;
            VVM_LOG_INFO("Level Zero initialized, driver: {}", reinterpret_cast<uintptr_t>(s_zeDriver));
        }
    });
    return s_zeAvailable;
}

std::optional<GpuDirectRegistration> registerIntelLevelZero(const GpuDirectConfig& config) {
    if (!initLevelZero()) {
        VVM_LOG_WARN("Level Zero not available for Intel GPU-direct");
        return std::nullopt;
    }

    GpuDirectRegistration reg;
    reg.vendorId = 0x8086;

    ze_context_handle_t context = config.zeContext;
    ze_device_handle_t device = config.zeDevice;

    if (!context || !device) {
        VVM_LOG_ERROR("Intel path: missing zeContext or zeDevice in config");
        return std::nullopt;
    }

    VkExternalMemoryHandleTypeFlagBits exportType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkMemoryGetWin32HandleInfoKHR win32Info{};
    win32Info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    win32Info.memory = config.vkMemory;
    win32Info.handleType = exportType;

    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR =
        (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(config.vkDevice, "vkGetMemoryWin32HandleKHR");
    if (!vkGetMemoryWin32HandleKHR) {
        VVM_LOG_ERROR("vkGetMemoryWin32HandleKHR not available");
        return std::nullopt;
    }

    HANDLE win32Handle = nullptr;
    VkResult vkRes = vkGetMemoryWin32HandleKHR(config.vkDevice, &win32Info, &win32Handle);
    if (vkRes != VK_SUCCESS) {
        VVM_LOG_ERROR("vkGetMemoryWin32HandleKHR failed: {}", vkResultToString(vkRes));
        return std::nullopt;
    }

    reg.win32Handle = win32Handle;
    reg.zeExportFlags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;

    ze_ipc_mem_handle_t zeHandle{};
    std::memcpy(&zeHandle, &win32Handle, sizeof(HANDLE));
    reg.zeHandle = zeHandle;

    VkMemoryGetRemoteAddressInfoNV remoteInfo{};
    remoteInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_REMOTE_ADDRESS_INFO_NV;
    remoteInfo.memory = config.vkMemory;
    remoteInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_RDMA_ADDRESS_BIT_NV;

    PFN_vkGetMemoryRemoteAddressNV vkGetMemoryRemoteAddressNV =
        (PFN_vkGetMemoryRemoteAddressNV)vkGetDeviceProcAddr(config.vkDevice, "vkGetMemoryRemoteAddressNV");
    if (vkGetMemoryRemoteAddressNV) {
        VkResult r = vkGetMemoryRemoteAddressNV(config.vkDevice, &remoteInfo, &reg.remoteAddress);
        if (r == VK_SUCCESS) {
            VVM_LOG_INFO("Intel L0: got RDMA remote address 0x{:x}", reinterpret_cast<uint64_t>(reg.remoteAddress));
        }
    }

    reg.valid = true;
    VVM_LOG_INFO("Intel Level Zero GPU-direct registered: Win32 handle={}, L0 handle={}",
                 reg.win32Handle, reinterpret_cast<uintptr_t>(reg.zeHandle));
    return reg;
}

void unregisterIntelLevelZero(const GpuDirectRegistration& reg) {
    if (reg.win32Handle && reg.win32Handle != INVALID_HANDLE_VALUE) {
        CloseHandle(reg.win32Handle);
    }
    VVM_LOG_INFO("Intel Level Zero GPU-direct unregistered");
}

// ============================================================================
// AMD ROCm path (Linux) - DMA-BUF import/export
// ============================================================================

#if defined(__linux__)
static std::once_flag s_hipInitFlag;
static bool s_hipAvailable = false;

static bool initHip() {
    std::call_once(s_hipInitFlag, []() {
        void* hipLib = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_GLOBAL);
        if (!hipLib) hipLib = dlopen("libhip.so", RTLD_LAZY | RTLD_GLOBAL);
        if (hipLib) {
            s_hipAvailable = true;
            VVM_LOG_INFO("ROCm/HIP library loaded");
        } else {
            VVM_LOG_WARN("ROCm/HIP not available (libamdhip64.so not found)");
        }
    });
    return s_hipAvailable;
}

using hipImportExternalMemoryFn = hipError_t(*)(hipExternalMemory_t*, const hipExternalMemoryHandleDesc*);
using hipExternalMemoryGetMappedBufferFn = hipError_t(*)(void**, hipExternalMemory_t, const hipExternalMemoryBufferDesc*);
using hipDestroyExternalMemoryFn = hipError_t(*)(hipExternalMemory_t);

static hipImportExternalMemoryFn s_hipImportExternalMemory = nullptr;
static hipExternalMemoryGetMappedBufferFn s_hipExternalMemoryGetMappedBuffer = nullptr;
static hipDestroyExternalMemoryFn s_hipDestroyExternalMemory = nullptr;

static void loadHipSymbols() {
    if (s_hipImportExternalMemory) return;
    void* hipLib = dlopen("libamdhip64.so", RTLD_LAZY);
    if (!hipLib) hipLib = dlopen("libhip.so", RTLD_LAZY);
    if (hipLib) {
        s_hipImportExternalMemory = (hipImportExternalMemoryFn)dlsym(hipLib, "hipImportExternalMemory");
        s_hipExternalMemoryGetMappedBuffer = (hipExternalMemoryGetMappedBufferFn)dlsym(hipLib, "hipExternalMemoryGetMappedBuffer");
        s_hipDestroyExternalMemory = (hipDestroyExternalMemoryFn)dlsym(hipLib, "hipDestroyExternalMemory");
    }
}

std::optional<GpuDirectRegistration> registerAmdRocm(const GpuDirectConfig& config) {
    if (!initHip()) {
        VVM_LOG_WARN("ROCm/HIP not available for AMD GPU-direct");
        return std::nullopt;
    }
    loadHipSymbols();
    if (!s_hipImportExternalMemory || !s_hipExternalMemoryGetMappedBuffer || !s_hipDestroyExternalMemory) {
        VVM_LOG_ERROR("Required HIP external memory symbols not found");
        return std::nullopt;
    }

    GpuDirectRegistration reg;
    reg.vendorId = 0x1002;

    VkMemoryGetFdInfoKHR fdInfo{};
    fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fdInfo.memory = config.vkMemory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR =
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(config.vkDevice, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR) {
        VVM_LOG_ERROR("vkGetMemoryFdKHR not available");
        return std::nullopt;
    }

    int dmaBufFd = -1;
    VkResult vkRes = vkGetMemoryFdKHR(config.vkDevice, &fdInfo, &dmaBufFd);
    if (vkRes != VK_SUCCESS) {
        VVM_LOG_ERROR("vkGetMemoryFdKHR failed: {}", vkResultToString(vkRes));
        return std::nullopt;
    }

    reg.dmaBufFd = dmaBufFd;

    hipExternalMemoryHandleDesc desc{};
    desc.type = HIP_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF;
    desc.handle.fd = dmaBufFd;
    desc.size = config.size;
    desc.flags = 0;

    hipExternalMemory_t extMem = nullptr;
    hipError_t err = s_hipImportExternalMemory(&extMem, &desc);
    if (err != hipSuccess) {
        VVM_LOG_ERROR("hipImportExternalMemory failed: {}", err);
        close(dmaBufFd);
        return std::nullopt;
    }
    reg.hipExternalMemory = extMem;

    hipExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = config.offset;
    bufDesc.size = config.size;
    bufDesc.flags = 0;

    void* mappedPtr = nullptr;
    err = s_hipExternalMemoryGetMappedBuffer(&mappedPtr, extMem, &bufDesc);
    if (err != hipSuccess) {
        VVM_LOG_ERROR("hipExternalMemoryGetMappedBuffer failed: {}", err);
        s_hipDestroyExternalMemory(extMem);
        close(dmaBufFd);
        return std::nullopt;
    }
    reg.hipMappedPtr = mappedPtr;

    VkMemoryGetRemoteAddressInfoNV remoteInfo{};
    remoteInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_REMOTE_ADDRESS_INFO_NV;
    remoteInfo.memory = config.vkMemory;
    remoteInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_RDMA_ADDRESS_BIT_NV;

    PFN_vkGetMemoryRemoteAddressNV vkGetMemoryRemoteAddressNV =
        (PFN_vkGetMemoryRemoteAddressNV)vkGetDeviceProcAddr(config.vkDevice, "vkGetMemoryRemoteAddressNV");
    if (vkGetMemoryRemoteAddressNV) {
        VkResult r = vkGetMemoryRemoteAddressNV(config.vkDevice, &remoteInfo, &reg.remoteAddress);
        if (r == VK_SUCCESS) {
            VVM_LOG_INFO("AMD ROCm: got RDMA remote address 0x{:x}", reinterpret_cast<uint64_t>(reg.remoteAddress));
        }
    }

    reg.valid = true;
    VVM_LOG_INFO("AMD ROCm GPU-direct registered: dma-buf fd={}, mapped ptr={}",
                 dmaBufFd, mappedPtr);
    return reg;
}

void unregisterAmdRocm(const GpuDirectRegistration& reg) {
    if (s_hipDestroyExternalMemory && reg.hipExternalMemory) {
        s_hipDestroyExternalMemory(static_cast<hipExternalMemory_t>(reg.hipExternalMemory));
    }
    if (reg.dmaBufFd >= 0) {
        close(reg.dmaBufFd);
    }
    VVM_LOG_INFO("AMD ROCm GPU-direct unregistered");
}
#else
std::optional<GpuDirectRegistration> registerAmdRocm(const GpuDirectConfig&) {
    return std::nullopt;
}
void unregisterAmdRocm(const GpuDirectRegistration&) {}
#endif

// ============================================================================
// NVIDIA Vulkan path (existing, cleaned up)
// ============================================================================

std::optional<GpuDirectRegistration> registerNvidiaVulkan(const GpuDirectConfig& config) {
    GpuDirectRegistration reg;
    reg.vendorId = 0x10DE;

    PFN_vkGetMemoryRemoteAddressNV vkGetMemoryRemoteAddressNV =
        (PFN_vkGetMemoryRemoteAddressNV)vkGetDeviceProcAddr(config.vkDevice, "vkGetMemoryRemoteAddressNV");
    if (!vkGetMemoryRemoteAddressNV) {
        VVM_LOG_ERROR("vkGetMemoryRemoteAddressNV not available");
        return std::nullopt;
    }

    VkMemoryGetRemoteAddressInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_REMOTE_ADDRESS_INFO_NV;
    info.memory = config.vkMemory;
    info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_RDMA_ADDRESS_BIT_NV;

    VkResult result = vkGetMemoryRemoteAddressNV(config.vkDevice, &info, &reg.remoteAddress);
    if (result != VK_SUCCESS) {
        VVM_LOG_ERROR("vkGetMemoryRemoteAddressNV failed: {}", vkResultToString(result));
        return std::nullopt;
    }

    reg.valid = true;
    VVM_LOG_INFO("NVIDIA GPUDirect: remote address obtained (0x{})",
                 reinterpret_cast<uint64_t>(reg.remoteAddress));
    return reg;
}

void unregisterNvidiaVulkan(const GpuDirectRegistration& reg) {
    VVM_LOG_INFO("NVIDIA GPU-direct unregistered");
}

} // namespace detail

std::optional<GpuDirectRegistration> registerGpuMemoryForRdmaVendor(
    const GpuDirectConfig& config) {

    switch (config.vendorId) {
        case 0x10DE: // NVIDIA
            return detail::registerNvidiaVulkan(config);

        case 0x8086: // Intel
            return detail::registerIntelLevelZero(config);

        case 0x1002: // AMD
            return detail::registerAmdRocm(config);

        default:
            VVM_LOG_WARN("Unknown GPU vendor {:#x} for GPUDirect", config.vendorId);
            return std::nullopt;
    }
}

void unregisterVendorGpuMemory(const GpuDirectRegistration& reg) {
    switch (reg.vendorId) {
        case 0x10DE:
            detail::unregisterNvidiaVulkan(reg);
            break;
        case 0x8086:
            detail::unregisterIntelLevelZero(reg);
            break;
        case 0x1002:
            detail::unregisterAmdRocm(reg);
            break;
        default:
            break;
    }
}

} // namespace network
} // namespace vvm