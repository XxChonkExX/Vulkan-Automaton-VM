#include "vulkan_vm/cross_gpu/external_semaphore.hpp"
#include "vulkan_vm/utils.hpp"

#if defined(VVM_PLATFORM_WINDOWS)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace vvm {

ExternalSemaphore::~ExternalSemaphore() {
    if (semaphore != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
        vkDestroySemaphore(device, semaphore, nullptr);
    }
}

ExternalSemaphore::ExternalSemaphore(ExternalSemaphore&& other) noexcept
    : device(other.device), semaphore(other.semaphore), isTimeline(other.isTimeline) {
    other.device = VK_NULL_HANDLE;
    other.semaphore = VK_NULL_HANDLE;
}

ExternalSemaphore& ExternalSemaphore::operator=(ExternalSemaphore&& other) noexcept {
    if (this != &other) {
        if (semaphore != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        device = other.device;
        semaphore = other.semaphore;
        isTimeline = other.isTimeline;
        other.device = VK_NULL_HANDLE;
        other.semaphore = VK_NULL_HANDLE;
    }
    return *this;
}

VkSemaphore ExternalSemaphore::release() {
    VkSemaphore s = semaphore;
    semaphore = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    return s;
}

ExternalSemaphoreCaps queryExternalSemaphoreCaps(
    VkPhysicalDevice physicalDevice,
    VkExternalSemaphoreHandleTypeFlagBits handleType,
    bool timeline) {
    ExternalSemaphoreCaps caps;
    if (physicalDevice == VK_NULL_HANDLE) return caps;
    VkSemaphoreTypeCreateInfo typeInfo{};
    typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    typeInfo.semaphoreType =
        timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
    typeInfo.initialValue = 0;
    VkPhysicalDeviceExternalSemaphoreInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    info.pNext = &typeInfo;
    info.handleType = handleType;
    VkExternalSemaphoreProperties props{};
    props.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    vkGetPhysicalDeviceExternalSemaphoreProperties(physicalDevice, &info, &props);
    caps.exportable =
        (props.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) != 0;
    caps.importable =
        (props.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0;
    return caps;
}

std::optional<ExternalSemaphore> createExportableSemaphore(
    VkDevice device,
    VkExternalSemaphoreHandleTypeFlagBits handleType,
    bool timeline) {
    if (device == VK_NULL_HANDLE) return std::nullopt;
    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.handleTypes = handleType;
    VkSemaphoreTypeCreateInfo typeInfo{};
    typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    typeInfo.semaphoreType =
        timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
    typeInfo.initialValue = 0;
    typeInfo.pNext = &exportInfo;
    VkSemaphoreCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    ci.pNext = &typeInfo;
    VkSemaphore sem = VK_NULL_HANDLE;
    if (vkCreateSemaphore(device, &ci, nullptr, &sem) != VK_SUCCESS) {
        VVM_LOG_WARN("createExportableSemaphore: vkCreateSemaphore refused "
                     "(timeline={}, missing feature or extension?)",
                     timeline ? 1 : 0);
        return std::nullopt;
    }
    return ExternalSemaphore(device, sem, timeline);
}

std::optional<ExternalHandle> exportSemaphoreHandle(
    VkDevice device,
    VkSemaphore semaphore,
    VkExternalSemaphoreHandleTypeFlagBits handleType) {
    if (device == VK_NULL_HANDLE || semaphore == VK_NULL_HANDLE) {
        return std::nullopt;
    }
#if defined(VVM_PLATFORM_WINDOWS)
    if (handleType != VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
        VVM_LOG_WARN("exportSemaphoreHandle: only OPAQUE_WIN32 supported on Windows");
        return std::nullopt;
    }
    auto getHandle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
    if (!getHandle) {
        VVM_LOG_WARN("exportSemaphoreHandle: VK_KHR_external_semaphore_win32 absent");
        return std::nullopt;
    }
    VkSemaphoreGetWin32HandleInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    info.semaphore = semaphore;
    info.handleType = handleType;
    HANDLE h = nullptr;
    if (getHandle(device, &info, &h) != VK_SUCCESS || !h) {
        VVM_LOG_WARN("exportSemaphoreHandle: GetSemaphoreWin32Handle failed");
        return std::nullopt;
    }
    return ExternalHandle(h);
#else
    if (handleType != VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
        VVM_LOG_WARN("exportSemaphoreHandle: only OPAQUE_FD supported on POSIX");
        return std::nullopt;
    }
    auto getFd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
    if (!getFd) {
        VVM_LOG_WARN("exportSemaphoreHandle: VK_KHR_external_semaphore_fd absent");
        return std::nullopt;
    }
    VkSemaphoreGetFdInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    info.semaphore = semaphore;
    info.handleType = handleType;
    int fd = -1;
    if (getFd(device, &info, &fd) != VK_SUCCESS || fd < 0) {
        VVM_LOG_WARN("exportSemaphoreHandle: GetSemaphoreFd failed");
        return std::nullopt;
    }
    return ExternalHandle(fd);
#endif
}

std::optional<ExternalSemaphore> importSemaphoreHandle(
    VkDevice device,
    ExternalHandle&& handle,
    VkExternalSemaphoreHandleTypeFlagBits handleType,
    bool timeline) {
    if (device == VK_NULL_HANDLE || !handle) return std::nullopt;
    VkSemaphoreTypeCreateInfo typeInfo{};
    typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    typeInfo.semaphoreType =
        timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
    typeInfo.initialValue = 0;
    VkSemaphoreCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    ci.pNext = &typeInfo;
    VkSemaphore sem = VK_NULL_HANDLE;
    if (vkCreateSemaphore(device, &ci, nullptr, &sem) != VK_SUCCESS) {
        VVM_LOG_WARN("importSemaphoreHandle: vkCreateSemaphore refused");
        return std::nullopt;
    }
#if defined(VVM_PLATFORM_WINDOWS)
    auto importHandle = reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkImportSemaphoreWin32HandleKHR"));
    if (!importHandle) {
        VVM_LOG_WARN("importSemaphoreHandle: VK_KHR_external_semaphore_win32 absent");
        vkDestroySemaphore(device, sem, nullptr);
        return std::nullopt;
    }
    VkImportSemaphoreWin32HandleInfoKHR importInfo{};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
    importInfo.semaphore = sem;
    importInfo.handleType = handleType;
    importInfo.handle = handle.get();
    if (importHandle(device, &importInfo) != VK_SUCCESS) {
        VVM_LOG_WARN("importSemaphoreHandle: ImportSemaphoreWin32Handle refused");
        vkDestroySemaphore(device, sem, nullptr);
        return std::nullopt;
    }
#else
    auto importFd = reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(
        vkGetDeviceProcAddr(device, "vkImportSemaphoreFdKHR"));
    if (!importFd) {
        VVM_LOG_WARN("importSemaphoreHandle: VK_KHR_external_semaphore_fd absent");
        vkDestroySemaphore(device, sem, nullptr);
        return std::nullopt;
    }
    VkImportSemaphoreFdInfoKHR importInfo{};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
    importInfo.semaphore = sem;
    importInfo.handleType = handleType;
    importInfo.fd = handle.get();
    if (importFd(device, &importInfo) != VK_SUCCESS) {
        VVM_LOG_WARN("importSemaphoreHandle: ImportSemaphoreFd refused");
        vkDestroySemaphore(device, sem, nullptr);
        return std::nullopt;
    }
#endif
    // Consumed: the driver owns the OS handle now (R4 consume semantics).
    handle.release();
    return ExternalSemaphore(device, sem, timeline);
}

bool signalTimelineSemaphore(VkDevice device, VkSemaphore timeline, uint64_t value) {
    if (device == VK_NULL_HANDLE || timeline == VK_NULL_HANDLE) return false;
    VkSemaphoreSignalInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    info.semaphore = timeline;
    info.value = value;
    if (vkSignalSemaphore(device, &info) != VK_SUCCESS) {
        VVM_LOG_WARN("signalTimelineSemaphore: vkSignalSemaphore refused (value={})", value);
        return false;
    }
    return true;
}

bool waitTimelineSemaphore(VkDevice device, VkSemaphore timeline, uint64_t value,
                           uint64_t timeoutNs) {
    if (device == VK_NULL_HANDLE || timeline == VK_NULL_HANDLE) return false;
    VkSemaphoreWaitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    info.semaphoreCount = 1;
    info.pSemaphores = &timeline;
    info.pValues = &value;
    const VkResult r = vkWaitSemaphores(device, &info, timeoutNs);
    if (r == VK_TIMEOUT) return false;
    if (r != VK_SUCCESS) {
        VVM_LOG_WARN("waitTimelineSemaphore: vkWaitSemaphores failed ({})",
                     static_cast<int>(r));
        return false;
    }
    return true;
}

}  // namespace vvm
