#pragma once

// External semaphores: export/import VkSemaphore payloads as OS handles
// (opaque fd on Linux/Android, opaque Win32 on Windows) for cross-device
// and cross-process GPU synchronization.
//
// The multi-process inference story: process A renders/exports a semaphore,
// process B imports it and waits - no host round-trip, no polling. Same
// pattern feeds foreign APIs (CUDA/HIP import the same opaque fd).
//
// Ownership follows the external-memory contract (LIFETIME_CONTRACT R4):
// a successful import CONSUMES exactly one OS handle (the driver owns it
// afterwards). Use duplicateForImport() first when several peers import
// the same payload - Vulkan consumes one handle per successful import.
//
// All functions fail soft (nullopt/false + log) when the device lacks the
// external-semaphore extensions or the driver refuses the handle type.

#include "vulkan_vm/core.hpp"
#include "vulkan_vm/utils.hpp"

namespace vvm {

// Platform default handle type for semaphore export/import.
inline VkExternalSemaphoreHandleTypeFlagBits defaultSemaphoreHandleType() {
#if defined(VVM_PLATFORM_WINDOWS)
    return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

// Exportable/importable support for one (handle type, semaphore kind) pair.
struct ExternalSemaphoreCaps {
    bool exportable = false;
    bool importable = false;
};

// Query whether physicalDevice can export/import semaphore payloads of the
// given handle type and kind (timeline = true for timeline semaphores).
// Instance-level query (core Vulkan 1.1), no device needed.
ExternalSemaphoreCaps VVM_API queryExternalSemaphoreCaps(
    VkPhysicalDevice physicalDevice,
    VkExternalSemaphoreHandleTypeFlagBits handleType,
    bool timeline);

// Move-only RAII owner of an exportable-or-imported VkSemaphore. Destroys
// the semaphore; the VkDevice stays caller-owned (same split as Allocation).
struct VVM_API ExternalSemaphore {
    VkDevice device = VK_NULL_HANDLE;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    bool isTimeline = false;

    ExternalSemaphore() = default;
    ExternalSemaphore(VkDevice dev, VkSemaphore sem, bool timeline)
        : device(dev), semaphore(sem), isTimeline(timeline) {}
    ~ExternalSemaphore();
    ExternalSemaphore(ExternalSemaphore&& other) noexcept;
    ExternalSemaphore& operator=(ExternalSemaphore&& other) noexcept;
    ExternalSemaphore(const ExternalSemaphore&) = delete;
    ExternalSemaphore& operator=(const ExternalSemaphore&) = delete;

    explicit operator bool() const { return semaphore != VK_NULL_HANDLE; }
    // Release ownership without destroying (hand-off to another owner).
    VkSemaphore release();
};

// Create a semaphore carrying VkExportSemaphoreCreateInfo for handleType.
// timeline = true needs the timelineSemaphore feature enabled on device
// (core 1.2 or VK_KHR_timeline_semaphore); returns nullopt otherwise.
[[nodiscard]] VVM_API std::optional<ExternalSemaphore> createExportableSemaphore(
    VkDevice device,
    VkExternalSemaphoreHandleTypeFlagBits handleType,
    bool timeline);

// Export the payload to a fresh OS handle. The semaphore stays valid:
// timelines can be re-exported freely; a binary can be re-exported once
// it is back in the unsignaled state.
[[nodiscard]] VVM_API std::optional<ExternalHandle> exportSemaphoreHandle(
    VkDevice device,
    VkSemaphore semaphore,
    VkExternalSemaphoreHandleTypeFlagBits handleType);

// Import an OS handle as a semaphore of the given kind. Consumes the
// handle on success (driver-owned afterwards); on failure the handle
// stays valid and closes with its RAII owner.
[[nodiscard]] VVM_API std::optional<ExternalSemaphore> importSemaphoreHandle(
    VkDevice device,
    ExternalHandle&& handle,
    VkExternalSemaphoreHandleTypeFlagBits handleType,
    bool timeline);

// CPU-side timeline signal (no queue needed). Returns false for binary
// semaphores - signal those with a queue submit.
VVM_API bool signalTimelineSemaphore(VkDevice device, VkSemaphore timeline,
                                     uint64_t value);
// CPU-side timeline wait (no queue needed). Returns false on timeout
// (VK_TIMEOUT) or query failure - never blocks past timeoutNs.
VVM_API bool waitTimelineSemaphore(VkDevice device, VkSemaphore timeline,
                                   uint64_t value,
                                   uint64_t timeoutNs = UINT64_MAX);

}  // namespace vvm
