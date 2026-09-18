// external_semaphore_test: export/import round-trip for VkSemaphore payloads.
//
// Timeline path (no queue needed): create exportable timeline -> export ->
// dup -> import dup -> CPU-signal original -> CPU-wait imported ->
// counter check. Binary path: same export/import, then a real queue
// submit signals the original and waits the import.
//
// Skips (exit 0) with no device, or when the driver lacks the external
// semaphore extensions - CI runners take this path.

#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/cross_gpu/external_semaphore.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(VVM_PLATFORM_WINDOWS)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace vvm;

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

#define SKIP(msg)                                                              \
    do {                                                                       \
        std::printf("SKIP: %s\n", msg);                                        \
        return 0;                                                              \
    } while (0)

namespace {

bool hasExtension(VkPhysicalDevice phys, const char* name) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> exts(count);
    if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &count, exts.data()) != VK_SUCCESS) {
        return false;
    }
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

// Independent OS handle for a second importer (dup/DuplicateHandle).
// Mirrors duplicateForImport() for the semaphore flow.
ExternalHandle dupHandle(const ExternalHandle& src) {
    if (!src) return ExternalHandle{};
#if defined(VVM_PLATFORM_WINDOWS)
    HANDLE out = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), src.get(), GetCurrentProcess(),
                         &out, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
        return ExternalHandle{};
    }
    return ExternalHandle(out);
#else
    const int fd = dup(src.get());
    if (fd < 0) return ExternalHandle{};
    return ExternalHandle(fd);
#endif
}

// Timeline round-trip without touching a queue.
bool timelineRoundTrip(VkDevice device, VkExternalSemaphoreHandleTypeFlagBits ht) {
    auto orig = createExportableSemaphore(device, ht, true);
    CHECK(orig.has_value());
    if (!orig.has_value()) return false;

    auto exported = exportSemaphoreHandle(device, orig->semaphore, ht);
    CHECK(exported.has_value());
    if (!exported.has_value()) return false;

    // Peer B imports its OWN handle; the exporter's stays valid.
    ExternalHandle peer = dupHandle(*exported);
    CHECK(peer);
    if (!peer) return false;
    auto imported = importSemaphoreHandle(device, std::move(peer), ht, true);
    CHECK(imported.has_value());
    if (!imported.has_value()) return false;

    constexpr uint64_t kValue = 7;
    CHECK(signalTimelineSemaphore(device, orig->semaphore, kValue));
    CHECK(waitTimelineSemaphore(device, imported->semaphore, kValue,
                                5ull * 1000 * 1000 * 1000));
    uint64_t counter = 0;
    CHECK(vkGetSemaphoreCounterValue(device, imported->semaphore, &counter) == VK_SUCCESS);
    CHECK(counter == kValue);
    return failures == 0;
}

// Binary round-trip through a real queue submit (signal original, wait import).
bool binaryRoundTrip(VkDevice device, VkQueue queue,
                     VkExternalSemaphoreHandleTypeFlagBits ht) {
    auto orig = createExportableSemaphore(device, ht, false);
    CHECK(orig.has_value());
    if (!orig.has_value()) return false;

    auto exported = exportSemaphoreHandle(device, orig->semaphore, ht);
    CHECK(exported.has_value());
    if (!exported.has_value()) return false;

    ExternalHandle peer = dupHandle(*exported);
    CHECK(peer);
    if (!peer) return false;
    auto imported = importSemaphoreHandle(device, std::move(peer), ht, false);
    CHECK(imported.has_value());
    if (!imported.has_value()) return false;

    // Submit 1 signals the original; submit 2 waits the import. No command
    // buffers - pure sync exercise.
    VkSubmitInfo sig{};
    sig.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    sig.signalSemaphoreCount = 1;
    sig.pSignalSemaphores = &orig->semaphore;
    CHECK(vkQueueSubmit(queue, 1, &sig, VK_NULL_HANDLE) == VK_SUCCESS);
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo wait{};
    wait.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    wait.waitSemaphoreCount = 1;
    wait.pWaitSemaphores = &imported->semaphore;
    wait.pWaitDstStageMask = &stage;
    CHECK(vkQueueSubmit(queue, 1, &wait, VK_NULL_HANDLE) == VK_SUCCESS);
    CHECK(vkQueueWaitIdle(queue) == VK_SUCCESS);
    return failures == 0;
}

}  // namespace

int main() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM ExternalSemaphore Test";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        SKIP("vkCreateInstance failed");
    }

    auto devices = enumerateDevices(instance);
    if (devices.empty()) SKIP("no physical devices");
    const VkPhysicalDevice phys = devices[0].device;

    const VkExternalSemaphoreHandleTypeFlagBits ht = defaultSemaphoreHandleType();

    // Capability query is instance-level: exercised even on the SKIP path.
    const ExternalSemaphoreCaps tlCaps = queryExternalSemaphoreCaps(phys, ht, true);
    const ExternalSemaphoreCaps biCaps = queryExternalSemaphoreCaps(phys, ht, false);
    std::printf("timeline caps: export=%d import=%d; binary caps: export=%d import=%d\n",
                tlCaps.exportable ? 1 : 0, tlCaps.importable ? 1 : 0,
                biCaps.exportable ? 1 : 0, biCaps.importable ? 1 : 0);
    if (!tlCaps.exportable || !tlCaps.importable) {
        SKIP("driver refuses timeline semaphore export/import");
    }

    // Device extensions needed for the round-trip.
    const char* kExtBase = "VK_KHR_external_semaphore";
#if defined(VVM_PLATFORM_WINDOWS)
    const char* kExtOs = "VK_KHR_external_semaphore_win32";
#else
    const char* kExtOs = "VK_KHR_external_semaphore_fd";
#endif
    const char* kExtTimeline = "VK_KHR_timeline_semaphore";
    if (!hasExtension(phys, kExtBase) || !hasExtension(phys, kExtOs)) {
        SKIP("device lacks external semaphore extensions");
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    const bool core12 = props.apiVersion >= VK_API_VERSION_1_2;
    const bool khrTimeline = hasExtension(phys, kExtTimeline);

    std::vector<const char*> exts = {kExtBase, kExtOs};
    if (!core12 && khrTimeline) exts.push_back(kExtTimeline);

    VkPhysicalDeviceVulkan12Features feats12{};
    feats12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    feats12.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceTimelineSemaphoreFeaturesKHR featsKhr{};
    featsKhr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR;
    featsKhr.timelineSemaphore = VK_TRUE;

    auto queues = findQueueFamilies(phys);
    if (!queues.transfer && !queues.graphics && !queues.compute) {
        SKIP("no usable queue family");
    }
    const uint32_t qf = queues.transfer.value_or(
        queues.graphics.value_or(queues.compute.value()));
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = (core12 || khrTimeline)
        ? (core12 ? static_cast<void*>(&feats12) : static_cast<void*>(&featsKhr))
        : nullptr;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    dci.ppEnabledExtensionNames = exts.data();
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &dci, nullptr, &device) != VK_SUCCESS) {
        SKIP("vkCreateDevice with semaphore extensions failed");
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, qf, 0, &queue);

    const int baseFailures = failures;
    if (timelineRoundTrip(device, ht)) {
        std::printf("timeline export/import round-trip: PASS\n");
    }
    if (queue != VK_NULL_HANDLE && failures == baseFailures) {
        if (binaryRoundTrip(device, queue, ht)) {
            std::printf("binary export/import round-trip: PASS\n");
        }
    }

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    if (failures == 0) {
        std::printf("=== EXTERNAL SEMAPHORE TEST CLEAN ===\n");
        return 0;
    }
    std::printf("=== EXTERNAL SEMAPHORE TEST FAILURES (%d) ===\n", failures);
    return 1;
}
