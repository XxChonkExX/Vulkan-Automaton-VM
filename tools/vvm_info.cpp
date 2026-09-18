// vvm-info: read-only diagnostics bundle for bug reports. Prints instance
// version, then per physical device: identity, heaps/types, queue families,
// key features (device address, timeline semaphores), external-memory caps,
// and the cross-device P2P policy matrix. Creates no VkDevice and no pools.
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"
#include "vulkan_vm/cross_gpu/external_memory.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vvm;

static const char* devTypeStr(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: return "other";
    }
}

static bool hasDevExt(VkPhysicalDevice dev, const char* name) {
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> exts(n);
    if (vkEnumerateDeviceExtensionProperties(dev, nullptr, &n, exts.data()) != VK_SUCCESS) {
        return false;
    }
    for (auto& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    std::printf("== vvm-info ==\n");

    uint32_t loaderVer = 0;
    if (vkEnumerateInstanceVersion(&loaderVer) == VK_SUCCESS) {
        std::printf("loader: %u.%u.%u\n", VK_API_VERSION_MAJOR(loaderVer),
                    VK_API_VERSION_MINOR(loaderVer), VK_API_VERSION_PATCH(loaderVer));
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "vvm-info";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        std::printf("FAIL: vkCreateInstance\n");
        return 1;
    }

    auto devices = enumerateDevices(instance);
    std::printf("devices: %zu\n", devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto& d = devices[i];
        const auto& p = d.props;
        std::printf("\nDevice %zu: %s\n", i, p.deviceName);
        char ven[8], did[8];
        std::snprintf(ven, sizeof(ven), "0x%04x", p.vendorID);
        std::snprintf(did, sizeof(did), "0x%04x", p.deviceID);
        std::printf("  vendor=%s device=%s type=%s api=%u.%u.%u driver=0x%08x\n",
                    ven, did, devTypeStr(p.deviceType),
                    VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
                    VK_API_VERSION_PATCH(p.apiVersion), p.driverVersion);

        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(d.device, &mp);
        for (uint32_t h = 0; h < mp.memoryHeapCount; ++h) {
            std::printf("  heap %u: %llu MB%s\n", h,
                        (unsigned long long)(mp.memoryHeaps[h].size / (1024 * 1024)),
                        (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                            ? " DEVICE_LOCAL" : "");
        }
        std::printf("  memory types: %u\n", mp.memoryTypeCount);
        for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
            std::printf("    type %u: heap %u flags 0x%x\n", t,
                        mp.memoryTypes[t].heapIndex, mp.memoryTypes[t].propertyFlags);
        }

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d.device, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(d.device, &qn, qf.data());
        for (uint32_t q = 0; q < qn; ++q) {
            std::printf("  queue %u: count=%u%s%s%s\n", q, qf[q].queueCount,
                        (qf[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) ? " G" : "",
                        (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) ? " C" : "",
                        (qf[q].queueFlags & VK_QUEUE_TRANSFER_BIT) ? " T" : "");
        }

        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &f12;
        vkGetPhysicalDeviceFeatures2(d.device, &f2);
        std::printf("  features: deviceAddress=%s timelineSemaphore=%s sparseBinding=%s\n",
                    f12.bufferDeviceAddress ? "yes" : "no",
                    f12.timelineSemaphore ? "yes" : "no",
                    f2.features.sparseBinding ? "yes" : "no");

        const ExternalMemoryCaps caps = queryExternalMemoryCaps(d.device);
        std::printf("  external: opaqueWin32=%s d3d12Heap=%s opaqueFd=%s dmaBuf=%s hostImport=%s\n",
                    caps.supportsOpaqueWin32 ? "yes" : "no",
                    caps.supportsD3D12Heap ? "yes" : "no",
                    caps.supportsOpaqueFd ? "yes" : "no",
                    caps.supportsDmaBuf ? "yes" : "no",
                    hasDevExt(d.device, "VK_EXT_external_memory_host") ? "yes" : "no");
    }

    // P2P policy matrix over discrete pairs (mirrors the copyDeviceToDevice
    // gate: same-vendor direct on Windows, host-staged otherwise).
    std::printf("\nP2P policy:\n");
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i].props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        for (size_t j = 0; j < devices.size(); ++j) {
            if (i == j) continue;
            if (devices[j].props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
            const bool same = devices[i].vendorID == devices[j].vendorID;
            std::printf("  %zu -> %zu: %s\n", i, j,
                        same ? "direct-import (same vendor)"
                             : "host-staged / shared-arena (cross-vendor)");
        }
    }

    vkDestroyInstance(instance, nullptr);
    std::printf("\n(vvm-info is read-only: no VkDevice created, no pools, no allocations)\n");
    return 0;
}
