// storage_dstorage_test: L3 DirectStorage zero-copy bridge — GPU-gated.
// Verifies the LUID-matched D3D12 shared heap imports into the Vulkan pool as
// real, usable device memory: fill a pattern via a host-visible staging,
// vkCmdCopy into the imported heap, copy back out, verify. Skips cleanly
// without a suitable GPU or when the driver cannot import D3D12_HEAP handles.
//
// NOTE: real calls are never inside assert() — bound to values, checked
// explicitly (assert compiles out under NDEBUG).

#include "vulkan_vm/storage/dstorage_backend.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <cstring>
#include <iostream>
#include <set>
#include <vector>

using namespace vvm;
using namespace vvm::storage::backend;

int main() {
    std::cout << "storage_dstorage_test (zero-copy bridge, GPU-gated)\n";
    std::cout << "  runtime(dstorage.dll): " << (DStorageBackend::runtimeAvailable() ? "yes" : "no")
              << "  d3d12: " << (DStorageBackend::d3d12Available() ? "yes" : "no") << "\n";

    // ---- Vulkan instance ----
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VVM DStorage Bridge Test";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> wanted = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
    uint32_t availCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &availCount, nullptr);
    std::vector<VkExtensionProperties> avail(availCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &availCount, avail.data());
    std::vector<const char*> instanceExts;
    for (const char* w : wanted)
        for (auto& e : avail)
            if (!std::strcmp(e.extensionName, w)) { instanceExts.push_back(w); break; }

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    ici.enabledExtensionCount = static_cast<uint32_t>(instanceExts.size());
    ici.ppEnabledExtensionNames = instanceExts.data();
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        std::cout << "SKIP: vkCreateInstance failed\n";
        return 0;
    }

    auto devices = enumerateDevices(instance);
    if (devices.empty()) { std::cout << "SKIP: no GPUs\n"; vkDestroyInstance(instance, nullptr); return 0; }
    std::cout << "Found " << devices.size() << " device(s); trying each for the bridge\n";

    // external_memory_win32 is required for the D3D12_HEAP import; checked
    // per device below.

    // ---- per-device bridge attempt: find any card whose driver imports ----
    int failures = 0;
    bool bridged = false;

    for (const auto& cand : devices) {
        if (!checkDeviceExtensionSupport(cand.device, {"VK_KHR_external_memory_win32"})) {
            std::cout << "  " << cand.props.deviceName << ": no VK_KHR_external_memory_win32, skip\n";
            continue;
        }
        VkPhysicalDeviceVulkan12Features supp12{};
        supp12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceFeatures2 suppFeat{};
        suppFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        suppFeat.pNext = &supp12;
        vkGetPhysicalDeviceFeatures2(cand.device, &suppFeat);
        if (!supp12.bufferDeviceAddress || !supp12.timelineSemaphore) continue;

        auto queues = findQueueFamilies(cand.device);
        float prio = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> qis;
        std::set<uint32_t> seen;
        auto addQ = [&](std::optional<uint32_t> f) {
            if (f && seen.insert(*f).second) {
                VkDeviceQueueCreateInfo q{};
                q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                q.queueFamilyIndex = *f;
                q.queueCount = 1;
                q.pQueuePriorities = &prio;
                qis.push_back(q);
            }
        };
        addQ(queues.graphics);
        addQ(queues.compute);
        addQ(queues.transfer);

        VkPhysicalDeviceVulkan12Features v12{};
        v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        v12.bufferDeviceAddress = VK_TRUE;
        v12.timelineSemaphore = VK_TRUE;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &v12;

        std::vector<const char*> devExts = {
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
            "VK_KHR_external_memory_win32",
        };
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = static_cast<uint32_t>(qis.size());
        dci.pQueueCreateInfos = qis.data();
        dci.pNext = &f2;
        dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
        dci.ppEnabledExtensionNames = devExts.data();
        VkDevice device = VK_NULL_HANDLE;
        if (vkCreateDevice(cand.device, &dci, nullptr, &device) != VK_SUCCESS) continue;

        uint32_t tf = queues.transfer.value_or(queues.compute.value_or(queues.graphics.value_or(0)));
        VkQueue tq = VK_NULL_HANDLE, gq = VK_NULL_HANDLE, cq = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, tf, 0, &tq);
        if (queues.graphics) vkGetDeviceQueue(device, *queues.graphics, 0, &gq);
        if (queues.compute) vkGetDeviceQueue(device, *queues.compute, 0, &cq);

        DeviceConfig dc{};
        dc.physicalDevice = cand.device;
        dc.device = device;
        dc.graphicsQueueFamily = queues.graphics.value_or(0);
        dc.computeQueueFamily = queues.compute.value_or(dc.graphicsQueueFamily);
        dc.transferQueueFamily = tf;
        dc.graphicsQueue = gq;
        dc.computeQueue = cq;
        dc.transferQueue = tq;

        PoolConfig pc{};
        pc.blockSize = 64 * 1024 * 1024;
        pc.minAlignment = 64 * 1024;
        pc.enableHostVisible = true;
        pc.enableExternal = true;
        pc.enableDeviceAddress = true;
        pc.maxBlocks = 4;

        auto pool = UnifiedMemoryPool::create(dc, pc);
        if (!pool) { vkDestroyDevice(device, nullptr); continue; }

        DStorageBackend be({});
        constexpr VkDeviceSize kHeapBytes = 64ull * 1024 * 1024;
        constexpr VkDeviceSize kTestBytes = 4ull * 1024 * 1024;

        Allocation imported;
        Result r = be.importToPool(&*pool, cand.device, kHeapBytes, &imported);
        if (!r) {
            std::cout << "  " << cand.props.deviceName << ": " << r.message << "\n";
            pool.reset();
            vkDestroyDevice(device, nullptr);
            continue;
        }
        std::cout << "  bridged on: " << cand.props.deviceName << " ("
                  << (kHeapBytes >> 20) << " MiB shared heap)\n";

        // ---- round-trip through the imported memory ----
        {
            auto staging = pool->allocate(kTestBytes,
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            auto readback = pool->allocate(kTestBytes,
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (!staging || !staging->hostPtr || !readback || !readback->hostPtr) {
                std::cerr << "FAIL: staging/readback alloc\n";
                ++failures;
            } else {
                for (VkDeviceSize i = 0; i < kTestBytes; ++i)
                    static_cast<uint8_t*>(staging->hostPtr)[i] = static_cast<uint8_t>(i * 7 + 3);
                std::memset(readback->hostPtr, 0, static_cast<size_t>(kTestBytes));

                const bool w = pool->copyBuffer(*staging, imported, 0, 0, kTestBytes);
                const bool b = pool->copyBuffer(imported, *readback, 0, 0, kTestBytes);
                if (!w || !b) { std::cerr << "FAIL: copy through imported memory\n"; ++failures; }
                else {
                    const bool ok = std::memcmp(staging->hostPtr, readback->hostPtr,
                                                static_cast<size_t>(kTestBytes)) == 0;
                    if (ok) std::cout << "  bridge OK: pattern round-tripped through the "
                                         "imported shared heap\n";
                    else { std::cerr << "FAIL: data mismatch through imported heap\n"; ++failures; }
                }

                pool->deallocate(std::move(*staging));
                pool->deallocate(std::move(*readback));
            }
        }

        pool->deallocate(std::move(imported));
        pool.reset();
        vkDestroyDevice(device, nullptr);
        bridged = true;
        break;
    }

    if (!bridged && failures == 0) {
        std::cout << "SKIP: no device on this machine imports D3D12_HEAP handles "
                     "(NVIDIA is the D3D12-bridge vendor; AMD/Intel are OPAQUE_WIN32)\n";
    }

    vkDestroyInstance(instance, nullptr);

    std::cout << (failures == 0 ? "All bridge tests passed!\n" : "Bridge tests FAILED\n");
    return failures == 0 ? 0 : 1;
}