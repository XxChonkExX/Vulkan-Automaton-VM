/*
 * VulkanVM - Offload/Reload Verification Test
 *
 * Verifies device <-> host shadow round trip: write a known byte pattern to
 * a HOST_VISIBLE allocation, offload it (device -> host), verify the pattern
 * survives in the shadow, reload it (host -> device), and confirm the pattern
 * is intact.
 */

#include <vulkan_vm/vulkan_vm.hpp>
#include <vulkan_vm/utils.hpp>

#include <iostream>
#include <vector>
#include <set>
#include <cstring>
#include <cstdint>

using namespace vvm;

static void fillPattern(void* buf, VkDeviceSize size, uint8_t val) {
    std::memset(buf, val, static_cast<size_t>(size));
}

static bool verifyPattern(const void* buf, VkDeviceSize size, uint8_t expected) {
    const auto* p = static_cast<const uint8_t*>(buf);
    for (VkDeviceSize i = 0; i < size; ++i) {
        if (p[i] != expected) {
            std::cerr << "  verify failed at offset " << i << ": got "
                      << static_cast<int>(p[i]) << ", expected "
                      << static_cast<int>(expected) << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "=== VulkanVM Offload/Reload Test ===\n\n";

    // ---- Vulkan instance ----
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM Offload Test";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> instanceExts = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExts.size());
    instanceInfo.ppEnabledExtensionNames = instanceExts.data();

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
        std::cerr << "FAIL: create instance\n"; return 1;
    }

    auto devices = enumerateDevices(instance);
    auto bestDevice = selectBestDevice(devices, true, 1024);
    if (!bestDevice) { std::cerr << "FAIL: no suitable GPU\n"; return 1; }
    std::cout << "Using: " << bestDevice->props.deviceName << "\n";

    // ---- Logical device ----
    std::vector<const char*> deviceExts = {
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
    };
    if (!checkDeviceExtensionSupport(bestDevice->device, deviceExts)) {
        std::cerr << "FAIL: required device extensions not supported\n"; return 1;
    }

    auto queues = findQueueFamilies(bestDevice->device);
    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    std::set<uint32_t> uniqueFamilies;
    auto addQueue = [&](std::optional<uint32_t> fam) {
        if (fam && uniqueFamilies.insert(*fam).second) {
            VkDeviceQueueCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            info.queueFamilyIndex = *fam;
            info.queueCount = 1;
            info.pQueuePriorities = &priority;
            queueInfos.push_back(info);
        }
    };
    addQueue(queues.graphics);
    addQueue(queues.compute);
    addQueue(queues.transfer);

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDeviceBufferDeviceAddressFeatures addrFeatures{};
    addrFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    addrFeatures.bufferDeviceAddress = VK_TRUE;
    addrFeatures.pNext = features2.pNext;
    features2.pNext = &addrFeatures;
    VkPhysicalDeviceTimelineSemaphoreFeatures tsFeatures{};
    tsFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    tsFeatures.timelineSemaphore = VK_TRUE;
    tsFeatures.pNext = features2.pNext;
    features2.pNext = &tsFeatures;
    VkPhysicalDeviceVulkan12Features v12Features{};
    v12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12Features.bufferDeviceAddress = VK_TRUE;
    v12Features.timelineSemaphore = VK_TRUE;
    v12Features.pNext = features2.pNext;
    features2.pNext = &v12Features;

    VkDeviceCreateInfo devInfo{};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    devInfo.pQueueCreateInfos = queueInfos.data();
    devInfo.pNext = &features2;
    devInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExts.size());
    devInfo.ppEnabledExtensionNames = deviceExts.data();

    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(bestDevice->device, &devInfo, nullptr, &device) != VK_SUCCESS) {
        std::cerr << "FAIL: create device\n"; return 1;
    }

    VkQueue transferQueue = VK_NULL_HANDLE;
    uint32_t transferFamily = queues.transfer.value_or(queues.compute.value_or(queues.graphics.value_or(0)));
    vkGetDeviceQueue(device, transferFamily, 0, &transferQueue);

    // ---- Pool ----
    DeviceConfig devConfig{};
    devConfig.physicalDevice = bestDevice->device;
    devConfig.device = device;
    devConfig.graphicsQueueFamily = queues.graphics.value_or(0);
    devConfig.computeQueueFamily = queues.compute.value_or(devConfig.graphicsQueueFamily);
    devConfig.transferQueueFamily = transferFamily;
    VkQueue gq = VK_NULL_HANDLE, cq = VK_NULL_HANDLE;
    if (queues.graphics) vkGetDeviceQueue(device, *queues.graphics, 0, &gq);
    if (queues.compute) vkGetDeviceQueue(device, *queues.compute, 0, &cq);
    devConfig.graphicsQueue = gq;
    devConfig.computeQueue = cq;
    devConfig.transferQueue = transferQueue;

    PoolConfig poolConfig{};
    poolConfig.blockSize = 256 * 1024 * 1024;
    poolConfig.minAlignment = 256 * 1024;
    poolConfig.enableHostVisible = true;
    poolConfig.enableExternal = false;
    poolConfig.enableDeviceAddress = false;
    poolConfig.maxBlocks = 4;

    auto pool = UnifiedMemoryPool::create(devConfig, poolConfig);
    if (!pool) { std::cerr << "FAIL: create pool\n"; return 1; }
    std::cout << "Pool created\n";

    int failures = 0;
    const VkDeviceSize kTestSize = 4 * 1024 * 1024;  // 4 MB

    auto alloc = pool->allocate(
        kTestSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!alloc || !alloc->hostPtr) { std::cerr << "FAIL: allocate\n"; return 1; }
    std::cout << "Allocated " << kTestSize << " bytes\n";

    // ---- Write pattern, offload, reload round-trip ----
    fillPattern(alloc->hostPtr, kTestSize, 0x5A);
    if (!verifyPattern(alloc->hostPtr, kTestSize, 0x5A)) {
        std::cerr << "FAIL: initial pattern write\n"; ++failures;
    }

    auto offOp = pool->offloadToHost(*alloc);
    if (!offOp) {
        std::cerr << "FAIL: offload\n"; ++failures;
    } else {
        pool->waitMigration(*offOp);
        if (!verifyPattern(alloc->hostPtr, kTestSize, 0x5A)) {
            std::cerr << "FAIL: offload verify (shadow)\n"; ++failures;
        } else {
            std::cout << "  offload OK\n";
            auto relOp = pool->reloadToDevice(*alloc);
            if (!relOp) {
                std::cerr << "FAIL: reload\n"; ++failures;
            } else {
                pool->waitMigration(*relOp);
                if (!verifyPattern(alloc->hostPtr, kTestSize, 0x5A)) {
                    std::cerr << "FAIL: reload verify (device)\n"; ++failures;
                } else {
                    std::cout << "  reload OK\n";
                }
            }
        }
    }

    // ---- Reload without offload guard ----
    auto fresh = pool->allocate(kTestSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (fresh && pool->reloadToDevice(*fresh)) {
        std::cerr << "FAIL: reload non-offloaded alloc should refuse\n"; ++failures;
    }
    if (fresh) pool->deallocate(std::move(*fresh));

    // ---- Cleanup ----
    pool->deallocate(std::move(*alloc));
    pool.reset();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    std::cout << "\n=== " << (failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
              << " (" << failures << " failures) ===\n";
    return failures == 0 ? 0 : 1;
}