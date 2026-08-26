#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"
#include <iostream>
#include <cassert>

using namespace vvm;

int main() {
    std::cout << "Testing minimal pool creation..." << std::endl;
    
    // Create Vulkan instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Minimal Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;
    
    std::vector<const char*> extensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };
    
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instanceInfo.ppEnabledExtensionNames = extensions.data();
    
    VkInstance instance;
    VkResult result = vkCreateInstance(&instanceInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        std::cerr << "vkCreateInstance failed: VkResult=" << result << "\n";
        return 1;
    }

    // Enumerate devices
    auto devices = enumerateDevices(instance);
    if (devices.empty()) {
        std::cerr << "No Vulkan devices found\n";
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

    auto bestDevice = selectBestDevice(devices, true, 1024);
    if (!bestDevice) {
        std::cout << "SKIP: no device meets the minimum heap requirement\n";
        vkDestroyInstance(instance, nullptr);
        return 0;
    }
    
    // Check required extensions
    std::vector<const char*> requiredExts = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#if defined(VVM_PLATFORM_WINDOWS)
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#elif defined(__ANDROID__)
        VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
#endif
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    };
    
    bool extsSupported = checkDeviceExtensionSupport(bestDevice->device, requiredExts);
    std::cout << "Required extensions supported: " << (extsSupported ? "YES" : "NO") << "\n";

    // Capability gate: query what THIS device supports before requesting.
    // Adreno 610-class drivers lack Vulkan 1.2 bufferDeviceAddress - creating
    // a device that requests unsupported features fails, and a Release-build
    // assert() would let the null device sail into vkGetDeviceQueue (SEGV).
    {
        VkPhysicalDeviceVulkan12Features supported12{};
        supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceFeatures2 suppFeat{};
        suppFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        suppFeat.pNext = &supported12;
        vkGetPhysicalDeviceFeatures2(bestDevice->device, &suppFeat);

        if (!extsSupported || !supported12.bufferDeviceAddress ||
            !supported12.timelineSemaphore) {
            std::cout << "SKIP: " << bestDevice->props.deviceName
                      << " does not meet requirements (exts=" << extsSupported
                      << " BDA=" << supported12.bufferDeviceAddress
                      << " TS=" << supported12.timelineSemaphore << ")\n";
            vkDestroyInstance(instance, nullptr);
            return 0;
        }
    }
    
    // Create logical device
    auto queues = findQueueFamilies(bestDevice->device);
    
    float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    
    auto addQueue = [&](std::optional<uint32_t> family, const char* name) {
        if (family) {
            VkDeviceQueueCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            info.queueFamilyIndex = *family;
            info.queueCount = 1;
            info.pQueuePriorities = &queuePriority;
            queueInfos.push_back(info);
        }
    };
    
    addQueue(queues.graphics, "graphics");
    addQueue(queues.compute, "compute");
    addQueue(queues.transfer, "transfer");
    
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    
    // Request only via the aggregate Vulkan12 struct: chaining the individual
    // BufferDeviceAddress/TimelineSemaphore structs alongside it is illegal
    // (VUID-VkDeviceCreateInfo-pNext-02830).
    VkPhysicalDeviceVulkan12Features v12Features{};
    v12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12Features.bufferDeviceAddress = VK_TRUE;
    v12Features.timelineSemaphore = VK_TRUE;
    features2.pNext = &v12Features;
    
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    deviceInfo.pNext = &features2;
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExts.size());
    deviceInfo.ppEnabledExtensionNames = requiredExts.data();
    
    VkDevice device;
    result = vkCreateDevice(bestDevice->device, &deviceInfo, nullptr, &device);
    if (result != VK_SUCCESS) {
        std::cerr << "vkCreateDevice failed: VkResult=" << result << "\n";
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    
    // Get queues
    VkQueue graphicsQueue, computeQueue, transferQueue;
    if (queues.graphics) vkGetDeviceQueue(device, *queues.graphics, 0, &graphicsQueue);
    if (queues.compute) vkGetDeviceQueue(device, *queues.compute, 0, &computeQueue);
    if (queues.transfer) vkGetDeviceQueue(device, *queues.transfer, 0, &transferQueue);
    
    // Configure pool
    DeviceConfig devConfig;
    devConfig.physicalDevice = bestDevice->device;
    devConfig.device = device;
    devConfig.graphicsQueueFamily = queues.graphics.value_or(0);
    devConfig.computeQueueFamily = queues.compute.value_or(0);
    devConfig.transferQueueFamily = queues.transfer.value_or(0);
    devConfig.graphicsQueue = graphicsQueue;
    devConfig.computeQueue = computeQueue;
    devConfig.transferQueue = transferQueue;
    
    PoolConfig poolConfig;
    poolConfig.blockSize = 256 * 1024 * 1024;
    poolConfig.minAlignment = 256 * 1024;
    poolConfig.enableHostVisible = true;
    poolConfig.enableExternal = true;
    poolConfig.enableDeviceAddress = true;
    poolConfig.maxBlocks = 8;
    
    std::cout << "Creating pool..." << std::endl;
    
    // Create pool
    auto pool = UnifiedMemoryPool::create(devConfig, poolConfig);
    std::cout << "Pool created: " << (pool.has_value() ? "YES" : "NO") << std::endl;
    if (!pool.has_value()) {
        std::cerr << "Failed to create pool!" << std::endl;
        return 1;
    }
    
    std::cout << "Pool created successfully!" << std::endl;
    
    // Test allocation
    auto alloc1 = pool->allocateTensor(64 * 1024 * 1024);
    std::cout << "Alloc 1: " << (alloc1.has_value() ? "YES" : "NO") << std::endl;
    
    std::cout << "Testing allocation 2..." << std::flush;
    auto alloc2 = pool->allocateTensor(128 * 1024 * 1024);
    std::cout << "Alloc 2: " << (alloc2.has_value() ? "YES" : "NO") << std::endl;
    
    std::cout << "Testing allocation 3..." << std::flush;
    auto alloc3 = pool->allocateTensor(32 * 1024 * 1024);
    std::cout << "Alloc 3: " << (alloc3.has_value() ? "YES" : "NO") << std::endl;
    
    // Check stats
    auto stats = pool->getStats();
    std::cout << "Pool Stats:" << std::endl;
    std::cout << "  Total allocated: " << stats.totalAllocated / (1024*1024) << " MB" << std::endl;
    std::cout << "  Total used: " << stats.totalUsed / (1024*1024) << " MB" << std::endl;
    std::cout << "  Total free: " << stats.totalFree / (1024*1024) << " MB" << std::endl;
    std::cout << "  Fragmentation: " << (stats.fragmentationRatio * 100) << "%" << std::endl;
    std::cout << "  Blocks: " << stats.blockCount << std::endl;
    
    // Test external memory export
    if (poolConfig.enableExternal) {
        std::cout << "Testing external memory export..." << std::endl;
        auto exportInfo = pool->exportMemory(*alloc1, ExternalHandleType::OpaqueFd);
        if (exportInfo) {
            std::cout << "  Export successful" << std::endl;
            // RAII closes the exported handle when exportInfo goes out of scope.
        } else {
            std::cout << "  Export not supported on this device" << std::endl;
        }
    }
    
    // Test deallocation
    pool->deallocate(std::move(*alloc2));
    
    stats = pool->getStats();
    std::cout << "After dealloc: used=" << stats.totalUsed / (1024*1024) 
              << " MB, free=" << stats.totalFree / (1024*1024) << " MB" << std::endl;
    
    // Cleanup
    pool->deallocate(std::move(*alloc1));
    pool->deallocate(std::move(*alloc3));
    
    pool.reset();
    
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}