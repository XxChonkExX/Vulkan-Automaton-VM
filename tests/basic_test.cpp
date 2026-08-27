#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"
#include "vulkan_vm/buddy_allocator.hpp"

#include <iostream>
#include <vector>
#include <cassert>

using namespace vvm;

using namespace vvm;

void testBuddyAllocator() {
    std::cout << "Testing BuddyAllocator..." << std::endl;
    
    // Test basic allocation/deallocation
    BuddyAllocator allocator(256 * 1024 * 1024, 256 * 1024); // 256MB block, 256KB min
    
    // Allocate 64MB
    auto offset1 = allocator.allocate(64 * 1024 * 1024);
    if (!offset1.has_value() || *offset1 != 0) {
        std::cerr << "FAIL: First allocation should be at offset 0, got " << (offset1.has_value() ? std::to_string(*offset1) : "none") << std::endl;
        return;
    }
    std::cout << "  Allocated 64MB at offset " << *offset1 << std::endl;
    
    // Allocate 128MB - should be at offset 128MB (right half of block)
    // because left 128MB is partially used by first allocation
    auto offset2 = allocator.allocate(128 * 1024 * 1024);
    if (!offset2.has_value() || *offset2 != 128 * 1024 * 1024) {
        std::cerr << "FAIL: Second allocation should be at offset 128MB, got " << (offset2.has_value() ? std::to_string(*offset2) : "none") << std::endl;
        return;
    }
    std::cout << "  Allocated 128MB at offset " << *offset2 << std::endl;
    
    // Allocate 32MB - should fit in the free 64MB slot at offset 64MB
    auto offset3 = allocator.allocate(32 * 1024 * 1024);
    if (!offset3.has_value() || *offset3 != 64 * 1024 * 1024) {
        std::cerr << "FAIL: Third allocation should be at offset 64MB, got " << (offset3.has_value() ? std::to_string(*offset3) : "none") << std::endl;
        return;
    }
    std::cout << "  Allocated 32MB at offset " << *offset3 << std::endl;
    
    // Check largest free
    std::cout << "  Largest free: " << allocator.getLargestFree() / (1024*1024) << " MB" << std::endl;
    
    // Deallocate middle (the 128MB allocation)
    allocator.deallocate(*offset2, 128 * 1024 * 1024);
    std::cout << "  Deallocated 128MB" << std::endl;
    
    // Check largest free after merge - should be 128MB again
    std::cout << "  Largest free after dealloc: " << allocator.getLargestFree() / (1024*1024) << " MB" << std::endl;
    
    // Allocate again - should get the 128MB slot back
    auto offset4 = allocator.allocate(128 * 1024 * 1024);
    if (!offset4.has_value() || *offset4 != 128 * 1024 * 1024) {
        std::cerr << "FAIL: Fourth allocation should be at offset 128MB, got " << (offset4.has_value() ? std::to_string(*offset4) : "none") << std::endl;
        return;
    }
    std::cout << "  Allocated 128MB at offset " << *offset4 << std::endl;
    
    std::cout << "BuddyAllocator test passed!" << std::endl;
}

int main() {
    testBuddyAllocator();
    
    // ... rest of main
    
    // Create Vulkan instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;
    
    // Filter against what the platform loader actually provides - Android
    // loaders may lack VK_EXT_debug_utils (VK_ERROR_EXTENSION_NOT_PRESENT
    // otherwise kills instance creation).
    std::vector<const char*> wanted = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };
    uint32_t availCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &availCount, nullptr);
    std::vector<VkExtensionProperties> avail(availCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &availCount, avail.data());
    std::vector<const char*> extensions;
    for (const char* w : wanted) {
        for (const auto& e : avail) {
            if (!std::strcmp(e.extensionName, w)) { extensions.push_back(w); break; }
        }
    }
    
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
    assert(!devices.empty());
    
    std::cout << "Found " << devices.size() << " device(s):\n";
    for (const auto& dev : devices) {
        std::cout << "  " << dev.props.deviceName 
                  << " (vendor: 0x" << std::hex << dev.vendorID << std::dec
                  << ", type: " << (dev.discrete ? "discrete" : "integrated") << ")\n";
        
        printDeviceProperties(dev.device);
        printMemoryTypes(dev.memProps);
        printQueueFamilies(dev.device);
    }
    
    // Select best device
    auto bestDevice = selectBestDevice(devices, true, 1024);  // At least 1GB
    assert(bestDevice.has_value());
    
    std::cout << "\nSelected: " << bestDevice->props.deviceName << "\n";
    
    // Check required extensions
    // Critical for correctness; VK_EXT_memory_budget is OPTIONAL - the pool
    // probes it at runtime and degrades gracefully (Android 16 Adreno drivers
    // often omit it).
    std::vector<const char*> requiredExts = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#if defined(__ANDROID__)
        VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
#elif defined(VVM_PLATFORM_LINUX)
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,  // Linux
#elif defined(VVM_PLATFORM_WINDOWS)
        "VK_KHR_external_memory_win32",  // Windows
#endif
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    };
    std::vector<const char*> optionalExts = {
        VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
    };

    bool extsSupported =
        checkDeviceExtensionSupport(bestDevice->device, requiredExts);
    const bool budgetAvailable =
        checkDeviceExtensionSupport(bestDevice->device, optionalExts);

    // Capability gate (see minimal_test): Adreno 610-class drivers lack the
    // required extensions/features; a Release assert() would let a null device
    // sail into vkGetDeviceQueue (SEGV) instead of skipping cleanly.
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

    std::cout << "Required extensions supported: YES\n";

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
            std::cout << "  Queue " << name << ": family " << *family << "\n";
        }
    };
    
    addQueue(queues.graphics, "graphics");
    addQueue(queues.compute, "compute");
    addQueue(queues.transfer, "transfer");
    
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    
    VkPhysicalDeviceBufferDeviceAddressFeatures addrFeatures{};
    addrFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    addrFeatures.bufferDeviceAddress = VK_TRUE;
    addrFeatures.pNext = features2.pNext;
    features2.pNext = &addrFeatures;
    
    // Request only via the aggregate Vulkan12 struct (VUID-pNext-02830).
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
    std::vector<const char*> enableExts(requiredExts);
    if (budgetAvailable)
        enableExts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(enableExts.size());
    deviceInfo.ppEnabledExtensionNames = enableExts.data();
    
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
    poolConfig.blockSize = 256 * 1024 * 1024;  // 256MB blocks
    poolConfig.minAlignment = 256 * 1024;       // 256KB
    poolConfig.enableHostVisible = true;
    poolConfig.enableExternal = true;
    poolConfig.enableDeviceAddress = true;
    poolConfig.maxBlocks = 8;
    
    // Create pool
    auto pool = UnifiedMemoryPool::create(devConfig, poolConfig);
    assert(pool.has_value());
    
    std::cout << "\nPool created successfully\n";
    
    // Test allocations
    std::cout << "\nTesting allocations...\n";
    
    // Allocate tensor buffers
    auto alloc1 = pool->allocateTensor(64 * 1024 * 1024);  // 64MB
    assert(alloc1.has_value());
    std::cout << "  Alloc 1: 64MB at offset " << alloc1->offset 
              << ", device address: 0x" << std::hex << alloc1->deviceAddress << std::dec << "\n";
    
    auto alloc2 = pool->allocateTensor(128 * 1024 * 1024);  // 128MB
    assert(alloc2.has_value());
    std::cout << "  Alloc 2: 128MB at offset " << alloc2->offset << "\n";
    
    auto alloc3 = pool->allocateTensor(32 * 1024 * 1024);  // 32MB
    assert(alloc3.has_value());
    std::cout << "  Alloc 3: 32MB at offset " << alloc3->offset << "\n";
    
    // Check stats
    auto stats = pool->getStats();
    std::cout << "\nPool Stats:\n";
    std::cout << "  Total allocated: " << stats.totalAllocated / (1024*1024) << " MB\n";
    std::cout << "  Total used: " << stats.totalUsed / (1024*1024) << " MB\n";
    std::cout << "  Total free: " << stats.totalFree / (1024*1024) << " MB\n";
    std::cout << "  Fragmentation: " << (stats.fragmentationRatio * 100) << "%\n";
    std::cout << "  Blocks: " << stats.blockCount << "\n";
    
    // Test external memory export
    if (poolConfig.enableExternal) {
        std::cout << "\nTesting external memory export...\n";
        auto exportInfo = pool->exportMemory(*alloc1, ExternalHandleType::OpaqueFd);
        if (exportInfo) {
            std::cout << "  Export successful\n";
            // RAII closes the exported handle when exportInfo goes out of scope.
        } else {
            std::cout << "  Export not supported on this device\n";
        }
    }
    
    // Test deallocation
    std::cout << "\nTesting deallocation...\n";
    pool->deallocate(std::move(*alloc2));
    
    stats = pool->getStats();
    std::cout << "  After dealloc: used=" << stats.totalUsed / (1024*1024) 
              << " MB, free=" << stats.totalFree / (1024*1024) << " MB\n";
    
    // Cleanup
    pool->deallocate(std::move(*alloc1));
    pool->deallocate(std::move(*alloc3));
    
    // Destroy pool before device (pool owns Vulkan objects tied to the device)
    pool.reset();
    
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    
    std::cout << "\nAll tests passed!\n";
    return 0;
}