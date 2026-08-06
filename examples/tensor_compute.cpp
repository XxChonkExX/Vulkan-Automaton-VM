/*
 * VulkanVM - Tensor Compute Example
 * 
 * Demonstrates: persistent allocation, bindless access, cross-GPU sharing
 */

#include <vulkan_vm/vulkan_vm.hpp>
#include <vulkan_vm/utils.hpp>

#include <iostream>
#include <vector>
#include <chrono>
#include <set>

using namespace vvm;

int main() {
    std::cout << "VulkanVM Tensor Compute Example\n";
    
    // 1. Create Vulkan instance with required extensions
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM Tensor Example";
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
    
    VkInstance instance;
    vkCreateInstance(&instanceInfo, nullptr, &instance);
    
    // 2. Find best device (prefer discrete, min 2GB VRAM)
    auto devices = enumerateDevices(instance);
    auto bestDevice = selectBestDevice(devices, true, 2048);
    
    if (!bestDevice) {
        std::cerr << "No suitable device found\n";
        return 1;
    }
    
    std::cout << "Using: " << bestDevice->props.deviceName << "\n";
    
    // 3. Check required device extensions
    std::vector<const char*> deviceExts = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
    #ifdef VVM_PLATFORM_LINUX
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    #elif defined(VVM_PLATFORM_WINDOWS)
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    #endif
    };
    
    if (!checkDeviceExtensionSupport(bestDevice->device, deviceExts)) {
        std::cerr << "Required extensions not supported\n";
        return 1;
    }
    
    // 4. Create logical device with features
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
    
    // Enable required features
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
    
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    deviceInfo.pNext = &features2;
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExts.size());
    deviceInfo.ppEnabledExtensionNames = deviceExts.data();
    
    VkDevice device;
    vkCreateDevice(bestDevice->device, &deviceInfo, nullptr, &device);
    
    // Get queues
    VkQueue computeQueue, transferQueue;
    vkGetDeviceQueue(device, queues.compute.value(), 0, &computeQueue);
    vkGetDeviceQueue(device, queues.transfer.value_or(queues.compute.value()), 0, &transferQueue);
    
    // 5. Configure and create memory pool
    DeviceConfig devConfig;
    devConfig.physicalDevice = bestDevice->device;
    devConfig.device = device;
    devConfig.computeQueueFamily = queues.compute.value();
    devConfig.transferQueueFamily = queues.transfer.value_or(queues.compute.value());
    devConfig.computeQueue = computeQueue;
    devConfig.transferQueue = transferQueue;
    
    PoolConfig poolConfig;
    poolConfig.blockSize = 512 * 1024 * 1024;  // 512MB blocks
    poolConfig.minAlignment = 256 * 1024;       // Tensor core alignment
    poolConfig.enableDeviceAddress = true;
    poolConfig.enableExternal = true;
    poolConfig.maxBlocks = 8;
    
    auto pool = UnifiedMemoryPool::create(devConfig, poolConfig);
    if (!pool) {
        std::cerr << "Failed to create memory pool\n";
        return 1;
    }
    
    std::cout << "Memory pool created\n";
    
    // 6. Allocate tensor buffers (persistent, bindless-ready)
    struct TensorBuffer {
        Allocation alloc;
        VkDeviceSize size;
        std::string name;
    };
    
    std::vector<TensorBuffer> tensors;
    
    // Allocate various sizes (typical for ML workloads)
    auto allocA = pool->allocateTensor(64 * 1024 * 1024);   // 64MB - weight matrix
    auto allocB = pool->allocateTensor(128 * 1024 * 1024);  // 128MB - activation buffer
    auto allocC = pool->allocateTensor(32 * 1024 * 1024);   // 32MB - output buffer
    auto allocD = pool->allocateTensor(16 * 1024 * 1024);   // 16MB - temporary
    
    if (allocA && allocB && allocC && allocD) {
        tensors.push_back({std::move(*allocA), 64 * 1024 * 1024, "weights"});
        tensors.push_back({std::move(*allocB), 128 * 1024 * 1024, "activations"});
        tensors.push_back({std::move(*allocC), 32 * 1024 * 1024, "output"});
        tensors.push_back({std::move(*allocD), 16 * 1024 * 1024, "temp"});
        
        std::cout << "\nAllocated tensors:\n";
        for (const auto& t : tensors) {
            std::cout << "  " << t.name << ": " << (t.size / (1024*1024)) << " MB"
                      << " @ 0x" << std::hex << t.alloc.deviceAddress << std::dec << "\n";
        }
    }
    
    // 7. Simulate compute workload (bindless access in shader)
    // In real usage, you'd pass device addresses to shader via push constants or descriptor sets
    
    // Example: Fill buffer via transfer queue
    VkCommandPool cmdPool;
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = devConfig.transferQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool);
    
    VkCommandBuffer cmd = beginSingleTimeCommands(device, cmdPool);
    
    // Fill weights buffer with pattern
    uint32_t pattern = 0x42424242;
    vkCmdFillBuffer(cmd, tensors[0].alloc.buffer, 0, tensors[0].size, pattern);
    
    // Copy weights to activations (simulate layer input)
    VkBufferCopy copy{};
    copy.size = std::min(tensors[0].size, tensors[1].size);
    vkCmdCopyBuffer(cmd, tensors[0].alloc.buffer, tensors[1].alloc.buffer, 1, &copy);
    
    endSingleTimeCommands(device, cmdPool, cmd, transferQueue);
    
    // 8. Check pool stats
    auto stats = pool->getStats();
    std::cout << "\nPool Statistics:\n";
    std::cout << "  Total: " << (stats.totalAllocated / (1024*1024)) << " MB\n";
    std::cout << "  Used:  " << (stats.totalUsed / (1024*1024)) << " MB\n";
    std::cout << "  Free:  " << (stats.totalFree / (1024*1024)) << " MB\n";
    std::cout << "  Frag:  " << (stats.fragmentationRatio * 100) << "%\n";
    
    // 9. Demonstrate external memory export (for multi-GPU)
    if (poolConfig.enableExternal) {
        std::cout << "\nExporting memory for cross-GPU sharing...\n";
        auto exportInfo = pool->exportMemory(tensors[0].alloc, ExternalHandleType::OpaqueFd);
        if (exportInfo) {
            std::cout << "  Exported: type=" << static_cast<int>(exportInfo->type)
                      << ", size=" << (exportInfo->size / (1024*1024)) << " MB\n";
            // RAII closes the exported handle when exportInfo goes out of scope.
        } else {
            std::cout << "  Export not supported on this device\n";
        }
    }
    
    // 10. Demonstrate offload (swap to host)
    try {
        OffloadConfig offloadConfig;
        offloadConfig.hostShadowSize = 2ull * 1024 * 1024 * 1024;  // 2GB
        offloadConfig.useMadvise = true;
        offloadConfig.useMprotect = true;
        offloadConfig.transferQueue = transferQueue;
        offloadConfig.transferQueueFamily = queues.transfer.value_or(queues.compute.value());
        
        VVM_LOG_INFO("Creating OffloadManager with transferQueue=%p, queueFamily=%u", 
                     transferQueue, queues.transfer.value_or(queues.compute.value()));
        
        OffloadManager offloadManager(&*pool, offloadConfig);
        
        std::cout << "\nOffloading temp buffer to host...\n";
        VVM_LOG_INFO("Calling offloadSync on tensor[3] (size=%llu MB)", tensors[3].size / (1024*1024));
        bool offloaded = offloadManager.offloadSync(tensors[3].alloc);
        if (offloaded) {
            std::cout << "  Offloaded " << (tensors[3].size / (1024*1024)) << " MB to host\n";
            
            // Reload back to device
            bool reloaded = offloadManager.reloadSync(tensors[3].alloc);
            if (reloaded) {
                std::cout << "  Reloaded back to device\n";
            }
        } else {
            std::cout << "  Offload not supported or failed\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Offload demo failed: " << e.what() << "\n";
    }
    
    // 11. Cleanup
    for (auto& t : tensors) {
        pool->deallocate(std::move(t.alloc));
    }
    
    // Destroy pool before device (pool owns Vulkan objects tied to the device)
    pool.reset();
    
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    
    std::cout << "\nExample completed successfully!\n";
    return 0;
}