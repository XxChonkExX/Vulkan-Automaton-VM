#include <vulkan_vm/vulkan_vm.hpp>
#include <android/hardware_buffer.h>
#include <android/log.h>
#include <cstdio>
#include <vector>
#include <set>

#define LOG_TAG "VulkanVM_AHB_Test"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static int test_ahardwarebuffer() {
    LOGI("Starting AHardwareBuffer test");
    
    // Create AHardwareBuffer
    AHardwareBuffer_Desc desc{};
    desc.width = 256;
    desc.height = 256;
    desc.layers = 1;
    desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    desc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | 
                 AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT;
    desc.stride = 0;
    
    AHardwareBuffer* hardwareBuffer = nullptr;
    int result = AHardwareBuffer_allocate(&desc, &hardwareBuffer);
    if (result != 0 || !hardwareBuffer) {
        LOGE("Failed to allocate AHardwareBuffer: %d", result);
        return -1;
    }
    LOGI("AHardwareBuffer allocated: %p", hardwareBuffer);
    
    // Create Vulkan instance
    LOGI("Creating Vulkan instance...");
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM_AHB_Test";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    
    VkInstance instance;
    VkResult vkResult = vkCreateInstance(&instanceInfo, nullptr, &instance);
    if (vkResult != VK_SUCCESS) {
        LOGE("Failed to create Vulkan instance: %d", vkResult);
        AHardwareBuffer_release(hardwareBuffer);
        return -1;
    }
    LOGI("Vulkan instance created");
    
    // Enumerate physical devices
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        LOGE("No Vulkan physical devices found");
        vkDestroyInstance(instance, nullptr);
        AHardwareBuffer_release(hardwareBuffer);
        return -1;
    }
    
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    VkPhysicalDevice physicalDevice = devices[0];
    LOGI("Found %u physical device(s)", deviceCount);
    
    // Find queue families
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    
    uint32_t graphicsQueueFamily = UINT32_MAX;
    uint32_t computeQueueFamily = UINT32_MAX;
    uint32_t transferQueueFamily = UINT32_MAX;
    
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) graphicsQueueFamily = i;
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) computeQueueFamily = i;
        if (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) transferQueueFamily = i;
    }
    
    if (graphicsQueueFamily == UINT32_MAX) graphicsQueueFamily = 0;
    if (computeQueueFamily == UINT32_MAX) computeQueueFamily = graphicsQueueFamily;
    if (transferQueueFamily == UINT32_MAX) transferQueueFamily = graphicsQueueFamily;
    
    // Create logical device
    float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {graphicsQueueFamily, computeQueueFamily, transferQueueFamily};
    
    for (uint32_t qf : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = qf;
        qci.queueCount = 1;
        qci.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(qci);
    }
    
    VkPhysicalDeviceFeatures deviceFeatures{};
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    deviceInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceInfo.pEnabledFeatures = &deviceFeatures;
    
    // Enable external memory extensions
    const char* deviceExtensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME
    };
    deviceInfo.enabledExtensionCount = 2;
    deviceInfo.ppEnabledExtensionNames = deviceExtensions;
    
    VkDevice device;
    vkResult = vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device);
    if (vkResult != VK_SUCCESS) {
        LOGE("Failed to create Vulkan device: %d", vkResult);
        vkDestroyInstance(instance, nullptr);
        AHardwareBuffer_release(hardwareBuffer);
        return -1;
    }
    LOGI("Vulkan device created");
    
    // Get queues
    VkQueue graphicsQueue, computeQueue, transferQueue;
    vkGetDeviceQueue(device, graphicsQueueFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);
    vkGetDeviceQueue(device, transferQueueFamily, 0, &transferQueue);
    
    // Create VulkanVM pool
    vvm::DeviceConfig devConfig{};
    devConfig.physicalDevice = physicalDevice;
    devConfig.device = device;
    devConfig.graphicsQueueFamily = graphicsQueueFamily;
    devConfig.computeQueueFamily = computeQueueFamily;
    devConfig.transferQueueFamily = transferQueueFamily;
    devConfig.graphicsQueue = graphicsQueue;
    devConfig.computeQueue = computeQueue;
    devConfig.transferQueue = transferQueue;
    
    vvm::PoolConfig poolConfig = vvm::PoolConfig::forDevice(physicalDevice);
    poolConfig.maxHeapFraction = 0.5f;
    
    auto pool = vvm::UnifiedMemoryPool::create(devConfig, poolConfig);
    if (!pool) {
        LOGE("Failed to create UnifiedMemoryPool");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        AHardwareBuffer_release(hardwareBuffer);
        return -1;
    }
    LOGI("UnifiedMemoryPool created");
    LOGI("About to import AHardwareBuffer...");
    vvm::ExternalMemoryInfo extInfo{};
    extInfo.type = vvm::ExternalHandleType::AndroidHardwareBuffer;
    extInfo.handle = vvm::ExternalHandle(hardwareBuffer);
    extInfo.size = 256 * 256 * 4; // RGBA8
    extInfo.dedicatedAllocation = true;
    
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    LOGI("Calling importMemory...");
    auto allocation = pool->importMemory(std::move(extInfo), usage);
    LOGI("importMemory returned");
    if (!allocation) {
        LOGE("Failed to import AHardwareBuffer");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        AHardwareBuffer_release(hardwareBuffer);
        return -1;
    }
    LOGI("AHardwareBuffer imported successfully!");
    LOGI("  Allocation size: %llu bytes", allocation->size);
    LOGI("  Device address: 0x%llx", allocation->deviceAddress);
    
    // Test using in shader (get device address)
    if (allocation->deviceAddress != 0) {
        LOGI("Device address valid - can use in shaders");
    }
    
    // Cleanup
    pool.reset();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    AHardwareBuffer_release(hardwareBuffer);
    
    LOGI("AHardwareBuffer test PASSED");
    return 0;
}

extern "C" int main() {
    LOGI("=== TEST START ===");
    return test_ahardwarebuffer();
}