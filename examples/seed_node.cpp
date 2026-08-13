#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/network/multi_node_manager.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>

static bool g_running = true;

void signalHandler(int) { g_running = false; }

int main() {
    std::signal(SIGINT, signalHandler);
    std::cout << "=== VulkanVM Seed Node ===\n";
    std::cout << "Listening on 0.0.0.0:51001 for Windows client...\n\n";

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM Seed Node";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> instExts = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
    std::vector<const char*> instExtsAvailable;
    for (const char* e : instExts) {
        uint32_t count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> props(count);
        vkEnumerateInstanceExtensionProperties(nullptr, &count, props.data());
        bool found = false;
        for (auto& p : props) {
            if (std::strcmp(p.extensionName, e) == 0) { found = true; break; }
        }
        if (found) instExtsAvailable.push_back(e);
    }

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    ici.enabledExtensionCount = static_cast<uint32_t>(instExtsAvailable.size());
    ici.ppEnabledExtensionNames = instExtsAvailable.data();

    VkInstance instance;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        std::cerr << "FAIL: create instance\n"; return 1;
    }

    auto devices = vvm::enumerateDevices(instance);
    if (devices.empty()) { std::cerr << "FAIL: no GPU\n"; return 1; }
    auto bestDevice = vvm::selectBestDevice(devices, true, 1024);
    if (!bestDevice) { std::cerr << "FAIL: no suitable GPU\n"; return 1; }
    VkPhysicalDevice physicalDevice = bestDevice->device;
    std::cout << "Selected: " << bestDevice->props.deviceName << "\n";

    auto queues = vvm::findQueueFamilies(physicalDevice);
    uint32_t graphicsFamily = queues.graphics.value_or(0);
    uint32_t computeFamily = queues.compute.value_or(0);
    uint32_t transferFamily = queues.transfer.value_or(0);

    const float prio = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qcis;
    auto addQueue = [&](std::optional<uint32_t> family) {
        if (!family) return;
        VkDeviceQueueCreateInfo q{};
        q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        q.queueFamilyIndex = *family;
        q.queueCount = 1;
        q.pQueuePriorities = &prio;
        qcis.push_back(q);
    };
    addQueue(queues.graphics);
    addQueue(queues.compute);
    addQueue(queues.transfer);

    const char* allDevExts[] = {
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
    };
    uint32_t devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtCount, nullptr);
    std::vector<VkExtensionProperties> devExtProps(devExtCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtCount, devExtProps.data());

    std::vector<const char*> devExts;
    for (const char* ext : allDevExts) {
        for (const auto& p : devExtProps) {
            if (std::strcmp(p.extensionName, ext) == 0) {
                devExts.push_back(ext);
                break;
            }
        }
    }

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
    dci.pQueueCreateInfos = qcis.data();
    dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
    dci.ppEnabledExtensionNames = devExts.data();

    VkPhysicalDeviceFeatures2 physFeats{};
    physFeats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeat{};
    bdaFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bdaFeat.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceTimelineSemaphoreFeatures tsFeat{};
    tsFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    tsFeat.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceVulkan12Features v12Feat{};
    v12Feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12Feat.bufferDeviceAddress = VK_TRUE;
    v12Feat.timelineSemaphore = VK_TRUE;
    bdaFeat.pNext = physFeats.pNext;
    physFeats.pNext = &bdaFeat;
    tsFeat.pNext = physFeats.pNext;
    physFeats.pNext = &tsFeat;
    v12Feat.pNext = physFeats.pNext;
    physFeats.pNext = &v12Feat;
    dci.pNext = &physFeats;

    VkDevice device;
    if (vkCreateDevice(physicalDevice, &dci, nullptr, &device) != VK_SUCCESS) {
        std::cerr << "FAIL: create device\n"; return 1;
    }

    VkQueue graphicsQueue, computeQueue, transferQueue;
    if (queues.graphics) vkGetDeviceQueue(device, *queues.graphics, 0, &graphicsQueue);
    if (queues.compute) vkGetDeviceQueue(device, *queues.compute, 0, &computeQueue);
    if (queues.transfer) vkGetDeviceQueue(device, *queues.transfer, 0, &transferQueue);

    vvm::DeviceConfig devCfg;
    devCfg.physicalDevice = physicalDevice;
    devCfg.device = device;
    devCfg.graphicsQueueFamily = graphicsFamily;
    devCfg.computeQueueFamily = computeFamily;
    devCfg.transferQueueFamily = transferFamily;
    devCfg.graphicsQueue = graphicsQueue;
    devCfg.computeQueue = computeQueue;
    devCfg.transferQueue = transferQueue;

    vvm::PoolConfig poolCfg;
    poolCfg.blockSize = 256 * 1024 * 1024;
    poolCfg.minAlignment = 256;
    poolCfg.enableHostVisible = true;
    poolCfg.enableExternal = true;
    poolCfg.enableDeviceAddress = true;
    poolCfg.maxBlocks = 8;
    poolCfg.maxHeapFraction = 0.0f;
    poolCfg.maxPoolBytes = 0;

    vvm::network::NetworkConfig netCfg;
    netCfg.listenAddress = "0.0.0.0:51001";
    netCfg.seedNodes = {};
    netCfg.enableRdma = false;
    netCfg.enableGpuDirect = false;
    netCfg.enableHostStagedFallback = true;

    auto mgr = vvm::network::MultiNodePoolManager::create({devCfg}, poolCfg, netCfg);
    if (!mgr) { std::cerr << "FAIL: create manager\n"; return 1; }
    if (!mgr->start()) { std::cerr << "FAIL: start\n"; return 1; }
    if (!mgr->registerWithCluster()) { std::cerr << "FAIL: registerWithCluster\n"; return 1; }

    std::cout << "\nSeed node running on 192.168.0.117:51001\n";
    std::cout << "Waiting for Windows client to connect...\n";
    std::cout << "Press Ctrl+C to stop\n\n";

    while (g_running) {
        auto view = mgr->getClusterView();
        std::cout << "\rCluster nodes: " << view.size() << " (seed + " << (view.size() > 1 ? view.size() - 1 : 0) << " clients)" << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    std::cout << "\nShutting down...\n";
    mgr->stop();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
