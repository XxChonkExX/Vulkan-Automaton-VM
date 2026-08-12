/*
 * VulkanVM Cross-Machine Tensor Network Test - SERVER (Evo-X2 Linux)
 * 
 * Run on the Evo-X2 (AMD Strix Halo / Radeon 890M) as the GPU server.
 * Usage: ./tensor_server_test --port 51000
 */
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/tensor_transport.hpp"
#include "vulkan_vm/utils.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cstring>
#include <csignal>

using namespace vvm;
using namespace vvm::tensor;

static std::atomic<bool> g_running{true};

static void fillPattern(void* buf, VkDeviceSize size, uint8_t val) {
    std::memset(buf, val, static_cast<size_t>(size));
}

static bool verifyPattern(const void* buf, VkDeviceSize size, uint8_t expected) {
    const auto* p = static_cast<const uint8_t*>(buf);
    for (VkDeviceSize i = 0; i < size; ++i) {
        if (p[i] != expected) return false;
    }
    return true;
}

struct TestDevice {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t graphicsFamily = UINT32_MAX;
    uint32_t computeFamily  = UINT32_MAX;
    uint32_t transferFamily = UINT32_MAX;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue computeQueue  = VK_NULL_HANDLE;
    VkQueue transferQueue = VK_NULL_HANDLE;
};

static TestDevice s_dev;

static bool initTestDevice() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM Tensor Server";
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

    if (vkCreateInstance(&instanceInfo, nullptr, &s_dev.instance) != VK_SUCCESS) {
        std::cerr << "FAIL: create instance\n"; return false;
    }

    auto devices = enumerateDevices(s_dev.instance);
    if (devices.empty()) { std::cerr << "FAIL: no GPU\n"; return false; }
    // On AMD Strix Halo, prefer the integrated GPU
    auto bestDevice = selectBestDevice(devices, false, 1024);
    if (!bestDevice) { std::cerr << "FAIL: no suitable GPU\n"; return false; }
    s_dev.physicalDevice = bestDevice->device;
    std::cout << "Server GPU: " << bestDevice->props.deviceName << "\n";

    auto queues = findQueueFamilies(s_dev.physicalDevice);
    s_dev.graphicsFamily = queues.graphics.value_or(0);
    s_dev.computeFamily = queues.compute.value_or(0);
    s_dev.transferFamily = queues.transfer.value_or(0);
    if (s_dev.graphicsFamily == UINT32_MAX || s_dev.transferFamily == UINT32_MAX) {
        std::cerr << "FAIL: missing required queues\n"; return false;
    }

    const float prio = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qcis;
    auto addQueue = [&](std::optional<uint32_t> fam) {
        if (!fam) return;
        VkDeviceQueueCreateInfo q{};
        q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        q.queueFamilyIndex = *fam;
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
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,  // Linux: DMA-BUF/OPAQUE_FD
        VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
    };
    std::vector<const char*> devExts;
    uint32_t devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(s_dev.physicalDevice, nullptr, &devExtCount, nullptr);
    std::vector<VkExtensionProperties> devExtProps(devExtCount);
    vkEnumerateDeviceExtensionProperties(s_dev.physicalDevice, nullptr, &devExtCount, devExtProps.data());
    for (const char* ext : allDevExts) {
        for (const auto& p : devExtProps) {
            if (std::strcmp(p.extensionName, ext) == 0) {
                devExts.push_back(ext);
                break;
            }
        }
    }

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

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
    dci.pQueueCreateInfos = qcis.data();
    dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
    dci.ppEnabledExtensionNames = devExts.data();
    dci.pNext = &physFeats;

    if (vkCreateDevice(s_dev.physicalDevice, &dci, nullptr, &s_dev.device) != VK_SUCCESS) {
        std::cerr << "FAIL: create device\n"; return false;
    }

    vkGetDeviceQueue(s_dev.device, s_dev.graphicsFamily, 0, &s_dev.graphicsQueue);
    vkGetDeviceQueue(s_dev.device, s_dev.computeFamily, 0, &s_dev.computeQueue);
    vkGetDeviceQueue(s_dev.device, s_dev.transferFamily, 0, &s_dev.transferQueue);
    return true;
}

static void destroyTestDevice() {
    if (s_dev.device) vkDestroyDevice(s_dev.device, nullptr);
    if (s_dev.instance) vkDestroyInstance(s_dev.instance, nullptr);
}

static DeviceConfig makeDevConfig() {
    DeviceConfig cfg;
    cfg.physicalDevice = s_dev.physicalDevice;
    cfg.device = s_dev.device;
    cfg.graphicsQueueFamily = s_dev.graphicsFamily;
    cfg.computeQueueFamily = s_dev.computeFamily;
    cfg.transferQueueFamily = s_dev.transferFamily;
    cfg.graphicsQueue = s_dev.graphicsQueue;
    cfg.computeQueue = s_dev.computeQueue;
    cfg.transferQueue = s_dev.transferQueue;
    return cfg;
}

static PoolConfig makePoolConfig() {
    // Use APU-appropriate config (Strix Halo is unified memory)
    PoolConfig cfg = PoolConfig::forDevice(s_dev.physicalDevice);
    // Add extra external memory support for cross-machine
    cfg.enableExternal = true;
    cfg.enableHostVisible = true;
    cfg.enableDeviceAddress = true;
    return cfg;
}

static bool deviceCopy(VkDevice device, VkQueue queue, VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandPool pool;
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpci.queueFamilyIndex = s_dev.transferFamily;
    if (vkCreateCommandPool(device, &cpci, nullptr, &pool) != VK_SUCCESS) return false;

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &cbai, &cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(device, pool, nullptr);
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkBufferCopy copy{};
    copy.size = size;
    vkBeginCommandBuffer(cmd, &beginInfo);
    vkCmdCopyBuffer(cmd, src, dst, 1, &copy);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, pool, 1, &cmd);
        vkDestroyCommandPool(device, pool, nullptr);
        return false;
    }

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    bool ok = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS;
    if (ok) ok = vkQueueWaitIdle(queue) == VK_SUCCESS;

    vkFreeCommandBuffers(device, pool, 1, &cmd);
    vkDestroyCommandPool(device, pool, nullptr);
    return ok;
}

int main(int argc, char** argv) {
    uint16_t port = 51000;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--port <port>] (default 51000)\n";
            return 0;
        }
    }

    std::cout << "=== VulkanVM Tensor Server (Evo-X2) ===\n";
    std::cout << "Listening on port " << port << "\n\n";

    if (!initTestDevice()) {
        std::cerr << "FAILED to initialize Vulkan device\n";
        return 1;
    }

    auto devCfg = makeDevConfig();
    auto poolCfg = makePoolConfig();

    TransportConfig cfg;
    cfg.preference = TransportConfig::Preference::NetworkOnly;
    cfg.networkPort = port;
    cfg.enableAsyncPipeline = true;
    cfg.maxInFlightTransfers = 4;

    auto transport = vvm::tensor::Transport::create(cfg, {devCfg}, poolCfg);
    if (!transport) {
        std::cerr << "FAIL: create transport\n";
        destroyTestDevice();
        return 1;
    }

    if (!transport->initialize()) {
        std::cerr << "FAIL: initialize transport\n";
        destroyTestDevice();
        return 1;
    }

    // Server listens - join cluster with itself as bootstrap
    std::string nodeId = transport->getLocalNodeId();
    std::cout << "Local Node ID: " << nodeId << "\n";
    
    if (!transport->joinCluster(nodeId)) {
        std::cerr << "FAIL: join cluster\n";
        destroyTestDevice();
        return 1;
    }

    std::cout << "Server ready. Waiting for client...\n";
    std::cout << "Press Ctrl+C to exit.\n\n";

    // Allocate a server tensor and announce it for clients to pull
    const VkDeviceSize kBytes = 16ull * 1024 * 1024;
    vvm::tensor::TensorMetadata serverMeta;
    serverMeta.dtype = vvm::tensor::DataType::Float32;
    serverMeta.name = "server_to_client_tensor";
    serverMeta.shape = vvm::tensor::TensorShape::makeContiguous({1, 4, 1024, 1024});

    auto serverTensor = transport->allocateTensor(serverMeta, 0);
    if (!serverTensor) {
        std::cerr << "FAIL: allocate server tensor\n";
        transport->shutdown();
        destroyTestDevice();
        return 1;
    }

    // Fill server tensor with pattern via staging
    auto& pool = transport->getPoolManager()->getPool(0);
    auto staging = pool.allocate(
        kBytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    if (!staging || !staging->hostPtr) {
        std::cerr << "FAIL: staging buffer for server tensor\n";
        transport->shutdown();
        destroyTestDevice();
        return 1;
    }

    // Fill with 0xA5 pattern
    std::memset(staging->hostPtr, 0xA5, static_cast<size_t>(kBytes));
    if (!deviceCopy(s_dev.device, s_dev.transferQueue, staging->buffer, serverTensor->allocation.buffer, kBytes)) {
        std::cerr << "FAIL: fill server tensor VRAM\n";
        pool.deallocate(std::move(*staging));
        transport->shutdown();
        destroyTestDevice();
        return 1;
    }
    pool.deallocate(std::move(*staging));
    
    std::cout << "Server tensor 'server_to_client_tensor' allocated and filled with 0xA5\n";

    // Keep running - wait for Ctrl+C
    std::signal(SIGINT, [](int) { g_running = false; });
    
    // Announce tensor to connected clients periodically
    std::thread([&]() {
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            
            // Get cluster view and announce to peers
            auto view = transport->getClusterView();
            for (const auto& node : view) {
                if (node.id.toString() != nodeId) {
                    std::cout << "  Announcing tensor to " << node.id.toString() << "\n";
                    transport->sendTensor(*serverTensor, node.id.toString(), 
                        [](bool ok, const std::string& err) {
                            if (!ok) std::cerr << "  announce failed: " << err << "\n";
                        });
                }
            }
        }
    }).detach();
    
    std::thread([&]() {
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            std::cout << "  [heartbeat] Server running on " << nodeId << "\n";
        }
    }).detach();

    // Wait for interrupt
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\nShutting down...\n";
    transport->shutdown();
    transport.reset();
    destroyTestDevice();
    return 0;
}