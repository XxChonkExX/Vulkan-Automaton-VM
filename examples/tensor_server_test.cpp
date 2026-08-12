/*
 * VulkanVM Cross-Machine Tensor Network Test - SERVER (Evo-X2 Linux)
 * 
 * Run on the Evo-X2 (AMD Strix Halo / Radeon 890M) as the GPU server.
 * Usage: ./tensor_server_test --port 51000 [--verbose] [--size-mb 16]
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
#include <atomic>
#include <cstdint>

using namespace vvm;
using namespace vvm::tensor;

static std::atomic<bool> g_running{true};
static bool g_verbose = false;
static int g_announce_count = 0;
static int g_send_success = 0;
static int g_send_fail = 0;

#define LOG_VERBOSE(msg) do { if (g_verbose) std::cout << "[VERBOSE] " << msg << "\n"; } while(0)
#define LOG_INFO(msg) std::cout << "[INFO] " << msg << "\n"
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << "\n"

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

static uint32_t crc32(const void* data, size_t length) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; ++i) {
        crc ^= bytes[i];
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
        }
    }
    return ~crc;
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
        LOG_ERROR("create instance");
        return false;
    }

    auto devices = enumerateDevices(s_dev.instance);
    if (devices.empty()) { LOG_ERROR("no GPU"); return false; }
    auto bestDevice = selectBestDevice(devices, false, 1024);
    if (!bestDevice) { LOG_ERROR("no suitable GPU"); return false; }
    s_dev.physicalDevice = bestDevice->device;
    LOG_INFO("GPU: " << bestDevice->props.deviceName);
    LOG_VERBOSE("Vendor ID: 0x" << std::hex << bestDevice->props.vendorID << std::dec);

    auto queues = findQueueFamilies(s_dev.physicalDevice);
    s_dev.graphicsFamily = queues.graphics.value_or(0);
    s_dev.computeFamily = queues.compute.value_or(0);
    s_dev.transferFamily = queues.transfer.value_or(0);
    if (s_dev.graphicsFamily == UINT32_MAX || s_dev.transferFamily == UINT32_MAX) {
        LOG_ERROR("missing required queues");
        return false;
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
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
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
                LOG_VERBOSE("Device extension: " << ext);
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
        LOG_ERROR("create device");
        return false;
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
    PoolConfig cfg = PoolConfig::forDevice(s_dev.physicalDevice);
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
    uint64_t tensorSizeMB = 16;
    std::string rdmaNic;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--verbose" || arg == "-v") {
            g_verbose = true;
        } else if (arg == "--size-mb" && i + 1 < argc) {
            tensorSizeMB = std::stoull(argv[++i]);
        } else if (arg == "--rdma-nic" && i + 1 < argc) {
            rdmaNic = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--port <port>] [--size-mb <size>] [--rdma-nic <device>] [--verbose]\n";
            return 0;
        }
    }

    LOG_INFO("=== VulkanVM Tensor Server (Evo-X2 Linux) ===");
    LOG_INFO("Port: " << port << ", Tensor size: " << tensorSizeMB << " MB"
             << (rdmaNic.empty() ? "" : ", RDMA NIC: " + rdmaNic));

    if (!initTestDevice()) {
        LOG_ERROR("Failed to initialize Vulkan device");
        return 1;
    }

    auto devCfg = makeDevConfig();
    auto poolCfg = makePoolConfig();

    TransportConfig cfg;
    cfg.preference = TransportConfig::Preference::NetworkOnly;
    cfg.networkPort = port;
    cfg.rdmaNicName = rdmaNic;
    cfg.enableAsyncPipeline = true;
    cfg.maxInFlightTransfers = 4;

    auto transport = Transport::create(cfg, {devCfg}, poolCfg);
    if (!transport) {
        LOG_ERROR("Failed to create transport");
        destroyTestDevice();
        return 1;
    }

    if (!transport->initialize()) {
        LOG_ERROR("Failed to initialize transport");
        destroyTestDevice();
        return 1;
    }

    std::string nodeId = transport->getLocalNodeId();
    LOG_INFO("Node ID: " << nodeId);
    
    if (!transport->joinCluster(nodeId)) {
        LOG_ERROR("Failed to join cluster");
        destroyTestDevice();
        return 1;
    }
    LOG_INFO("Joined cluster as bootstrap");

    // Allocate a server tensor and fill with pattern
    const VkDeviceSize kBytes = tensorSizeMB * 1024 * 1024;
    const uint8_t pattern = 0xA5;

    TensorMetadata serverMeta;
    serverMeta.dtype = DataType::Float32;
    serverMeta.name = "server_to_client_tensor";
    serverMeta.shape = TensorShape::makeContiguous({1, 4, 1024, 1024});

    auto serverTensor = transport->allocateTensor(serverMeta, 0);
    if (!serverTensor) {
        LOG_ERROR("Failed to allocate server tensor");
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
        LOG_ERROR("Failed to allocate staging buffer");
        transport->shutdown();
        destroyTestDevice();
        return 1;
    }

    fillPattern(staging->hostPtr, kBytes, pattern);
    if (!deviceCopy(s_dev.device, s_dev.transferQueue, staging->buffer, serverTensor->allocation.buffer, kBytes)) {
        LOG_ERROR("Failed to fill server tensor VRAM");
        pool.deallocate(std::move(*staging));
        transport->shutdown();
        destroyTestDevice();
        return 1;
    }
    pool.deallocate(std::move(*staging));
    
    uint32_t checksum = crc32(&pattern, 1);  // CRC of the fill pattern
    LOG_INFO("Tensor 'server_to_client_tensor' ready: " << kBytes << " bytes, pattern=0x" 
             << std::hex << (int)pattern << std::dec << ", CRC32=0x" << std::hex << checksum << std::dec);

    // Keep running - wait for Ctrl+C
    std::signal(SIGINT, [](int) { 
        LOG_INFO("Caught SIGINT, shutting down...");
        g_running = false; 
    });
    
    // Announce tensor to connected clients periodically
    LOG_INFO("Starting announce thread (interval: 2s)");
    std::thread announceThread([&]() {
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            if (!g_running) break;
            
            // Get cluster view and announce to peers
            auto view = transport->getClusterView();
            LOG_VERBOSE("Cluster view: " << view.size() << " nodes");
            
            for (const auto& node : view) {
                if (node.id.toString() != nodeId) {
                    g_announce_count++;
                    LOG_VERBOSE("Announcing tensor to " << node.id.toString() << " (attempt #" << g_announce_count << ")");
                    
                    try {
                        transport->sendTensor(serverTensor, node.id.toString(), 
                            [](bool ok, const std::string& err) {
                                if (ok) {
                                    g_send_success++;
                                    LOG_VERBOSE("Announce callback: SUCCESS");
                                } else {
                                    g_send_fail++;
                                    LOG_ERROR("Announce callback: FAILED - " << err);
                                }
                            });
                    } catch (const std::exception& e) {
                        g_send_fail++;
                        LOG_ERROR("Announce exception: " << e.what());
                    }
                }
            }
        }
    });
    
    LOG_INFO("Server ready. Waiting for clients... (Ctrl+C to exit)");
    auto startTime = std::chrono::steady_clock::now();
    
    // Wait for interrupt
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto endTime = std::chrono::steady_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    
    LOG_INFO("=== Server Statistics ===");
    LOG_INFO("Uptime: " << uptime << "s");
    LOG_INFO("Announce attempts: " << g_announce_count);
    LOG_INFO("Announce success: " << g_send_success);
    LOG_INFO("Announce failed: " << g_send_fail);

    LOG_INFO("Shutting down...");
    if (announceThread.joinable()) announceThread.join();
    transport->shutdown();
    transport.reset();
    destroyTestDevice();
    return 0;
}