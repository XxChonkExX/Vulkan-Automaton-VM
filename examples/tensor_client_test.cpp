/*
 * VulkanVM Cross-Machine Tensor Network Test - CLIENT (Windows)
 * 
 * Run on Windows to connect to Evo-X2 server and test GPU-to-GPU VRAM sharing.
 * Usage: tensor_client_test --server 192.168.0.117 --port 51000 --local-port 51005
 */
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/tensor_transport.hpp"
#include "vulkan_vm/utils.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>

using namespace vvm;
using namespace vvm::tensor;

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
    appInfo.pApplicationName = "VulkanVM Tensor Client";
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
    // Windows: prefer discrete GPU
    auto bestDevice = selectBestDevice(devices, true, 1024);
    if (!bestDevice) { std::cerr << "FAIL: no suitable GPU\n"; return false; }
    s_dev.physicalDevice = bestDevice->device;
    std::cout << "Client GPU: " << bestDevice->props.deviceName << "\n";

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
#if !defined(__linux__)
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,  // Windows
#endif
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,     // Linux (DMA-BUF)
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME, // Linux: required for DMA-BUF fd ops
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
    std::string serverIp = "192.168.0.117";
    uint16_t serverPort = 51000;
    uint16_t localPort = 51005;
    std::string rdmaNic;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--server" && i + 1 < argc) {
            serverIp = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            serverPort = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--local-port" && i + 1 < argc) {
            localPort = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--rdma-nic" && i + 1 < argc) {
            rdmaNic = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "  --server <ip>        Evo-X2 server IP (default: 192.168.0.117)\n";
            std::cout << "  --port <port>        Evo-X2 server port (default: 51000)\n";
            std::cout << "  --local-port <port>  Local client listen port (default: 51005)\n";
            std::cout << "  --rdma-nic <device>  RDMA device to use, e.g. rxe0 (default: auto)\n";
            return 0;
        }
    }

    std::cout << "=== VulkanVM Tensor Client (Ubuntu Linux) ===\n";
    std::cout << "Connecting to Evo-X2 at " << serverIp << ":" << serverPort << "\n";
    std::cout << "Local listen port: " << localPort
              << (rdmaNic.empty() ? "" : ", RDMA NIC: " + rdmaNic) << "\n\n";

    if (!initTestDevice()) {
        std::cerr << "FAILED to initialize Vulkan device\n";
        return 1;
    }

    auto devCfg = makeDevConfig();
    auto poolCfg = makePoolConfig();

    TransportConfig cfg;
    cfg.preference = TransportConfig::Preference::NetworkOnly;
    cfg.networkPort = localPort;
    cfg.rdmaNicName = rdmaNic;
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

    std::string clientNodeId = transport->getLocalNodeId();
    std::cout << "Local Node ID: " << clientNodeId << "\n";

    // Connect to server
    std::string serverNodeId = serverIp + ":" + std::to_string(serverPort) + "#0";
    std::cout << "Joining cluster with server: " << serverNodeId << "\n";
    
    if (!transport->joinCluster(serverNodeId)) {
        std::cerr << "FAIL: join cluster with server\n";
        destroyTestDevice();
        return 1;
    }

    std::cout << "Connected to cluster!\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    int failures = 0;
    const VkDeviceSize kBytes = 16ull * 1024 * 1024;

    // Test 1: Allocate tensor on client, fill with pattern, send to server
    std::cout << "\n--- Test 1: Client sends tensor to Server (VRAM over TCP) ---\n";
    
    TensorMetadata meta;
    meta.dtype = DataType::Float32;
    meta.name = "client_to_server_tensor";
    meta.shape = TensorShape::makeContiguous({1, 4, 1024, 1024});

    auto clientTensor = transport->allocateTensor(meta, 0);
    if (!clientTensor) {
        std::cerr << "FAIL: allocate client tensor\n";
        ++failures;
    } else {
        // Fill client tensor with known pattern via staging
        auto& pool = transport->getPoolManager()->getPool(0);
        auto staging = pool.allocate(
            kBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        if (!staging || !staging->hostPtr) {
            std::cerr << "FAIL: staging buffer\n";
            ++failures;
        } else {
            fillPattern(staging->hostPtr, kBytes, 0x5A);
            if (!deviceCopy(s_dev.device, s_dev.transferQueue, staging->buffer, clientTensor->allocation.buffer, kBytes)) {
                std::cerr << "FAIL: fill client tensor VRAM\n";
                ++failures;
            } else {
                std::cout << "  Client tensor filled with 0x5A pattern\n";
                
                // Send to server
                std::cout << "  Sending tensor to server...\n";
                std::atomic<bool> sendDone = false;
                std::string sendErr;
                
                transport->sendTensor(clientTensor, serverNodeId, 
                    [&](bool ok, const std::string& err) {
                        if (!ok) sendErr = err;
                        sendDone = true;
                    });
                
                // Wait for completion
                int waitCount = 0;
                while (!sendDone && waitCount < 600) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    ++waitCount;
                }
                
                if (!sendDone) {
                    std::cerr << "FAIL: sendTensor timeout\n";
                    ++failures;
                } else if (!sendErr.empty()) {
                    std::cerr << "FAIL: sendTensor: " << sendErr << "\n";
                    ++failures;
                } else {
                    std::cout << "  sendTensor: PASS\n";
                }
            }
        }
        pool.deallocate(std::move(*staging));
        // transport->freeTensor is not in the interface; tensor is freed when handle goes out of scope
    }

    // Test 2: Allocate tensor on client, recv from server
    std::cout << "\n--- Test 2: Client receives tensor from Server (VRAM over TCP) ---\n";
    
    // Use the server's tensor name
    vvm::tensor::TensorMetadata serverMeta;
    serverMeta.dtype = vvm::tensor::DataType::Float32;
    serverMeta.name = "server_to_client_tensor";
    serverMeta.shape = vvm::tensor::TensorShape::makeContiguous({1, 4, 1024, 1024});
    
    auto recvTensor = transport->allocateTensor(serverMeta, 0);
    if (!recvTensor) {
        std::cerr << "FAIL: allocate recv tensor\n";
        ++failures;
    } else {
        std::cout << "  Receiving tensor from server...\n";
        std::atomic<bool> recvDone = false;
        std::string recvErr;
        
        transport->recvTensor(recvTensor, serverNodeId,
            [&](bool ok, const std::string& err) {
                if (!ok) recvErr = err;
                recvDone = true;
            });
        
        // Wait for receive to complete (up to 30 seconds)
        int waitCount = 0;
        while (!recvDone && waitCount < 300) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            ++waitCount;
        }
        
        if (!recvDone) {
            std::cerr << "  FAIL: recvTensor timeout\n";
            ++failures;
        } else if (!recvErr.empty()) {
            std::cerr << "  FAIL: recvTensor: " << recvErr << "\n";
            ++failures;
        } else {
            std::cout << "  recvTensor: PASS (received " << kBytes << " bytes)\n";
            
            // Verify received data (copy back to host)
            auto& pool = transport->getPoolManager()->getPool(0);
            auto readback = pool.allocate(
                kBytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            
            if (!readback || !readback->hostPtr) {
                std::cerr << "FAIL: readback buffer\n";
                ++failures;
            } else {
                if (!deviceCopy(s_dev.device, s_dev.transferQueue, recvTensor->allocation.buffer, readback->buffer, kBytes)) {
                    std::cerr << "FAIL: readback copy\n";
                    ++failures;
                } else {
                    // Server should send a tensor with a different pattern (e.g., 0xA5)
                    bool ok = verifyPattern(readback->hostPtr, kBytes, 0xA5);
                    std::cout << "  VRAM content verify (expecting 0xA5 from server): " << (ok ? "PASS" : "FAIL") << "\n";
                    if (!ok) ++failures;
                }
                pool.deallocate(std::move(*readback));
            }
        }
        // recvTensor freed when handle goes out of scope
    }

    // Cleanup
    std::cout << "\n--- Cleanup ---\n";
    transport->shutdown();
    transport.reset();
    destroyTestDevice();

    std::cout << "\n=== " << (failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
              << " (" << failures << " failures) ===\n";
    return failures;
}