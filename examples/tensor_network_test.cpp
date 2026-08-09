#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"
#include "vulkan_vm/tensor_transport.hpp"
#include "vulkan_vm/network/multi_node_manager.hpp"

#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>

// ---------------------------------------------------------------------------
// Minimal device setup (single GPU, graphics + compute + transfer queues)
// ---------------------------------------------------------------------------
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
    appInfo.pApplicationName = "VulkanVM Tensor Network Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> instExts = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
        "VK_KHR_get_surface_capabilities2"
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

    if (vkCreateInstance(&ici, nullptr, &s_dev.instance) != VK_SUCCESS) {
        std::cerr << "FAIL: create instance\n"; return false;
    }

    auto devices = vvm::enumerateDevices(s_dev.instance);
    if (devices.empty()) { std::cerr << "FAIL: no GPU\n"; return false; }
    auto bestDevice = vvm::selectBestDevice(devices, true, 1024);
    if (!bestDevice) { std::cerr << "FAIL: no suitable GPU\n"; return false; }
    s_dev.physicalDevice = bestDevice->device;
    std::cout << "Selected: " << bestDevice->props.deviceName << "\n" << std::flush;

    auto queues = vvm::findQueueFamilies(s_dev.physicalDevice);
    s_dev.graphicsFamily = queues.graphics.value_or(0);
    s_dev.computeFamily = queues.compute.value_or(0);
    s_dev.transferFamily = queues.transfer.value_or(0);
    if (s_dev.graphicsFamily == UINT32_MAX || s_dev.transferFamily == UINT32_MAX) {
        std::cerr << "FAIL: missing required queues\n"; return false;
    }

    const float prio = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qcis;
    auto addQueue = [&](std::optional<uint32_t> family, const char* name) {
        if (!family) return;
        VkDeviceQueueCreateInfo q{};
        q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        q.queueFamilyIndex = *family;
        q.queueCount = 1;
        q.pQueuePriorities = &prio;
        qcis.push_back(q);
        std::cout << "  Queue " << name << ": family " << *family << "\n" << std::flush;
    };
    addQueue(queues.graphics, "graphics");
    addQueue(queues.compute, "compute");
    addQueue(queues.transfer, "transfer");

    uint32_t devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(s_dev.physicalDevice, nullptr, &devExtCount, nullptr);
    std::vector<VkExtensionProperties> devExtProps(devExtCount);
    vkEnumerateDeviceExtensionProperties(s_dev.physicalDevice, nullptr, &devExtCount, devExtProps.data());

    const char* allDevExts[] = {
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#ifdef VVM_PLATFORM_WINDOWS
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#endif
        VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
    };
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

    VkResult devResult = vkCreateDevice(s_dev.physicalDevice, &dci, nullptr, &s_dev.device);
    if (devResult != VK_SUCCESS) {
        std::cerr << "FAIL: create device (VkResult=" << devResult << ")\n"; return false;
    }

    if (queues.graphics) vkGetDeviceQueue(s_dev.device, *queues.graphics, 0, &s_dev.graphicsQueue);
    if (queues.compute) vkGetDeviceQueue(s_dev.device, *queues.compute, 0, &s_dev.computeQueue);
    if (queues.transfer) vkGetDeviceQueue(s_dev.device, *queues.transfer, 0, &s_dev.transferQueue);
    return true;
}

static void destroyTestDevice() {
    if (s_dev.device) vkDestroyDevice(s_dev.device, nullptr);
    if (s_dev.instance) vkDestroyInstance(s_dev.instance, nullptr);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static vvm::DeviceConfig makeDevConfig() {
    vvm::DeviceConfig cfg;
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

static vvm::PoolConfig makePoolConfig() {
    vvm::PoolConfig cfg;
    cfg.blockSize = 256 * 1024 * 1024;
    cfg.minAlignment = 256;
    cfg.enableHostVisible = true;
    cfg.enableExternal = true;
    cfg.enableDeviceAddress = true;
    cfg.maxBlocks = 8;
    cfg.maxHeapFraction = 0.0f;
    cfg.maxPoolBytes = 0;
    return cfg;
}

// Device-side copy (src buffer -> dst buffer) using a transient command pool.
static bool deviceCopy(VkDevice device, VkQueue queue, VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandPool pooled[1];
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpci.queueFamilyIndex = s_dev.transferFamily;
    if (vkCreateCommandPool(device, &cpci, nullptr, &pooled[0]) != VK_SUCCESS) return false;

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pooled[0];
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &cbai, &cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(device, pooled[0], nullptr);
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
        vkFreeCommandBuffers(device, pooled[0], 1, &cmd);
        vkDestroyCommandPool(device, pooled[0], nullptr);
        return false;
    }

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    bool ok = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS;
    if (ok) ok = vkQueueWaitIdle(queue) == VK_SUCCESS;

    vkFreeCommandBuffers(device, pooled[0], 1, &cmd);
    vkDestroyCommandPool(device, pooled[0], nullptr);
    return ok;
}

static bool verifyPattern(const void* buf, size_t size, uint8_t expected) {
    const auto* p = static_cast<const uint8_t*>(buf);
    for (size_t i = 0; i < size; ++i) {
        if (p[i] != expected) return false;
    }
    return true;
}

static void fillPattern(void* buf, size_t size, uint8_t val) {
    std::memset(buf, val, size);
}

// ---------------------------------------------------------------------------
// Main: GPU-to-GPU VRAM share over TCP (unified tensor transport)
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== VulkanVM Tensor Network Test (GPU-to-GPU VRAM over TCP) ===\n\n";

    if (!initTestDevice()) {
        std::cerr << "FAILED to initialize Vulkan device\n";
        return 1;
    }

    int failures = 0;
    auto devCfg = makeDevConfig();
    auto poolCfg = makePoolConfig();

    // ---- Two tensor transports as two "nodes" on loopback ----
    const uint16_t portA = 51011;
    const uint16_t portB = 51012;

    vvm::tensor::TransportConfig cfgA;
    cfgA.preference = vvm::tensor::TransportConfig::Preference::NetworkOnly;
    cfgA.listenAddress = "127.0.0.1";
    cfgA.networkPort = portA;
    cfgA.enableAsyncPipeline = false;

    vvm::tensor::TransportConfig cfgB = cfgA;
    cfgB.networkPort = portB;
    cfgB.seedNodes = {"127.0.0.1:" + std::to_string(portA)};

    auto nodeA = vvm::tensor::createTensorTransport(cfgA, {devCfg}, poolCfg);
    auto nodeB = vvm::tensor::createTensorTransport(cfgB, {devCfg}, poolCfg);
    if (!nodeA || !nodeB) {
        std::cerr << "FAIL: create transports\n";
        destroyTestDevice();
        return 1;
    }

    if (!nodeA->initialize() || !nodeB->initialize()) {
        std::cerr << "FAIL: initialize transports\n";
        destroyTestDevice();
        return 1;
    }

    if (!nodeA->joinCluster("127.0.0.1:" + std::to_string(portA)) ||
        !nodeB->joinCluster("127.0.0.1:" + std::to_string(portB))) {
        std::cerr << "FAIL: joinCluster\n";
        destroyTestDevice();
        return 1;
    }

    std::cout << "Node A listening on port " << portA << "\n";
    std::cout << "Node B listening on port " << portB << " (seed -> A)\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ---- Build a named tensor on node B and fill its VRAM ----
    const VkDeviceSize kBytes = 16ull * 1024 * 1024;

    vvm::tensor::TensorMetadata meta;
    meta.dtype = vvm::tensor::DataType::Float32;
    meta.name = "gpu_to_gpu_tensor";
    meta.shape = vvm::tensor::TensorShape::makeContiguous({1, 4, 1024, 1024});

    auto srcB = nodeB->allocateTensor(meta, 0);
    auto dstA = nodeA->allocateTensor(meta, 0);
    if (!srcB || !dstA) {
        std::cerr << "FAIL: allocate tensors\n";
        failures++;
    } else {
        // Fill B's VRAM tensor via a host-visible staging buffer + device copy.
        auto& poolB = nodeB->getPoolManager()->getPool(0);
        auto staging = poolB.allocate(
            kBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!staging || !staging->hostPtr) {
            std::cerr << "FAIL: staging buffer on B\n";
            failures++;
        } else {
            fillPattern(staging->hostPtr, kBytes, 0x5A);
            if (!deviceCopy(s_dev.device, s_dev.transferQueue, staging->buffer, srcB->buffer(), kBytes)) {
                std::cerr << "FAIL: fill tensor VRAM on B\n";
                failures++;
            } else {
                // ---- B sends its GPU VRAM tensor to A over TCP ----
                std::cout << "\n--- B sends tensor to A (VRAM over TCP) ---\n";
                std::string nodeBId = "127.0.0.1:" + std::to_string(portB) + "#0";
                std::string nodeAId = "127.0.0.1:" + std::to_string(portA) + "#0";

                bool sent = false;
                bool recvd = false;
                std::string err;
                std::thread sender([&] {
                    sent = nodeB->sendTensor(*srcB, nodeAId, [&](bool ok, const std::string& e) {
                        err = e;
                    });
                });
                std::thread receiver([&] {
                    recvd = nodeA->recvTensor(*dstA, nodeBId, [&](bool ok, const std::string& e) {
                        err = e;
                    });
                });
                sender.join();
                receiver.join();

                if (!sent) {
                    std::cerr << "FAIL: sendTensor: " << err << "\n";
                    failures++;
                } else {
                    std::cout << "  sendTensor: PASS\n";
                }
                if (!recvd) {
                    std::cerr << "FAIL: recvTensor: " << err << "\n";
                    failures++;
                } else {
                    std::cout << "  recvTensor: PASS\n";
                }

                // ---- Verify the data landed in A's GPU VRAM ----
                auto& poolA = nodeA->getPoolManager()->getPool(0);
                auto readback = poolA.allocate(
                    kBytes,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                if (!readback || !readback->hostPtr) {
                    std::cerr << "FAIL: readback buffer on A\n";
                    failures++;
                } else {
                    if (!deviceCopy(s_dev.device, s_dev.transferQueue, dstA->buffer(), readback->buffer, kBytes)) {
                        std::cerr << "FAIL: readback copy on A\n";
                        failures++;
                    } else {
                        bool ok = verifyPattern(readback->hostPtr, kBytes, 0x5A);
                        std::cout << "  VRAM content verify on A: " << (ok ? "PASS" : "FAIL") << "\n";
                        if (!ok) failures++;
                    }
                    poolA.deallocate(std::move(*readback));
                }
            }
            poolB.deallocate(std::move(*staging));
        }
        nodeA->freeTensor(std::move(dstA));
        nodeB->freeTensor(std::move(srcB));
    }

    // ---- Cleanup ----
    std::cout << "\n--- Cleanup ---\n";
    nodeA->leaveCluster();
    nodeB->leaveCluster();
    nodeA->shutdown();
    nodeB->shutdown();
    nodeA.reset();
    nodeB.reset();
    destroyTestDevice();

    std::cout << "\n=== " << (failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
              << " (" << failures << " failures) ===\n";
    return failures;
}
