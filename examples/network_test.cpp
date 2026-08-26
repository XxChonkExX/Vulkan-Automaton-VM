#ifdef VVM_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"
#include "vulkan_vm/network/multi_node_manager.hpp"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <unordered_map>

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
static TestDevice s_devB;

// Create the shared VkInstance.
static bool initVulkanInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM Network Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> instExts = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
        // NOTE: no surface extensions - this test never presents.
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
    return true;
}

// Create a logical device (queues + extensions) for ONE physical device.
static bool initLogicalDevice(TestDevice& dev, VkPhysicalDevice phys) {
    dev.physicalDevice = phys;
    auto queues = vvm::findQueueFamilies(dev.physicalDevice);
    dev.graphicsFamily = queues.graphics.value_or(0);
    dev.computeFamily = queues.compute.value_or(0);
    dev.transferFamily = queues.transfer.value_or(0);
    if (dev.graphicsFamily == UINT32_MAX || dev.transferFamily == UINT32_MAX) {
        std::cerr << "FAIL: missing required queues\n"; return false;
    }

    const float prio = 1.0f;
    // Roles may share a family (llvmpipe: one family for everything; Raphael:
    // graphics==compute). Deduplicate and clamp to the family's real queue
    // count (VUID-VkDeviceCreateInfo-queueFamilyIndex-02802 / -06755).
    std::vector<VkQueueFamilyProperties> famProps;
    {
        uint32_t n = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev.physicalDevice, &n, nullptr);
        famProps.resize(n);
        vkGetPhysicalDeviceQueueFamilyProperties(dev.physicalDevice, &n, famProps.data());
    }
    std::vector<VkDeviceQueueCreateInfo> qcis;
    std::unordered_map<uint32_t, uint32_t> requested;  // family -> queues wanted
    auto addQueue = [&](std::optional<uint32_t> family) {
        if (!family) return;
        ++requested[*family];
    };
    addQueue(queues.graphics);
    addQueue(queues.compute);
    addQueue(queues.transfer);
    for (auto& [family, want] : requested) {
        const uint32_t available =
            family < famProps.size() ? famProps[family].queueCount : 1;
        VkDeviceQueueCreateInfo q{};
        q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        q.queueFamilyIndex = family;
        q.queueCount = std::max(1u, std::min(want, available));
        q.pQueuePriorities = &prio;
        qcis.push_back(q);
        std::cout << "  Queue family " << family << ": " << q.queueCount << " queue(s)\n" << std::flush;
    }

    // Query which extensions this physical device actually supports.
    uint32_t devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(dev.physicalDevice, nullptr, &devExtCount, nullptr);
    std::vector<VkExtensionProperties> devExtProps(devExtCount);
    vkEnumerateDeviceExtensionProperties(dev.physicalDevice, nullptr, &devExtCount, devExtProps.data());

    const char* allDevExts[] = {
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
#ifdef VVM_PLATFORM_LINUX
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#endif
        // NOTE: VK_EXT_memory_budget deliberately NOT enabled here - under
        // Mesa 26.0.3 ANV, vkAllocateMemory segfaults when importing an
        // external handle on a device created with this extension (see
        // docs/LINUX_TEST_RESULTS_2026-08-25.md).
        // VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
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
    std::cout << "Device extensions enabled: ";
    for (const char* e : devExts) std::cout << e << " ";
    std::cout << "\n" << std::flush;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
    dci.pQueueCreateInfos = qcis.data();
    dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
    dci.ppEnabledExtensionNames = devExts.data();

    VkPhysicalDeviceFeatures2 physFeats{};
    physFeats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    // Vulkan12Features already covers bufferDeviceAddress + timelineSemaphore;
    // chaining the individual VkPhysicalDeviceBufferDeviceAddressFeatures /
    // VkPhysicalDeviceTimelineSemaphoreFeatures structs alongside it is
    // illegal (VUID-VkDeviceCreateInfo-pNext-02830).
    VkPhysicalDeviceVulkan12Features v12Feat{};
    v12Feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12Feat.bufferDeviceAddress = VK_TRUE;
    v12Feat.timelineSemaphore = VK_TRUE;

    physFeats.pNext = &v12Feat;
    dci.pNext = &physFeats;

    VkResult devResult = vkCreateDevice(dev.physicalDevice, &dci, nullptr, &dev.device);
    if (devResult != VK_SUCCESS) {
        std::cerr << "FAIL: create device (VkResult=" << devResult << ")\n"; return false;
    }

    if (queues.graphics) vkGetDeviceQueue(dev.device, *queues.graphics, 0, &dev.graphicsQueue);
    if (queues.compute) vkGetDeviceQueue(dev.device, *queues.compute, 0, &dev.computeQueue);
    if (queues.transfer) vkGetDeviceQueue(dev.device, *queues.transfer, 0, &dev.transferQueue);
    return true;
}

// Init the shared instance and two logical devices: node A on the best-scored
// physical device, node B on the next physical from a DIFFERENT device when
// available (falls back to A's physical on single-GPU boxes). Distinct
// physicals avoid the same-device DMA-BUF re-import driver crash (observed on
// ANV/Battlemage: segfault inside libvulkan_intel.so) AND make the zero-copy
// import check exercise the verified cross-vendor sharing path.
static bool initTestDevice() {
    if (!initVulkanInstance()) return false;

    auto devices = vvm::enumerateDevices(s_dev.instance);
    if (devices.empty()) { std::cerr << "FAIL: no GPU\n"; return false; }
    auto bestDevice = vvm::selectBestDevice(devices, true, 1024);
    if (!bestDevice) { std::cerr << "FAIL: no suitable GPU\n"; return false; }

    VkPhysicalDevice physB = bestDevice->device;  // fallback: same physical
    // Prefer a different-VENDOR hardware device (exercises the verified
    // cross-vendor DMA-BUF path), then any different non-CPU device. Software
    // devices (lavapipe) are excluded: their DMA-BUF export is a stub and
    // importing it crashes the importer.
    const vvm::DeviceScore* crossVendor = nullptr;
    const vvm::DeviceScore* anyHardware = nullptr;
    for (const auto& d : devices) {
        if (d.device == bestDevice->device) continue;
        if (d.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
        if (!anyHardware) anyHardware = &d;
        if (d.vendorID != bestDevice->vendorID && !crossVendor) crossVendor = &d;
    }
    if (crossVendor) physB = crossVendor->device;
    else if (anyHardware) physB = anyHardware->device;

    if (!initLogicalDevice(s_dev, bestDevice->device)) return false;
    std::cout << "Node A device: " << bestDevice->props.deviceName << "\n" << std::flush;

    // Report B's name by matching the handle (enumerateDevices carries props).
    for (const auto& d : devices) {
        if (d.device == physB) {
            std::cout << "Node B device: " << d.props.deviceName
                      << (physB == bestDevice->device ? "  (same as A - single GPU)" : "") << "\n"
                      << std::flush;
            break;
        }
    }
    if (!initLogicalDevice(s_devB, physB)) return false;
    return true;
}

static void destroyTestDevice() {
    if (s_dev.device) vkDestroyDevice(s_dev.device, nullptr);
    if (s_devB.device) vkDestroyDevice(s_devB.device, nullptr);
    if (s_dev.instance) vkDestroyInstance(s_dev.instance, nullptr);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static vvm::DeviceConfig makeDevConfig(const TestDevice& dev) {
    vvm::DeviceConfig cfg;
    cfg.physicalDevice = dev.physicalDevice;
    cfg.device = dev.device;
    cfg.graphicsQueueFamily = dev.graphicsFamily;
    cfg.computeQueueFamily = dev.computeFamily;
    cfg.transferQueueFamily = dev.transferFamily;
    cfg.graphicsQueue = dev.graphicsQueue;
    cfg.computeQueue = dev.computeQueue;
    cfg.transferQueue = dev.transferQueue;
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
    cfg.maxHeapFraction = 0.0f;  // Explicitly disable budget cap for tests
    cfg.maxPoolBytes = 0;
    return cfg;
}

static vvm::network::NetworkConfig makeNetConfig(const std::string& listen, const std::vector<std::string>& seeds) {
    vvm::network::NetworkConfig cfg;
    cfg.listenAddress = listen;
    // Advertise exactly what we bind, so single-box (loopback) clusters
    // connect to themselves instead of the auto-detected LAN IP.
    // NOTE: advertiseAddress is host-only; the port comes from listenAddress.
    cfg.advertiseAddress = listen.substr(0, listen.rfind(':'));
    cfg.seedNodes = seeds;
    cfg.enableRdma = false;
    cfg.enableGpuDirect = false;
    cfg.enableHostStagedFallback = true;
    // NOTE: windowed push/pull streaming is now opted into per request via
    // StreamIO window callbacks (see tcp_transport.hpp); NetworkConfig no
    // longer carries enableAdaptiveWindow/streamPipelineBuffers.
    return cfg;
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
// Striped TCP sender regression: exercises the parallel stripe engine
// (multi-socket pooling, TCP_NODELAY + buffered sockets, 1:1
// socket-to-segment mapping). A raw loopback listener mirrors the receive
// side contract: connection k carries segment k, of size base+(rem if last).
// Mirrors the sender's stripe math exactly so a mapping mismatch fails.
// ---------------------------------------------------------------------------
#ifdef VVM_PLATFORM_WINDOWS
static bool runStripedSenderTest() {
    using vvm::network::TcpTransport;

    constexpr uint64_t kSliceBytes = 4ull * 1024 * 1024;
    const uint64_t kStripeTotal = (33ull * 1024 * 1024) + 12345; // remainder tail
    const size_t poolSize = 4;

    size_t hw = std::thread::hardware_concurrency();
    uint64_t sliceCount = (kStripeTotal + kSliceBytes - 1) / kSliceBytes;
    size_t stripes = poolSize;
    if (hw > 0 && stripes > hw) stripes = hw;
    if (stripes > sliceCount) stripes = static_cast<size_t>(sliceCount);
    const uint64_t base = kStripeTotal / stripes;
    const uint64_t rem = kStripeTotal % stripes;

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;

    SOCKET listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener == INVALID_SOCKET) return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0; // ephemeral port; resolved below
    if (bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        listen(listener, static_cast<int>(stripes)) != 0) {
        closesocket(listener);
        return false;
    }
    sockaddr_in bound{};
    int blen = static_cast<int>(sizeof(bound));
    if (getsockname(listener, reinterpret_cast<sockaddr*>(&bound), &blen) != 0) {
        closesocket(listener);
        return false;
    }
    const uint16_t port = ntohs(bound.sin_port);

    std::vector<uint8_t> src(kStripeTotal);
    fillPattern(src.data(), src.size(), 0x55);
    std::vector<uint8_t> assembled(kStripeTotal);
    std::atomic<bool> listenOk{true};

    std::thread listenerThread([&]() {
        std::vector<std::thread> readers;
        readers.reserve(stripes);
        for (size_t k = 0; k < stripes; ++k) {
            SOCKET c = accept(listener, nullptr, nullptr);
            if (c == INVALID_SOCKET) { listenOk = false; break; }
            int rcvTimeout = 15000; // ms: fail instead of hanging on mapping bugs
            setsockopt(c, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char*>(&rcvTimeout), sizeof(rcvTimeout));
            uint64_t segLen = base + (k == stripes - 1 ? rem : 0);
            uint8_t* dst = assembled.data() + k * base;
            readers.emplace_back([c, dst, segLen, &listenOk]() {
                size_t got = 0;
                while (got < segLen) {
                    int chunk = static_cast<int>((std::min)(segLen - got, uint64_t{INT_MAX}));
                    int n = recv(c, reinterpret_cast<char*>(dst + got), chunk, 0);
                    if (n <= 0) { listenOk = false; break; }
                    got += static_cast<size_t>(n);
                }
                closesocket(c);
            });
        }
        for (auto& r : readers) r.join();
        closesocket(listener);
    });

    TcpTransport t;
    bool sent = t.writeStreamStriped("127.0.0.1", port, src.data(), kStripeTotal, poolSize);
    t.shutdownConnectionPool("127.0.0.1", port);
    listenerThread.join();

    bool ok = sent && listenOk &&
              std::memcmp(src.data(), assembled.data(), kStripeTotal) == 0;
    WSACleanup();
    return ok;
}
#endif

// Device-side copy (src buffer -> dst buffer) using a transient command pool.
static bool deviceCopy(VkDevice device, VkQueue queue, uint32_t transferFamily,
                       VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandPool pooled[1];
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpci.queueFamilyIndex = transferFamily;
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== VulkanVM Network Test ===\n\n";

    if (!initTestDevice()) {
        std::cerr << "FAILED to initialize Vulkan device\n";
        return 1;
    }

    int failures = 0;

    // Same-process zero-copy import across VENDOR-different devices segfaults
    // inside ANV (Mesa 26.0.3) vkAllocateMemory - RADV->RADV is fine. Disable
    // ZC for cross-vendor pairs unless explicitly overridden, so the cluster
    // flow exercises host-staged migration instead of dying in the driver.
    // See docs/LINUX_TEST_RESULTS_2026-08-25.md.
    VkPhysicalDeviceProperties propsA{}, propsB{};
    vkGetPhysicalDeviceProperties(s_dev.physicalDevice, &propsA);
    vkGetPhysicalDeviceProperties(s_devB.physicalDevice, &propsB);
    const bool crossVendor = propsA.vendorID != propsB.vendorID;
    const bool zcAllowed = !crossVendor ||
                           std::getenv("VVM_ALLOW_CROSSVENDOR_ZC") != nullptr;
    if (!zcAllowed) {
        setenv("VVM_DISABLE_SAME_PROCESS_ZC", "1", 1);
        std::cout << "Cross-vendor pair (" << propsA.deviceName << " / "
                  << propsB.deviceName << "): same-process ZC disabled "
                  << "(set VVM_ALLOW_CROSSVENDOR_ZC=1 to force)\n";
    }

    // ---- Create two "nodes" on loopback ----
    auto devCfgA = makeDevConfig(s_dev);
    auto devCfgB = makeDevConfig(s_devB);
    auto poolCfg = makePoolConfig();

    const uint16_t portA = 51001;
    const uint16_t portB = 51002;
    const std::string hostA = "127.0.0.1:" + std::to_string(portA);
    const std::string hostB = "127.0.0.1:" + std::to_string(portB);

    auto mgrA = vvm::network::MultiNodePoolManager::create(
        {devCfgA}, poolCfg, makeNetConfig(hostA, {}));
    auto mgrB = vvm::network::MultiNodePoolManager::create(
        {devCfgB}, poolCfg, makeNetConfig(hostB, {hostA}));

    if (!mgrA || !mgrB) {
        std::cerr << "FAIL: could not create managers\n";
        destroyTestDevice();
        return 1;
    }

    bool startA = mgrA->start();
    bool startB = mgrB->start();
    if (!startA || !startB) {
        std::cerr << "FAIL: start()\n";
        return 1;
    }

    if (!mgrA->registerWithCluster() || !mgrB->registerWithCluster()) {
        std::cerr << "FAIL: registerWithCluster\n";
        return 1;
    }
    // Give heartbeat a moment to propagate
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    auto viewA = mgrA->getClusterView();
    auto viewB = mgrB->getClusterView();
    std::cout << "Node A cluster view: " << viewA.size() << " peers\n";
    std::cout << "Node B cluster view: " << viewB.size() << " peers\n";
    if (viewA.empty()) { std::cerr << "FAIL: A sees no peers\n"; ++failures; }
    if (viewB.empty()) { std::cerr << "FAIL: B sees no peers\n"; ++failures; }

    // ---- Remote allocate (B asks A for memory) ----
    const VkDeviceSize kTestSize = 16ull * 1024 * 1024;

    std::cout << "\n--- Remote allocate ---\n";
    auto remoteDesc = mgrB->allocateRemote(
        mgrA->getLocalNodeId(), kTestSize,
        0,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!remoteDesc) {
        std::cerr << "FAIL: allocateRemote returned nullopt\n";
        ++failures;
    } else {
        std::cout << "  Allocated " << remoteDesc->size << " bytes on node "
                  << remoteDesc->owner.toString() << ", localAllocId=" << remoteDesc->localAllocId << "\n";
        // Clean up remote allocation
        if (!mgrA->deallocateRemote(*remoteDesc)) {
            std::cerr << "FAIL: deallocateRemote (A)\n";
            ++failures;
        }
    }

    // ---- Push migration: B -> A ----
    std::cout << "\n--- Push migration: B -> A ---\n";

    auto srcB = mgrB->getLocalPool().allocate(
        kTestSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!srcB || !srcB->hostPtr) {
        std::cerr << "FAIL: allocate source on B\n";
        ++failures;
    } else {
        fillPattern(srcB->hostPtr, kTestSize, 0xAB);
        // Ask A to allocate the destination buffer remotely.
        auto dstOnA = mgrB->allocateRemote(
            mgrA->getLocalNodeId(), kTestSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!dstOnA) {
            std::cerr << "FAIL: allocateRemote dest on A (push)\n";
            ++failures;
        } else {
            auto op = mgrB->migrateToRemote(*srcB, *dstOnA, false);
            if (!op || !op->completed) {
                std::cerr << "FAIL: migrateToRemote\n";
                ++failures;
            } else {
                // Verify data arrived in A's registered allocation.
                auto regA = mgrA->getRegisteredAllocation(dstOnA->localAllocId);
                if (!regA || !regA->hostPtr) {
                    std::cerr << "FAIL: cannot read back A's registered allocation\n";
                    ++failures;
                } else {
                    bool ok = verifyPattern(regA->hostPtr, kTestSize, 0xAB);
                    if (!ok) {
                        const uint8_t* q = static_cast<const uint8_t*>(regA->hostPtr);
                        for (size_t di = 0; di < kTestSize; ++di) {
                            if (q[di] != 0xAB) {
                                std::cout << "  Push first mismatch at byte " << di << " (chunk "
                                          << (di / (4ull * 1024 * 1024)) << ")\n";
                                break;
                            }
                        }
                    }
                    std::cout << "  Push verify: " << (ok ? "PASS" : "FAIL") << "\n";
                    if (!ok) ++failures;
                }
            }
            mgrA->deallocateRemote(*dstOnA);
        }
        mgrB->getLocalPool().deallocate(std::move(*srcB));
    }

    // ---- Pull migration: A pulls from B ----
    std::cout << "\n--- Pull migration: A pulls from B ---\n";

    auto srcB2 = mgrB->getLocalPool().allocate(
        kTestSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!srcB2 || !srcB2->hostPtr) {
        std::cerr << "FAIL: allocate source2 on B\n";
        ++failures;
    } else {
        fillPattern(srcB2->hostPtr, kTestSize, 0xCD);
        auto exportedB2 = mgrB->exportForRemote(*srcB2, false, true);
        if (!exportedB2) {
            std::cerr << "FAIL: exportForRemote (B2)\n";
            ++failures;
        } else {
            auto dstA2 = mgrA->getLocalPool().allocate(
                kTestSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (!dstA2 || !dstA2->hostPtr) {
                std::cerr << "FAIL: allocate dest2 on A\n";
                ++failures;
            } else {
                auto op2 = mgrA->migrateFromRemote(*exportedB2, *dstA2, false);
                if (!op2 || !op2->completed) {
                    std::cerr << "FAIL: migrateFromRemote\n";
                    ++failures;
                } else {
                    bool ok2 = verifyPattern(dstA2->hostPtr, kTestSize, 0xCD);
                    std::cout << "  Pull verify: " << (ok2 ? "PASS" : "FAIL") << "\n";
                    if (!ok2) ++failures;
                    mgrA->getLocalPool().deallocate(std::move(*dstA2));
                }
            }
            mgrB->deallocateRemote(*exportedB2);
        }
        mgrB->getLocalPool().deallocate(std::move(*srcB2));
    }

    // ---- Zero-copy handle import: B exports, A imports on the SAME device ----
    std::cout << "\n--- Zero-copy handle import (same-host loopback) ---\n";

    if (!zcAllowed) {
        // Cross-vendor ZC is disabled for this run (see note above): imports
        // take the host-staged path and cannot share memory. Report SKIP
        // instead of FAIL - this mirrors VVM_DISABLE_SAME_PROCESS_ZC=1 runs.
        std::cout << "  Zero-copy import: SKIPPED (cross-vendor ZC disabled; "
                     "host-staged migration verified above)\n";
    } else {
    auto srcB3 = mgrB->getLocalPool().allocate(
        kTestSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!srcB3 || !srcB3->hostPtr) {
        std::cerr << "FAIL: allocate source3 on B\n";
        ++failures;
    } else {
        fillPattern(srcB3->hostPtr, kTestSize, 0xEF);
        auto exportedB3 = mgrB->exportForRemote(*srcB3, false, false);
        if (!exportedB3) {
            std::cerr << "FAIL: exportForRemote (B3)\n";
            ++failures;
        } else {
            // The exported handle was promoted into a dedicated copy on B.
            // If A's import truly shares that memory, A must observe the
            // pattern (0xEF) written into the promoted copy at export time.
            auto importOnA = mgrA->importRemote(
                *exportedB3,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            if (!importOnA) {
                std::cerr << "FAIL: importRemote (zero-copy)\n";
                ++failures;
            } else {
                auto aRead = mgrA->getLocalPool().allocate(
                    kTestSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                if (!aRead || !aRead->hostPtr) {
                    std::cerr << "FAIL: allocate host-visible read-back on A\n";
                    ++failures;
                } else {
                    bool copied = deviceCopy(s_dev.device, s_dev.transferQueue, s_dev.transferFamily,
                                             importOnA->buffer, aRead->buffer, kTestSize);
                    const bool shared = copied && verifyPattern(aRead->hostPtr, kTestSize, 0xEF);
                    std::cout << "  Zero-copy import: "
                              << (shared ? "PASS (A reads B's exported memory)"
                                         : "FAIL (import did not share memory)")
                              << "\n";
                    if (!shared) ++failures;
                    mgrA->getLocalPool().deallocate(std::move(*aRead));
                }
                mgrA->getLocalPool().deallocate(std::move(*importOnA));
            }
            mgrB->deallocateRemote(*exportedB3);
        }
        mgrB->getLocalPool().deallocate(std::move(*srcB3));
    }
    }  // zcAllowed

    // ---- Striped TCP sender (parallel stripe engine) ----
    std::cout << "\n--- Striped TCP transfer (parallel sockets) ---\n";

#ifdef VVM_PLATFORM_WINDOWS
    bool striped = runStripedSenderTest();
    std::cout << "  Striped verify: " << (striped ? "PASS" : "FAIL") << "\n";
    if (!striped) ++failures;
#else
    std::cout << "  Striped verify: SKIPPED (Windows-only raw listener)\n";
#endif

    // ---- Cleanup ----
    std::cout << "\n--- Cleanup ---\n";
    mgrA->stop();
    mgrB->stop();
    mgrA.reset();
    mgrB.reset();
    destroyTestDevice();

    std::cout << "\n=== " << (failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
              << " (" << failures << " failures) ===\n";
    return failures;
}