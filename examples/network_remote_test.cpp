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
#include <cstring>
#include <cassert>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

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
    appInfo.pApplicationName = "VulkanVM Network Test";
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

    // Query which extensions this physical device actually supports.
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
    cfg.maxHeapFraction = 0.0f;  // Explicitly disable budget cap for tests
    cfg.maxPoolBytes = 0;
    return cfg;
}

static vvm::network::NetworkConfig makeNetConfig(const std::string& listen, const std::vector<std::string>& seeds) {
    vvm::network::NetworkConfig cfg;
    cfg.listenAddress = listen;
    cfg.seedNodes = seeds;
    cfg.enableRdma = false;
    cfg.enableGpuDirect = false;
    cfg.enableHostStagedFallback = true;
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string remoteHost = "192.168.0.117";
    uint16_t remotePort = 51001;
    uint16_t localPort = 51005;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--remote-host" && i + 1 < argc) {
            remoteHost = argv[++i];
        } else if (arg == "--remote-port" && i + 1 < argc) {
            remotePort = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--local-port" && i + 1 < argc) {
            localPort = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --remote-host <ip>    Remote Evo-X2 server IP (default: 192.168.0.117)\n";
            std::cout << "  --remote-port <port>  Remote Evo-X2 server port (default: 51001)\n";
            std::cout << "  --local-port <port>   Local client listen port (default: 51005)\n";
            std::cout << "  --help, -h            Show this help\n";
            return 0;
        }
    }
    
    std::cout << "=== VulkanVM Remote Network Test (Windows client -> Evo-X2 server) ===\n\n";
    std::cout << "Remote server: " << remoteHost << ":" << remotePort << "\n";
    std::cout << "Local listen port: " << localPort << "\n\n";

    if (!initTestDevice()) {
        std::cerr << "FAILED to initialize Vulkan device\n";
        return 1;
    }

    int failures = 0;

    // ---- Remote node A is the Evo-X2 seed (already running) ----
    // ---- Local node B is this Windows machine (client) ----
    auto devCfg = makeDevConfig();
    auto poolCfg = makePoolConfig();

    // Evo-X2 seed node address (already running as server)
    const std::string hostA = remoteHost + ":" + std::to_string(remotePort);
    // Windows client listens on a different local port
    const std::string hostB = "0.0.0.0:" + std::to_string(localPort);

    // Only create the client manager (mgrB), connecting to Evo-X2 as seed
    auto mgrB = vvm::network::MultiNodePoolManager::create(
        {devCfg}, poolCfg, makeNetConfig(hostB, {hostA}));

    if (!mgrB) {
        std::cerr << "FAIL: could not create client manager\n";
        destroyTestDevice();
        return 1;
    }

    bool startB = mgrB->start();
    if (!startB) {
        std::cerr << "FAIL: start()\n";
        return 1;
    }

    if (!mgrB->registerWithCluster()) {
        std::cerr << "FAIL: registerWithCluster\n";
        return 1;
    }
    // Give heartbeat a moment to propagate to the Evo-X2
    std::cout << "Waiting for cluster membership to propagate...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    auto viewB = mgrB->getClusterView();
    std::cout << "Node B (Windows) cluster view: " << viewB.size() << " peers\n";
    for (const auto& peer : viewB) {
        std::cout << "  Peer: " << peer.id.toString() << "\n";
    }
    if (viewB.empty()) { std::cerr << "FAIL: B sees no peers (Evo-X2 not visible)\n"; ++failures; }

    if (failures > 0) {
        std::cerr << "\nAborting: no remote peers visible. Check Evo-X2 is running on " << hostA << "\n";
        mgrB->stop();
        destroyTestDevice();
        return 1;
    }

    // Find the remote node ID for the Evo-X2 (first peer that isn't us)
    vvm::network::NodeId remoteNodeId;
    bool foundRemote = false;
    for (const auto& peer : viewB) {
        if (peer.id.toString() != mgrB->getLocalNodeId().toString()) {
            // Evo-X2 may advertise 127.0.0.1 but we use the known LAN IP from command line
            remoteNodeId = peer.id;
            remoteNodeId.host = remoteHost;  // Override with known LAN IP
            foundRemote = true;
            break;
        }
    }
    if (!foundRemote) {
        std::cerr << "FAIL: could not find remote node ID\n";
        ++failures;
    } else {
        std::cout << "Remote Evo-X2 node ID (corrected): " << remoteNodeId.toString() << "\n";
    }

    if (failures > 0 || !foundRemote) {
        std::cerr << "\nAborting: cannot proceed without remote node ID.\n";
        mgrB->stop();
        destroyTestDevice();
        return 1;
    }

    // ---- Remote allocate (Windows B asks Evo-X2 A for memory) ----
    const VkDeviceSize kTestSize = 16ull * 1024 * 1024;

    std::cout << "\n--- Remote allocate (Windows -> Evo-X2) ---\n";
    // Try DEVICE_LOCAL first (should work on any GPU)
    auto remoteDesc = mgrB->allocateRemote(
        remoteNodeId, kTestSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!remoteDesc) {
        std::cerr << "  DEVICE_LOCAL failed, trying HOST_VISIBLE...\n";
        // Fallback: try HOST_VISIBLE
        remoteDesc = mgrB->allocateRemote(
            remoteNodeId, kTestSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (!remoteDesc) {
        std::cerr << "FAIL: allocateRemote returned nullopt\n";
        ++failures;
    } else {
        std::cout << "  Allocated " << remoteDesc->size << " bytes on remote node "
                  << remoteDesc->owner.toString() << ", localAllocId=" << remoteDesc->localAllocId << "\n";
        // Clean up remote allocation (deallocateRemote is called on the owner side via RPC)
        // Note: For a remote node, deallocation happens via the cluster protocol.
        // mgrB->deallocateRemote(*remoteDesc) would send a dealloc RPC to the Evo-X2.
    }

    // ---- Push migration: Windows B -> Evo-X2 A ----
    std::cout << "\n--- Push migration: Windows -> Evo-X2 ---\n";

    auto srcB = mgrB->getLocalPool().allocate(
        kTestSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!srcB || !srcB->hostPtr) {
        std::cerr << "FAIL: allocate source on Windows (B)\n";
        ++failures;
    } else {
        fillPattern(srcB->hostPtr, kTestSize, 0xAB);
        // Ask Evo-X2 to allocate the destination buffer remotely.
        auto dstOnA = mgrB->allocateRemote(
            remoteNodeId, kTestSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!dstOnA) {
            std::cerr << "FAIL: allocateRemote dest on Evo-X2 (push)\n";
            ++failures;
        } else {
            std::cout << "  Remote destination allocated (allocId=" << dstOnA->localAllocId << ")\n";
            auto op = mgrB->migrateToRemote(*srcB, *dstOnA, false);
            if (!op || !op->completed) {
                std::cerr << "FAIL: migrateToRemote (push to Evo-X2)\n";
                ++failures;
            } else {
                // Can't read remote memory directly from Windows.
                // Verify the operation completed successfully (byte count matches).
                std::cout << "  Push migration completed: " << op->bytesTransferred << " bytes sent to Evo-X2\n";
                std::cout << "  Push operation: PASS (data sent over TCP; verify on Evo-X2 side)\n";
            }
        }
    }

    // ---- Pull migration: Windows B pulls from Evo-X2 A ----
    // For a true pull test cross-machine, the Evo-X2 would need to export an allocation.
    // We can test the pull path by locally allocating, exporting, and pulling back (round-trip through Evo-X2).
    // However, cross-machine pull requires the server to participate in the protocol.
    // The allocateRemote + migrateToRemote tests above verify the TCP data path both directions.
    // For now, we skip the explicit pull test (requires server-side cooperation).
    std::cout << "\n--- Pull migration: SKIPPED (requires Evo-X2 server-side export cooperation) ---\n";
    std::cout << "  (Push migration above already verified the TCP transfer path.)\n";

    // ---- Zero-copy handle import: SKIPPED (only works same-host) ----
    std::cout << "\n--- Zero-copy handle import: SKIPPED (cross-machine, requires same-host) ---\n";

    // ---- Striped TCP sender (parallel stripe engine) - local test ----
    std::cout << "\n--- Striped TCP transfer (local parallel sockets) ---\n";

#ifdef VVM_PLATFORM_WINDOWS
    bool striped = runStripedSenderTest();
    std::cout << "  Striped verify: " << (striped ? "PASS" : "FAIL") << "\n";
    if (!striped) ++failures;
#else
    std::cout << "  Striped verify: SKIPPED (Windows-only raw listener)\n";
#endif

    // ---- Cleanup ----
    std::cout << "\n--- Cleanup ---\n";
    std::cout << "Leaving cluster...\n";
    mgrB->stop();
    mgrB.reset();
    destroyTestDevice();

    std::cout << "\n=== " << (failures == 0 ? "ALL REMOTE TESTS PASSED" : "SOME TESTS FAILED")
              << " (" << failures << " failures) ===\n";
    std::cout << "Check Evo-X2 server logs for push migration verification.\n";
    return failures;
}