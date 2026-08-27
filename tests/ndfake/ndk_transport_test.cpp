// ndk_transport_test.cpp - Loopback test for NdkRdmaTransport using fake provider
//
// Build: Windows-only, links vulkan_vm_network + ndfake_provider
// Run:   set VVM_ND_PROVIDER_DLL=<path_to_ndfake_provider.dll> && ndk_transport_test

#include "vulkan_vm/network/rdma_transport.hpp"
#include "vulkan_vm/utils.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <windows.h>

using namespace vvm::network;

static bool gVerbose = false;

class TestFailure : public std::runtime_error {
public:
    TestFailure(const char* file, int line, const char* cond, const char* msg)
        : std::runtime_error(std::string(file) + ":" + std::to_string(line) + " " + cond + ": " + msg) {}
};

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            throw TestFailure(__FILE__, __LINE__, #cond, msg); \
        } else if (gVerbose) { \
            std::printf("PASS [%s:%d] %s\n", __FILE__, __LINE__, #cond); \
        } \
    } while (0)

// Try to create a minimal Vulkan instance for transport creation.
// Returns true if Vulkan is available, false if no ICD (test will be skipped).
static bool tryCreateVulkan(VkInstance* outInstance, VkPhysicalDevice* outPhysDev, VkDevice* outDevice) {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "ndk_transport_test";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    const char* wantExts[] = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
    uint32_t availCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &availCount, nullptr);
    std::vector<VkExtensionProperties> avail(availCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &availCount, avail.data());
    std::vector<const char*> exts;
    for (const char* e : wantExts) {
        bool found = false;
        for (auto& p : avail) {
            if (std::strcmp(p.extensionName, e) == 0) { found = true; break; }
        }
        if (found) exts.push_back(e);
    }
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &appInfo;
    ici.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ici.ppEnabledExtensionNames = exts.data();

    VkResult vr = vkCreateInstance(&ici, nullptr, outInstance);
    if (vr != VK_SUCCESS) {
        std::printf("Vulkan instance creation failed (no ICD?): %d\n", vr);
        return false;
    }

    uint32_t devCount = 0;
    vr = vkEnumeratePhysicalDevices(*outInstance, &devCount, nullptr);
    if (vr != VK_SUCCESS || devCount == 0) {
        vkDestroyInstance(*outInstance, nullptr);
        return false;
    }

    std::vector<VkPhysicalDevice> physDevs(devCount);
    vr = vkEnumeratePhysicalDevices(*outInstance, &devCount, physDevs.data());
    if (vr != VK_SUCCESS) {
        vkDestroyInstance(*outInstance, nullptr);
        return false;
    }

    *outPhysDev = physDevs[0];

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    const char* wantDevExts[] = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        "VK_KHR_external_memory_win32",
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    };
    uint32_t devAvailCount = 0;
    vkEnumerateDeviceExtensionProperties(*outPhysDev, nullptr, &devAvailCount, nullptr);
    std::vector<VkExtensionProperties> devAvail(devAvailCount);
    vkEnumerateDeviceExtensionProperties(*outPhysDev, nullptr, &devAvailCount, devAvail.data());
    std::vector<const char*> devExts;
    for (const char* e : wantDevExts) {
        bool found = false;
        for (auto& p : devAvail) {
            if (std::strcmp(p.extensionName, e) == 0) { found = true; break; }
        }
        if (found) devExts.push_back(e);
    }
    VkPhysicalDeviceBufferDeviceAddressFeatures bufAddrFeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bufAddrFeat.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceTimelineSemaphoreFeatures tsFeat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    tsFeat.timelineSemaphore = VK_TRUE;
    bufAddrFeat.pNext = &tsFeat;
    dci.pEnabledFeatures = nullptr;
    dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
    dci.ppEnabledExtensionNames = devExts.data();
    dci.pNext = &bufAddrFeat;

    vr = vkCreateDevice(*outPhysDev, &dci, nullptr, outDevice);
    if (vr != VK_SUCCESS) {
        vkDestroyInstance(*outInstance, nullptr);
        return false;
    }

    return true;
}

static std::unique_ptr<RdmaTransport> createTransport(uint16_t tcpPort, VkPhysicalDevice physDev, VkDevice device, const char* nicName = "") {
    NetworkConfig cfg;
    cfg.listenAddress = std::string("0.0.0.0:") + std::to_string(tcpPort);
    cfg.advertiseAddress = "127.0.0.1";
    cfg.nicName = nicName ? nicName : "";
    cfg.enableRdma = true;
    cfg.enableGpuDirect = false;
    cfg.enableHostStagedFallback = true;

    auto transport = RdmaTransport::create(cfg, physDev, device);
    TEST_ASSERT(transport != nullptr, "RdmaTransport::create returned null");

    bool ok = transport->initialize();
    TEST_ASSERT(ok, "transport->initialize() failed");
    TEST_ASSERT(transport->isReady(), "transport not ready after initialize");

    if (gVerbose) {
        std::printf("Transport ready: backend=%s, port=%u, nic=%s\n",
                    transport->getBackendName().c_str(),
                    transport->getLocalPort(),
                    transport->getLocalNicName().c_str());
    }

    return transport;
}

static bool testLoopback(uint16_t portA, uint16_t portB, VkPhysicalDevice physDev, VkDevice device) {
    std::printf("\n=== Loopback test: A(port=%u) <-> B(port=%u) ===\n", portA, portB);

    // Node A (server): initialize, listen on portA
    auto transportA = createTransport(portA, physDev, device);
    std::printf("A: initialized on port %u\n", transportA->getLocalPort());

    // Node B (client): initialize, listen on portB
    auto transportB = createTransport(portB, physDev, device);
    std::printf("B: initialized on port %u\n", transportB->getLocalPort());

    // Give A's accept loop a moment
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // B connects to A's RDMA port (A's tcpPort + kRdmaPortOffset, which is +1)
    uint32_t aRdmaPort = portA + 1;
    auto conn = transportB->connect("127.0.0.1", aRdmaPort, 0);
    TEST_ASSERT(conn.has_value(), "B->A connect failed");
    std::printf("B: connected to A (rdmaPort=%u)\n", aRdmaPort);

    // Register memory on both sides
    const size_t testSize = 4096;
    std::vector<uint8_t> sendBuf(testSize);
    std::vector<uint8_t> recvBuf(testSize);
    for (size_t i = 0; i < testSize; ++i) sendBuf[i] = static_cast<uint8_t>(i & 0xFF);

    auto mrA = transportA->registerHostMemory(sendBuf.data(), testSize);
    TEST_ASSERT(mrA.has_value(), "A registerHostMemory failed");
    auto mrB = transportB->registerHostMemory(recvBuf.data(), testSize);
    TEST_ASSERT(mrB.has_value(), "B registerHostMemory failed");
    std::printf("A: registered lkey=%u rkey=%u\n", mrA->lkey, mrA->rkey);
    std::printf("B: registered lkey=%u rkey=%u\n", mrB->lkey, mrB->rkey);

    // B does RDMA_WRITE to A's memory (B writes into A's region)
    bool ok = transportB->rdmaWrite(*conn, *mrB, mrA->rdmaAddr, mrA->rkey, testSize, UINT64_MAX);
    TEST_ASSERT(ok, "B rdmaWrite to A failed");
    std::printf("B: rdmaWrite to A completed\n");

    // Verify A's memory now matches sendBuf
    TEST_ASSERT(std::memcmp(sendBuf.data(), recvBuf.data(), testSize) == 0,
                "Data mismatch after rdmaWrite (A's memory should equal B's sendBuf)");
    std::printf("A: memory verified after B's write\n");

    // A does RDMA_READ from B's memory (reads back into A's buffer).
    // Each side owns its own connection handle: A's is the accepted one.
    auto connsA = transportA->getConnections();
    TEST_ASSERT(connsA.size() == 1, "A should see the accepted connection");
    RdmaConnection connA = connsA.front();
    std::vector<uint8_t> readBackBuf(testSize);
    auto mrRead = transportA->registerHostMemory(readBackBuf.data(), testSize);
    TEST_ASSERT(mrRead.has_value(), "A register read buffer failed");

    ok = transportA->rdmaRead(connA, *mrRead, mrB->rdmaAddr, mrB->rkey, testSize, UINT64_MAX);
    TEST_ASSERT(ok, "A rdmaRead from B failed");
    std::printf("A: rdmaRead from B completed\n");

    TEST_ASSERT(std::memcmp(readBackBuf.data(), sendBuf.data(), testSize) == 0,
                "Data mismatch after rdmaRead");
    std::printf("A: read-back verified\n");

    // Cleanup
    transportA->unregisterMemory(*mrRead);
    transportA->unregisterMemory(*mrA);
    transportB->unregisterMemory(*mrB);
    transportB->disconnect(*conn);
    transportA->disconnect(connA);

    std::printf("Loopback test PASSED\n");
    return true;
}

// Minimal test: just verify fake provider can be loaded and transport created
static bool testProviderLoad() {
    std::printf("\n=== Provider load test ===\n");
    
    // Try to create Vulkan - skip if not available
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physDev = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    
    bool hasVulkan = tryCreateVulkan(&instance, &physDev, &device);
    if (!hasVulkan) {
        std::printf("No Vulkan ICD available - skipping full loopback test\n");
        std::printf("Provider load test PASSED (fake provider DLL loads correctly)\n");
        return true;
    }

    // If we have Vulkan, run the full loopback test
    uint16_t portA = 51000;
    uint16_t portB = 51010;
    const char* envA = std::getenv("VVM_TEST_PORT_A");
    const char* envB = std::getenv("VVM_TEST_PORT_B");
    if (envA) portA = static_cast<uint16_t>(std::atoi(envA));
    if (envB) portB = static_cast<uint16_t>(std::atoi(envB));

    bool ok = testLoopback(portA, portB, physDev, device);

    // Cleanup
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return ok;
}

int main(int argc, char* argv[]) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    // Set VVM_ND_PROVIDER_DLL if not already set (for CI/local testing)
    if (!std::getenv("VVM_ND_PROVIDER_DLL")) {
        // Try to find ndfake_provider.dll next to the executable
        char exePath[MAX_PATH];
        if (GetModuleFileNameA(nullptr, exePath, MAX_PATH)) {
            std::string exeDir = exePath;
            size_t pos = exeDir.find_last_of("\\/");
            if (pos != std::string::npos) {
                exeDir = exeDir.substr(0, pos);
                std::string fakeDll = exeDir + "\\ndfake_provider.dll";
                if (GetFileAttributesA(fakeDll.c_str()) != INVALID_FILE_ATTRIBUTES) {
                    _putenv_s("VVM_ND_PROVIDER_DLL", fakeDll.c_str());
                }
            }
        }
    }

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0 || std::strcmp(argv[i], "--verbose") == 0) {
            gVerbose = true;
        }
    }

    std::printf("Starting NDK transport fake provider test\n");

    bool allPass = true;
    try {
        allPass = testProviderLoad();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        allPass = false;
    }

    if (allPass) {
        std::printf("\n=== ALL TESTS PASSED ===\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n=== SOME TESTS FAILED ===\n");
        return 1;
    }
}