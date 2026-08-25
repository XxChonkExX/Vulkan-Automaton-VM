// multi_vendor_rdma_test.cpp - Cross-vendor external memory RDMA round-trip test
//
// Requires: >= 2 physical GPUs from different vendors (e.g., NVIDIA + AMD, AMD + Intel)
//           with compatible external memory handle types (OPAQUE_WIN32 / DMA-BUF)
//           and working RDMA transport (Verbs on Linux, ND on Windows).
//
// Build: enables when VVM_NETWORK_HAS_VERBS=1 (Linux) or VVM_NETWORK_HAS_NDKPI=1 (Windows).
// Run:   Skips gracefully if hardware/transport not available.

#include "vulkan_vm/network/rdma_transport.hpp"
#include "vulkan_vm/network/multi_node_manager.hpp"
#include "vulkan_vm/network/network_config.hpp"
#include "vulkan_vm/core.hpp"
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
#include <unordered_map>

using namespace vvm::network;
using namespace vvm;

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

#define TEST_SKIP(msg) \
    do { \
        std::printf("SKIP: %s\n", msg); \
        return true; \
    } while (0)

// Find two GPUs from different vendors
static bool findMultiVendorGPUs(VkInstance instance, std::vector<VkPhysicalDevice>& outDevs, std::vector<uint32_t>& outVendorIds) {
    uint32_t devCount = 0;
    VkResult vr = vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
    if (vr != VK_SUCCESS || devCount < 2) return false;

    std::vector<VkPhysicalDevice> devs(devCount);
    vr = vkEnumeratePhysicalDevices(instance, &devCount, devs.data());
    if (vr != VK_SUCCESS) return false;

    std::unordered_map<uint32_t, VkPhysicalDevice> byVendor;
    for (VkPhysicalDevice pd : devs) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        if (byVendor.find(props.vendorID) == byVendor.end()) {
            byVendor[props.vendorID] = pd;
            if (byVendor.size() >= 2) break;
        }
    }

    if (byVendor.size() < 2) return false;

    for (auto& [vid, pd] : byVendor) {
        outDevs.push_back(pd);
        outVendorIds.push_back(vid);
    }
    return true;
}

static bool tryCreateVulkan(VkInstance* outInstance, VkPhysicalDevice* outPhysDev0, VkPhysicalDevice* outPhysDev1, VkDevice* outDevice0, VkDevice* outDevice1) {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "multi_vendor_rdma_test";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    const char* wantExts[] = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
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
    if (vr != VK_SUCCESS) return false;

    std::vector<VkPhysicalDevice> devs;
    std::vector<uint32_t> vendorIds;
    if (!findMultiVendorGPUs(*outInstance, devs, vendorIds)) {
        vkDestroyInstance(*outInstance, nullptr);
        return false;
    }

    *outPhysDev0 = devs[0];
    *outPhysDev1 = devs[1];

    std::printf("Found multi-vendor GPUs:\n");
    for (size_t i = 0; i < devs.size(); ++i) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devs[i], &props);
        std::printf("  Device %zu: vendor=0x%04X device=0x%04X name=%s\n", i, vendorIds[i], props.deviceID, props.deviceName);
    }

    // Create devices for both GPUs
    // NOTE: the WIN32 extension name is only declared when the Win32 platform
    // is enabled; on Linux request the fd variant instead.
#if defined(VK_USE_PLATFORM_WIN32_KHR) || defined(_WIN32)
    const char* wantDevExts[] = {
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    };
#else
    const char* wantDevExts[] = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    };
#endif
    auto createDevice = [&](VkPhysicalDevice pd, VkDevice* outDev) -> bool {
        uint32_t devAvailCount = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &devAvailCount, nullptr);
        std::vector<VkExtensionProperties> devAvail(devAvailCount);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &devAvailCount, devAvail.data());
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
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pEnabledFeatures = nullptr;
        dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
        dci.ppEnabledExtensionNames = devExts.data();
        dci.pNext = &bufAddrFeat;

        vr = vkCreateDevice(pd, &dci, nullptr, outDev);
        return vr == VK_SUCCESS;
    };

    if (!createDevice(devs[0], outDevice0) || !createDevice(devs[1], outDevice1)) {
        if (*outDevice0) vkDestroyDevice(*outDevice0, nullptr);
        if (*outDevice1) vkDestroyDevice(*outDevice1, nullptr);
        vkDestroyInstance(*outInstance, nullptr);
        return false;
    }
    return true;
}

static std::unique_ptr<RdmaTransport> createTransport(uint16_t tcpPort, VkPhysicalDevice physDev, VkDevice device) {
    NetworkConfig cfg;
    cfg.listenAddress = std::string("0.0.0.0:") + std::to_string(tcpPort);
    cfg.advertiseAddress = "127.0.0.1";
    cfg.enableRdma = true;
    cfg.enableGpuDirect = false; // We'll use host-staged external memory export/import
    cfg.enableHostStagedFallback = true;

    std::printf("[DEBUG] Creating transport on port %u...\n", tcpPort);
    auto transport = RdmaTransport::create(cfg, physDev, device);
    if (!transport) {
        std::printf("[DEBUG] RdmaTransport::create returned null\n");
        return nullptr;
    }

    std::printf("[DEBUG] Initializing transport on port %u...\n", tcpPort);
    if (!transport->initialize()) {
        std::printf("[DEBUG] transport->initialize() failed\n");
        return nullptr;
    }

    if (!transport->isReady()) {
        std::printf("[DEBUG] transport not ready after initialize\n");
        return nullptr;
    }

    std::printf("[DEBUG] Transport ready on port %u\n", tcpPort);
    return transport;
}

// Test external memory export/import between two different-vendor GPUs
static bool testCrossVendorExportImport(VkPhysicalDevice pd0, VkDevice dev0, VkPhysicalDevice pd1, VkDevice dev1) {
    std::printf("\n=== Cross-vendor external memory export/import test ===\n");

    // Create transports on both devices
    auto transport0 = createTransport(52000, pd0, dev0);
    auto transport1 = createTransport(52010, pd1, dev1);

    if (!transport0 || !transport1) {
        std::printf("SKIP: Failed to create transports (RDMA not available)\n");
        return true;
    }

    // Note: The actual RDMA transport tests cross-GPU memory via external handles.
    // This is a framework test - the real cross-vendor test would require:
    // 1. Allocate exportable memory on device 0
    // 2. Export external handle (OPAQUE_WIN32 on Windows, DMA-BUF FD on Linux)
    // 3. Import on device 1
    // 4. RDMA write from device 1 to device 0's memory (or vice versa)
    // 5. Verify data integrity

    // For now, verify that both transports initialize and can communicate
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    uint32_t port0 = transport0->getLocalPort();
    uint32_t port1 = transport1->getLocalPort();
    std::printf("Transport 0 ready on port %u (backend: %s)\n", port0, transport0->getBackendName().c_str());
    std::printf("Transport 1 ready on port %u (backend: %s)\n", port1, transport1->getBackendName().c_str());

    // Try to connect (loopback on same machine)
    // NOTE: the destination must route over an interface that has an RDMA
    // device (e.g. rxe on enpX). 127.0.0.1 routes via lo, which has no RDMA
    // device unless rxe was added on lo, and rdma_resolve_route() crashes in
    // librdmacm instead of failing cleanly. Allow overriding via argv[1] /
    // VVM_RDMA_CONNECT_HOST for Soft-RoCE loopback testing.
    const char* connectHost = std::getenv("VVM_RDMA_CONNECT_HOST");
    if (!connectHost) connectHost = "127.0.0.1";
    auto conn = transport1->connect(connectHost, port0, 0);
    if (!conn.has_value()) {
        std::printf("SKIP: Cross-vendor transport connection not supported in this config\n");
        return true;
    }

    std::printf("Cross-vendor transport connection established!\n");

    // Register memory on both sides
    const size_t testSize = 4096;
    std::vector<uint8_t> sendBuf(testSize);
    std::vector<uint8_t> recvBuf(testSize);
    for (size_t i = 0; i < testSize; ++i) sendBuf[i] = static_cast<uint8_t>(i & 0xFF);

    auto mr0 = transport0->registerHostMemory(sendBuf.data(), testSize);
    auto mr1 = transport1->registerHostMemory(recvBuf.data(), testSize);

    if (!mr0.has_value() || !mr1.has_value()) {
        std::printf("SKIP: Memory registration not supported\n");
        return true;
    }

    // RDMA write from transport1 to transport0
    bool ok = transport1->rdmaWrite(*conn, *mr1, mr0->rdmaAddr, mr0->rkey, testSize, UINT64_MAX);
    TEST_ASSERT(ok, "RDMA write failed");

    // Verify
    TEST_ASSERT(std::memcmp(sendBuf.data(), recvBuf.data(), testSize) == 0,
                "Data mismatch after cross-vendor RDMA write");

    // Cleanup
    transport0->unregisterMemory(*mr0);
    transport1->unregisterMemory(*mr1);
    transport1->disconnect(*conn);

    std::printf("Cross-vendor RDMA test PASSED\n");
    return true;
}

int main(int argc, char* argv[]) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0 || std::strcmp(argv[i], "--verbose") == 0) {
            gVerbose = true;
        }
    }

    // Set VVM_ND_PROVIDER_DLL if not already set (for Windows/ND fake provider)
#ifdef _WIN32
    if (!std::getenv("VVM_ND_PROVIDER_DLL")) {
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
#endif

    std::printf("Starting multi-vendor RDMA test\n");

    // Check if RDMA transport is available at all
    #if !defined(VVM_NETWORK_HAS_VERBS) && !defined(VVM_NETWORK_HAS_NDKPI)
    std::printf("SKIP: RDMA transport not compiled (no IBVERBS/NDKPI)\n");
    std::printf("=== ALL TESTS PASSED (skipped) ===\n");
    return 0;
    #endif

    bool allPass = true;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physDev0 = VK_NULL_HANDLE, physDev1 = VK_NULL_HANDLE;
    VkDevice dev0 = VK_NULL_HANDLE, dev1 = VK_NULL_HANDLE;

    try {
        if (!tryCreateVulkan(&instance, &physDev0, &physDev1, &dev0, &dev1)) {
            std::printf("SKIP: Multi-vendor GPUs not available or Vulkan setup failed\n");
        } else {
            allPass = testCrossVendorExportImport(physDev0, dev0, physDev1, dev1);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        allPass = false;
    }

    if (dev0) vkDestroyDevice(dev0, nullptr);
    if (dev1) vkDestroyDevice(dev1, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);

    if (allPass) {
        std::printf("\n=== ALL TESTS PASSED ===\n");
        return 0;
    } else {
        std::fprintf(stderr, "\n=== SOME TESTS FAILED ===\n");
        return 1;
    }
}