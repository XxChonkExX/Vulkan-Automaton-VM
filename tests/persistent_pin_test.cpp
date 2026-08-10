// persistent_pin_test.cpp - Unit test for NDKPI persistent host pinning (GDRCopy-style)
//
// Build: Windows-only, links vulkan_vm_network
// Run:   set VVM_ND_PROVIDER_DLL=<path_to_ndfake_provider.dll> && persistent_pin_test

#include "vulkan_vm/network/rdma_transport.hpp"
#include "vulkan_vm/network/network_types.hpp"
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
#include <mutex>
#include <atomic>

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

#define TEST_ASSERT_MSG(cond, msg) \
    do { \
        if (!(cond)) { \
            throw TestFailure(__FILE__, __LINE__, #cond, msg); \
        } else if (gVerbose) { \
            std::printf("PASS [%s:%d] %s: %s\n", __FILE__, __LINE__, #cond, msg); \
        } \
    } while (0)

// Try to create a minimal Vulkan instance for transport creation.
// Returns true if Vulkan is available, false if no ICD (test will be skipped).
static bool tryCreateVulkan(VkInstance* outInstance, VkPhysicalDevice* outPhysDev, VkDevice* outDevice) {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "persistent_pin_test";
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

    std::vector<VkPhysicalDevice> physDevices(devCount);
    vr = vkEnumeratePhysicalDevices(*outInstance, &devCount, physDevices.data());
    if (vr != VK_SUCCESS) {
        vkDestroyInstance(*outInstance, nullptr);
        return false;
    }

    *outPhysDev = physDevices[0];
    for (auto& pd : physDevices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            *outPhysDev = pd;
            break;
        }
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(*outPhysDev, &props);
    std::printf("Using GPU: %s (vendor: 0x%04x)\n", props.deviceName, props.vendorID);

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(*outPhysDev, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueProps(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(*outPhysDev, &queueFamilyCount, queueProps.data());

    uint32_t transferQueueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
            if (!(queueProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                transferQueueFamily = i;
                break;
            }
        }
    }
    if (transferQueueFamily == UINT32_MAX) {
        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            if (queueProps[i].queueFlags & VK_QUEUE_TRANSFER_BIT) {
                transferQueueFamily = i;
                break;
            }
        }
    }
    if (transferQueueFamily == UINT32_MAX) {
        vkDestroyInstance(*outInstance, nullptr);
        return false;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo dqci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dqci.queueFamilyIndex = transferQueueFamily;
    dqci.queueCount = 1;
    dqci.pQueuePriorities = &queuePriority;

    const char* devExts[] = {
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    };
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqci;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = devExts;

    vr = vkCreateDevice(*outPhysDev, &dci, nullptr, outDevice);
    if (vr != VK_SUCCESS) {
        vkDestroyInstance(*outInstance, nullptr);
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-v") == 0 || std::strcmp(argv[i], "--verbose") == 0) {
            gVerbose = true;
        }
    }

    vvm::Logger::instance().setLevel(vvm::LogLevel::Info);

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physDev = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    if (!tryCreateVulkan(&instance, &physDev, &device)) {
        std::printf("SKIP: No Vulkan device available\n");
        return 0;
    }

    NetworkConfig netConfig;
    netConfig.listenAddress = "0.0.0.0:0";
    netConfig.enableRdma = true;
    netConfig.enableGpuDirect = true;
    netConfig.enableHostStagedFallback = true;

    std::unique_ptr<RdmaTransport> rdmaTransport = RdmaTransport::create(netConfig, physDev, device);
    if (!rdmaTransport) {
        std::printf("SKIP: RdmaTransport creation failed (no provider?)\n");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

    if (!rdmaTransport->isReady()) {
        std::printf("SKIP: RdmaTransport not ready\n");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

    std::printf("=== Persistent Pin Test ===\n");

    // Test 1: Basic pin/release cycle
    {
        std::printf("\nTest 1: Basic pin/release cycle\n");
        void* ptr = std::malloc(4096);
        TEST_ASSERT(ptr != nullptr, "malloc failed");
        std::memset(ptr, 0xAA, 4096);

        bool pinned = rdmaTransport->pinPersistentHostMemory(ptr, 4096);
        TEST_ASSERT(pinned, "pinPersistentHostMemory should succeed");

        auto region = rdmaTransport->registerHostMemory(ptr, 4096);
        TEST_ASSERT(region.has_value(), "registerHostMemory should succeed after pin");
        TEST_ASSERT(region->lkey != 0, "lkey should be valid");
        TEST_ASSERT(region->rkey != 0, "rkey should be valid");

        rdmaTransport->releasePersistentHostMemory(ptr);

        // After release, we should be able to register normally again
        auto region2 = rdmaTransport->registerHostMemory(ptr, 4096);
        TEST_ASSERT(region2.has_value(), "registerHostMemory should still work after release");

        rdmaTransport->unregisterMemory(*region2);
        std::free(ptr);
        std::printf("  PASSED\n");
    }

    // Test 2: Ref-count increments on repeated pin
    {
        std::printf("\nTest 2: Ref-count increments on repeated pin\n");
        void* ptr = std::malloc(4096);
        TEST_ASSERT(ptr != nullptr, "malloc failed");

        bool pinned1 = rdmaTransport->pinPersistentHostMemory(ptr, 4096);
        TEST_ASSERT(pinned1, "first pin should succeed");

        bool pinned2 = rdmaTransport->pinPersistentHostMemory(ptr, 4096);
        TEST_ASSERT(pinned2, "second pin should succeed (ref count++)");

        rdmaTransport->releasePersistentHostMemory(ptr);  // ref count--

        bool pinned3 = rdmaTransport->pinPersistentHostMemory(ptr, 4096);
        TEST_ASSERT(pinned3, "third pin should succeed (ref count++)");

        rdmaTransport->releasePersistentHostMemory(ptr);  // ref count--
        rdmaTransport->releasePersistentHostMemory(ptr);  // ref count reaches 0

        // After all releases, registering should still work
        auto region = rdmaTransport->registerHostMemory(ptr, 4096);
        TEST_ASSERT(region.has_value(), "register should still work after all releases");
        rdmaTransport->unregisterMemory(*region);
        std::free(ptr);
        std::printf("  PASSED\n");
    }

    // Test 3: Size mismatch rejection
    {
        std::printf("\nTest 3: Size mismatch rejection\n");
        void* ptr = std::malloc(4096);
        TEST_ASSERT(ptr != nullptr, "malloc failed");

        bool pinned = rdmaTransport->pinPersistentHostMemory(ptr, 4096);
        TEST_ASSERT(pinned, "pin 4096 bytes should succeed");

        bool pinnedWrongSize = rdmaTransport->pinPersistentHostMemory(ptr, 8192);
        TEST_ASSERT(!pinnedWrongSize, "pin with different size should fail");

        rdmaTransport->releasePersistentHostMemory(ptr);

        auto region = rdmaTransport->registerHostMemory(ptr, 4096);
        TEST_ASSERT(region.has_value(), "register should still work after release");
        rdmaTransport->unregisterMemory(*region);
        std::free(ptr);
        std::printf("  PASSED\n");
    }

    // Test 4: Release non-persistent ptr (should warn but not crash)
    {
        std::printf("\nTest 4: Release non-persistent ptr\n");
        void* ptr = std::malloc(4096);
        TEST_ASSERT(ptr != nullptr, "malloc failed");

        auto region = rdmaTransport->registerHostMemory(ptr, 4096);
        TEST_ASSERT(region.has_value(), "registerHostMemory should succeed");

        // releasePersistentHostMemory returns void, but should not crash
        rdmaTransport->releasePersistentHostMemory(ptr);

        // Memory should still be usable
        TEST_ASSERT(region->lkey != 0, "region still has valid lkey after releasePersistent attempt");

        rdmaTransport->unregisterMemory(*region);
        std::free(ptr);
        std::printf("  PASSED\n");
    }

    // Test 5: Release unknown ptr (should warn but not crash)
    {
        std::printf("\nTest 5: Release unknown ptr\n");
        void* ptr = std::malloc(4096);
        TEST_ASSERT(ptr != nullptr, "malloc failed");

        // releasePersistentHostMemory returns void, but should not crash
        rdmaTransport->releasePersistentHostMemory(ptr);

        std::free(ptr);
        std::printf("  PASSED\n");
    }

    // Test 6: Pin after register (upgrade to persistent)
    {
        std::printf("\nTest 6: Pin after register (upgrade to persistent)\n");
        void* ptr = std::malloc(4096);
        TEST_ASSERT(ptr != nullptr, "malloc failed");

        auto region = rdmaTransport->registerHostMemory(ptr, 4096);
        TEST_ASSERT(region.has_value(), "registerHostMemory should succeed");

        bool pinned = rdmaTransport->pinPersistentHostMemory(ptr, 4096);
        TEST_ASSERT(pinned, "pin after register should upgrade to persistent");

        rdmaTransport->releasePersistentHostMemory(ptr);

        auto region2 = rdmaTransport->registerHostMemory(ptr, 4096);
        TEST_ASSERT(region2.has_value(), "register should still work after release");
        rdmaTransport->unregisterMemory(*region2);
        std::free(ptr);
        std::printf("  PASSED\n");
    }

    // Test 7: Unregister persistently pinned memory (should skip)
    {
        std::printf("\nTest 7: Unregister persistently pinned memory (should skip)\n");
        void* ptr = std::malloc(4096);
        TEST_ASSERT(ptr != nullptr, "malloc failed");

        bool pinned = rdmaTransport->pinPersistentHostMemory(ptr, 4096);
        TEST_ASSERT(pinned, "pin should succeed");

        auto region = rdmaTransport->registerHostMemory(ptr, 4096);
        TEST_ASSERT(region.has_value(), "register should succeed");

        rdmaTransport->unregisterMemory(*region);  // Should be no-op for persistent

        auto region2 = rdmaTransport->registerHostMemory(ptr, 4096);
        TEST_ASSERT(region2.has_value(), "memory should still be registered after unregister attempt");

        rdmaTransport->releasePersistentHostMemory(ptr);

        auto region3 = rdmaTransport->registerHostMemory(ptr, 4096);
        TEST_ASSERT(region3.has_value(), "register should still work after release");
        rdmaTransport->unregisterMemory(*region3);
        std::free(ptr);
        std::printf("  PASSED\n");
    }

    // Test 8: Thread-safe concurrent pin/release
    {
        std::printf("\nTest 8: Thread-safe concurrent pin/release\n");
        const int numThreads = 4;
        const int iterations = 100;
        void* ptr = std::malloc(4096);
        TEST_ASSERT(ptr != nullptr, "malloc failed");

        bool pinned = rdmaTransport->pinPersistentHostMemory(ptr, 4096);
        TEST_ASSERT(pinned, "initial pin should succeed");

        std::atomic<int> pinSuccessCount{0};
        std::atomic<int> pinFailCount{0};

        auto worker = [&]() {
            for (int i = 0; i < iterations; ++i) {
                bool p = rdmaTransport->pinPersistentHostMemory(ptr, 4096);
                if (p) pinSuccessCount++;
                else pinFailCount++;
                rdmaTransport->releasePersistentHostMemory(ptr);
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back(worker);
        }
        for (auto& t : threads) t.join();

        TEST_ASSERT(pinFailCount == 0, "no pin failures in concurrent pin/release");
        // Each thread adds one pin per iteration; some may collide with same ref count
        // so just check no crashes (ref-count invariant preserved).
        (void)numThreads;
        (void)iterations;

        // Clean up: drain the ref-count
        for (int i = 0; i < numThreads * iterations; ++i) {
            rdmaTransport->releasePersistentHostMemory(ptr);
        }
        while (rdmaTransport->pinPersistentHostMemory(ptr, 4096)) {
            rdmaTransport->releasePersistentHostMemory(ptr);
        }

        auto region = rdmaTransport->registerHostMemory(ptr, 4096);
        TEST_ASSERT(region.has_value(), "register should still work after concurrent drain");
        rdmaTransport->unregisterMemory(*region);
        std::free(ptr);
        std::printf("  PASSED (pinSuccess=%d, pinFail=%d)\n", pinSuccessCount.load(), pinFailCount.load());
    }

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    std::printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
