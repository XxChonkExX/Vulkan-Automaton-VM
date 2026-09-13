// storage_stream_gpu_test: ExpertStreamer SSD->Chonk round-trip on a real GPU.
// Skips gracefully (exit 0) when no suitable device exists, like basic_test.
//
// Flow: write temp .vmex (3 experts x 4MB known patterns) -> open streamer
// (cap 2 resident) -> ensureResident(0,1) -> copy device->readback -> verify
// bytes -> ensureResident(2) evicts LRU -> verify stats + cold-load GB/s.

#include "vulkan_vm/storage_stream.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <set>
#include <vector>

using namespace vvm;
using vvm::storage::ExpertStreamer;
using vvm::storage::StorageStreamConfig;
using vvm::storage::StreamBackend;

static bool verifyPattern(const void* buf, size_t n, uint8_t id) {
    const auto* p = static_cast<const uint8_t*>(buf);
    // demo pattern: first/last 8 bytes == id, middle == 0xA0+id
    for (size_t i = 0; i < n; ++i) {
        uint8_t want = (i < 8 || i >= n - 8) ? id : static_cast<uint8_t>(0xA0 + id);
        if (p[i] != want) {
            std::cerr << "  mismatch @ " << i << ": got " << (int)p[i] << " want " << (int)want << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "=== storage_stream_gpu_test ===\n";

    // ---- instance ----
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VVM MoE Stream Test";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    std::vector<const char*> wanted = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
    uint32_t availCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &availCount, nullptr);
    std::vector<VkExtensionProperties> avail(availCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &availCount, avail.data());
    std::vector<const char*> instanceExts;
    for (const char* w : wanted)
        for (auto& e : avail)
            if (!std::strcmp(e.extensionName, w)) {
                instanceExts.push_back(w);
                break;
            }
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    ici.enabledExtensionCount = static_cast<uint32_t>(instanceExts.size());
    ici.ppEnabledExtensionNames = instanceExts.data();
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        std::cout << "SKIP: vkCreateInstance failed\n";
        return 0;
    }

    auto devices = enumerateDevices(instance);
    if (devices.empty()) {
        std::cout << "SKIP: no GPUs\n";
        vkDestroyInstance(instance, nullptr);
        return 0;
    }
    auto best = selectBestDevice(devices, true, 512);
    if (!best) {
        std::cout << "SKIP: no suitable GPU\n";
        vkDestroyInstance(instance, nullptr);
        return 0;
    }
    std::cout << "Using: " << best->props.deviceName << "\n";

    std::vector<const char*> required = {
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    };
    if (!checkDeviceExtensionSupport(best->device, required)) {
        std::cout << "SKIP: missing BDA/timeline\n";
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

    // ---- device ----
    auto queues = findQueueFamilies(best->device);
    float prio = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> qis;
    std::set<uint32_t> seen;
    auto addQ = [&](std::optional<uint32_t> f) {
        if (f && seen.insert(*f).second) {
            VkDeviceQueueCreateInfo q{};
            q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            q.queueFamilyIndex = *f;
            q.queueCount = 1;
            q.pQueuePriorities = &prio;
            qis.push_back(q);
        }
    };
    addQ(queues.graphics);
    addQ(queues.compute);
    addQ(queues.transfer);

    VkPhysicalDeviceVulkan12Features v12{};
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.bufferDeviceAddress = VK_TRUE;
    v12.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &v12;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = static_cast<uint32_t>(qis.size());
    dci.pQueueCreateInfos = qis.data();
    dci.pNext = &f2;
    dci.enabledExtensionCount = static_cast<uint32_t>(required.size());
    dci.ppEnabledExtensionNames = required.data();
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(best->device, &dci, nullptr, &device) != VK_SUCCESS) {
        std::cout << "SKIP: vkCreateDevice failed\n";
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

    uint32_t tf = queues.transfer.value_or(queues.compute.value_or(queues.graphics.value_or(0)));
    VkQueue tq = VK_NULL_HANDLE, gq = VK_NULL_HANDLE, cq = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, tf, 0, &tq);
    if (queues.graphics) vkGetDeviceQueue(device, *queues.graphics, 0, &gq);
    if (queues.compute) vkGetDeviceQueue(device, *queues.compute, 0, &cq);

    DeviceConfig dc{};
    dc.physicalDevice = best->device;
    dc.device = device;
    dc.graphicsQueueFamily = queues.graphics.value_or(0);
    dc.computeQueueFamily = queues.compute.value_or(dc.graphicsQueueFamily);
    dc.transferQueueFamily = tf;
    dc.graphicsQueue = gq;
    dc.computeQueue = cq;
    dc.transferQueue = tq;

    PoolConfig pc{};
    pc.blockSize = 64 * 1024 * 1024;
    pc.minAlignment = 64 * 1024;
    pc.enableHostVisible = true;
    pc.enableExternal = false;
    pc.enableDeviceAddress = true;
    pc.maxBlocks = 4;

    auto pool = UnifiedMemoryPool::create(dc, pc);
    if (!pool) {
        std::cerr << "FAIL: pool create\n";
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    std::cout << "Pool created\n";

    // ---- temp pack: 3 experts x 4MB ----
    const std::string packPath = "moe_gpu_test.vmex";
    constexpr size_t kExpertBytes = 4 * 1024 * 1024;
    {
        std::vector<std::pair<uint32_t, std::vector<uint8_t>>> experts;
        for (uint32_t id = 0; id < 3; ++id) {
            std::vector<uint8_t> blob(kExpertBytes, static_cast<uint8_t>(0xA0 + id));
            for (int i = 0; i < 8; ++i) blob[i] = static_cast<uint8_t>(id);
            for (int i = 0; i < 8; ++i) blob[blob.size() - 1 - i] = static_cast<uint8_t>(id);
            experts.emplace_back(id, std::move(blob));
        }
        if (!vvm::storage::writePackFile(packPath, experts, nullptr)) {
            std::cerr << "FAIL: write pack\n";
            return 1;
        }
    }

    StorageStreamConfig sc;
    sc.packPath = packPath;
    sc.backend = StreamBackend::PureVulkan;
    sc.maxResidentExperts = 2; // force eviction on 3rd
    sc.stagingBytes = 16ull * 1024 * 1024;
    ExpertStreamer streamer(&*pool, sc);
    if (auto r = streamer.open(); !r) {
        std::cerr << "FAIL: open: " << r.message << "\n";
        return 1;
    }
    std::cout << "Streamer open (backend=pure)\n";

    // readback buffer (host-visible) for verification
    auto readback = pool->allocate(kExpertBytes,
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!readback || !readback->hostPtr) {
        std::cerr << "FAIL: readback alloc\n";
        return 1;
    }

    int failures = 0;
    auto loadAndCheck = [&](uint32_t id) {
        auto t0 = std::chrono::steady_clock::now();
        auto res = streamer.ensureResident(id);
        auto t1 = std::chrono::steady_clock::now();
        if (!res) {
            std::cerr << "FAIL: ensureResident(" << id << "): " << res.message << "\n";
            ++failures;
            return;
        }
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gbps = (kExpertBytes / 1e9) / (ms / 1e3);
        std::cout << "  expert " << id << " resident in " << ms << "ms (" << gbps << " GB/s)\n";
        Allocation* a = res.value;
        if (!pool->copyBuffer(*a, *readback, 0, 0, kExpertBytes)) {
            std::cerr << "FAIL: copyback expert " << id << "\n";
            ++failures;
            return;
        }
        if (!verifyPattern(readback->hostPtr, kExpertBytes, static_cast<uint8_t>(id))) {
            std::cerr << "FAIL: pattern expert " << id << "\n";
            ++failures;
            return;
        }
        std::cout << "  expert " << id << " pattern OK\n";
    };

    loadAndCheck(0);
    loadAndCheck(1);
    // prefetch hit path
    if (auto r = streamer.prefetch({0, 1}); !r) {
        std::cerr << "FAIL: prefetch\n";
        ++failures;
    }
    // 3rd expert evicts LRU (cap 2)
    loadAndCheck(2);
    auto stats = streamer.stats();
    std::cout << "  stats: streamed=" << stats.bytesStreamed
              << " hits=" << stats.prefetchHits << " miss=" << stats.prefetchMisses
              << " evict=" << stats.evictions << "\n";
    if (stats.evictions == 0) {
        std::cerr << "FAIL: expected eviction with cap=2\n";
        ++failures;
    }
    if (stats.bytesStreamed != 3 * kExpertBytes) {
        std::cerr << "FAIL: streamed bytes " << stats.bytesStreamed << "\n";
        ++failures;
    }

    pool->deallocate(std::move(*readback));
    streamer.close();
    std::remove(packPath.c_str());
    pool.reset();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    std::cout << (failures == 0 ? "ALL GPU TESTS PASSED\n" : "FAILURES\n");
    return failures == 0 ? 0 : 1;
}
