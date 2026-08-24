// pool_device.cpp - Vulkan device selection + Chonk Buffer pool lifecycle.

#include "pool_device.hpp"

#include <vulkan_vm/utils.hpp>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace vvm_torch {

static VkInstance g_instance = VK_NULL_HANDLE;
static VkDevice g_device = VK_NULL_HANDLE;
static vvm::DeviceConfig g_devCfg;
static std::unique_ptr<vvm::UnifiedMemoryPool> g_pool;

vvm::UnifiedMemoryPool* pool() {
    return g_pool.get();
}

DevicePreference devicePreferenceFromEnv() {
    const char* p = getenv("CHONK_DEVICE_PREFERENCE");
    if (!p) return DevicePreference::PreferAmd;  // legacy default (see note in initPoolCore)
    std::string v(p);
    if (v == "best") return DevicePreference::BestScore;
    if (v == "prefer_discrete") return DevicePreference::PreferDiscrete;
    return DevicePreference::PreferAmd;
}

static VkDevice createDeviceForPool(const vvm::DeviceScore& score, vvm::DeviceConfig& out) {
    auto queues = vvm::findQueueFamilies(score.device);
    if (!queues.transfer && !queues.graphics) return VK_NULL_HANDLE;
    uint32_t family = queues.transfer.value_or(queues.graphics.value());
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceVulkan12Features v12{};
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.bufferDeviceAddress = VK_TRUE;
    v12.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceFeatures feats{};
    feats.sparseBinding = VK_TRUE;

    const char* exts[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    };

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &feats;
    dci.pNext = &v12;
    dci.enabledExtensionCount = 5;
    dci.ppEnabledExtensionNames = exts;

    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(score.device, &dci, nullptr, &device) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    out.physicalDevice = score.device;
    out.device = device;
    out.graphicsQueueFamily = family;
    out.computeQueueFamily = family;
    out.transferQueueFamily = family;
    VkQueue q = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &q);
    out.graphicsQueue = q;
    out.computeQueue = q;
    out.transferQueue = q;
    return device;
}

PoolInitInfo initPoolCore() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM Chonk Buffer";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&ici, nullptr, &g_instance) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateInstance failed");
    }

    auto devices = vvm::enumerateDevices(g_instance);
    if (devices.empty()) {
        throw std::runtime_error("no Vulkan physical devices found");
    }

    PoolInitInfo out;
    bool saw_amd = false;
    for (auto& d : devices) {
        out.devices.push_back({std::string(d.props.deviceName), d.vendorID,
                               d.score, d.memProps.memoryHeaps[0].size});
        if (d.vendorID == 0x1002) saw_amd = true;
    }

    // Device selection policy. NOTE on the legacy default: scoring favors
    // large device-local heaps, which lets llvmpipe (software, 121 GiB host
    // heap) outscore the real AMD GPU on some machines - so PreferAmd is the
    // historical default for this integration. Set
    // CHONK_DEVICE_PREFERENCE=best|prefer_discrete for neutral behavior.
    const DevicePreference pref = devicePreferenceFromEnv();
    vvm::DeviceScore best = devices[0];
    for (auto& d : devices) {
        if (pref == DevicePreference::PreferAmd && saw_amd && d.vendorID != 0x1002) continue;
        if (pref == DevicePreference::PreferDiscrete && d.props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && best.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        if (d.score > best.score || (pref == DevicePreference::PreferAmd && saw_amd && best.vendorID != 0x1002)) best = d;
    }

    if (createDeviceForPool(best, g_devCfg) == VK_NULL_HANDLE) {
        throw std::runtime_error("vkCreateDevice failed");
    }

    // Configurable pool block size via env var (default 1GB; use 16/32GB for 262K
    // to reduce buddy fragmentation and leave contiguous space for dedicated
    // exportable allocations).
    size_t poolBlockGB = 1;
    const char* pBlock = getenv("CHONK_POOL_BLOCK_GB");
    if (pBlock) {
        double gb = atof(pBlock);
        if (gb >= 1.0) poolBlockGB = (size_t)gb;
    }

    // Configurable block sizes vector for multi-size routing (comma-separated GB values)
    std::vector<size_t> poolBlockSizesGB;
    const char* pBlockSizes = getenv("CHONK_POOL_BLOCK_SIZES_GB");
    if (pBlockSizes) {
        std::string str(pBlockSizes);
        size_t start = 0;
        size_t end = str.find(',');
        while (end != std::string::npos) {
            std::string token = str.substr(start, end - start);
            double gb = atof(token.c_str());
            if (gb >= 1.0) poolBlockSizesGB.push_back((size_t)gb);
            start = end + 1;
            end = str.find(',', start);
        }
        std::string token = str.substr(start);
        double gb = atof(token.c_str());
        if (gb >= 1.0) poolBlockSizesGB.push_back((size_t)gb);
    }

    vvm::PoolConfig cfg = vvm::PoolConfig::forAPU(128ull * 1024 * 1024 * 1024);
    cfg.blockSize = poolBlockGB * 1024ull * 1024 * 1024;
    if (!poolBlockSizesGB.empty()) {
        cfg.blockSizes.clear();
        for (size_t gb : poolBlockSizesGB) {
            cfg.blockSizes.push_back(gb * 1024ull * 1024 * 1024);
        }
    }
    cfg.maxBlocks = 64;
    cfg.maxHeapFraction = 0.0f;  // Disable budget check for Chonk Buffer training
    cfg.enableHostVisible = true;
    cfg.enableExternal = true;
    cfg.enableDeviceAddress = true;

    auto pool = vvm::UnifiedMemoryPool::create(g_devCfg, cfg);
    if (!pool) {
        throw std::runtime_error("UnifiedMemoryPool::create failed");
    }
    g_pool = std::make_unique<vvm::UnifiedMemoryPool>(std::move(*pool));

    out.deviceName = std::string(best.props.deviceName);
    out.heapBytes = best.memProps.memoryHeaps[0].size;
    return out;
}

std::vector<vvm::Allocation>& keptAllocations() {
    static std::vector<vvm::Allocation> kept;
    return kept;
}

void shutdownPoolCore() {
    // Kept allocations must be returned before the pool dies.
    for (auto& a : keptAllocations()) {
        if (g_pool) g_pool->deallocate(std::move(a));
    }
    keptAllocations().clear();
    g_pool.reset();
    if (g_device != VK_NULL_HANDLE) {
        vkDestroyDevice(g_device, nullptr);
        g_device = VK_NULL_HANDLE;
    }
    if (g_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(g_instance, nullptr);
        g_instance = VK_NULL_HANDLE;
    }
}

}  // namespace vvm_torch
