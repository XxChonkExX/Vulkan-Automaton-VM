// vulkanvm_pool_test.cpp
// Self-contained smoke test binding for the Chonk Buffer (UnifiedMemoryPool)
// on this machine. Creates a VkInstance + VkDevice from the best physical
// device, then exposes pool allocate/stats to Python.
//
// NOTE: the repo's python bindings call createInstance()/createDevice() which
// are declared nowhere and defined nowhere -- this binding inlines the device
// creation so we can actually exercise the pool.

#include <vulkan_vm/vulkan_vm.hpp>
#include <vulkan_vm/utils.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>

namespace py = pybind11;
using namespace vvm;

static VkInstance g_instance = VK_NULL_HANDLE;
static VkDevice g_device = VK_NULL_HANDLE;
static DeviceConfig g_devCfg;
static std::unique_ptr<UnifiedMemoryPool> g_pool;
static std::vector<vvm::Allocation> g_kept;

static VkDevice createDeviceForPool(const DeviceScore& score, DeviceConfig& out) {
    auto queues = findQueueFamilies(score.device);
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

static py::dict init() {
    if (g_pool) {
        throw std::runtime_error("already initialized");
    }
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM Chonk Buffer Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&ici, nullptr, &g_instance) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateInstance failed");
    }

    auto devices = enumerateDevices(g_instance);
    if (devices.empty()) {
        throw std::runtime_error("no Vulkan physical devices found");
    }
    py::list devlist;
    DeviceScore best = devices[0];
    bool saw_amd = false;
    for (auto& d : devices) {
        py::dict e;
        e["name"] = std::string(d.props.deviceName);
        e["vendor"] = d.vendorID;
        e["score"] = d.score;
        e["heap_mb"] = d.memProps.memoryHeaps[0].size >> 20;
        devlist.append(e);
        if (d.vendorID == 0x1002) saw_amd = true;
    }
    // Scoring favors large "device-local" heaps, which after a small BIOS
    // VRAM carve lets llvmpipe (software, 121 GiB host heap) outscore the
    // real RADV gfx1151 GPU. Prefer the AMD integrated GPU outright.
    for (auto& d : devices) {
        if (saw_amd && d.vendorID != 0x1002) continue;
        if (d.score > best.score || (saw_amd && best.vendorID != 0x1002)) best = d;
    }

    if (createDeviceForPool(best, g_devCfg) == VK_NULL_HANDLE) {
        throw std::runtime_error("vkCreateDevice failed");
    }

    PoolConfig cfg = PoolConfig::forAPU(128ull * 1024 * 1024 * 1024);
    cfg.blockSize = 1024ull * 1024 * 1024;
    cfg.maxBlocks = 64;
    cfg.enableHostVisible = true;
    cfg.enableExternal = true;
    cfg.enableDeviceAddress = true;

    auto pool = UnifiedMemoryPool::create(g_devCfg, cfg);
    if (!pool) {
        throw std::runtime_error("UnifiedMemoryPool::create failed");
    }
    g_pool = std::make_unique<UnifiedMemoryPool>(std::move(*pool));

    py::dict out;
    out["device"] = std::string(best.props.deviceName);
    out["heap_mb"] = best.memProps.memoryHeaps[0].size >> 20;
    out["devices"] = devlist;
    return out;
}

static py::dict alloc(size_t size, const std::string& name) {
    if (!g_pool) throw std::runtime_error("not initialized");
    vvm::AllocDesc desc;
    desc.size = size;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryUsage = vvm::MemoryUsage::GpuOnly;
    desc.mapped = true;
    desc.name = name;
    auto allocOpt = g_pool->allocate(desc);
    if (!allocOpt) throw std::runtime_error("allocation failed");
    auto a = std::move(*allocOpt);
    py::dict out;
    out["size"] = static_cast<uint64_t>(a.size);
    out["offset"] = static_cast<uint64_t>(a.offset);
    out["block"] = static_cast<uint64_t>(a.blockIndex);
    out["deviceAddress"] = static_cast<uint64_t>(a.deviceAddress);
    out["hostPtr"] = reinterpret_cast<uint64_t>(a.hostPtr);
    out["buffer"] = reinterpret_cast<uint64_t>(a.buffer);
    out["memory"] = reinterpret_cast<uint64_t>(a.memory);
    out["hostPtrValid"] = (a.hostPtr != nullptr);
    g_pool->deallocate(std::move(a));
    return out;
}

static py::dict allocKeep(size_t size, const std::string& name) {
    if (!g_pool) throw std::runtime_error("not initialized");
    vvm::AllocDesc desc;
    desc.size = size;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryUsage = vvm::MemoryUsage::GpuOnly;
    desc.mapped = true;
    desc.name = name;
    auto allocOpt = g_pool->allocate(desc);
    if (!allocOpt) throw std::runtime_error("allocation failed");
    auto a = std::move(*allocOpt);
    py::dict out;
    out["size"] = static_cast<uint64_t>(a.size);
    out["offset"] = static_cast<uint64_t>(a.offset);
    out["block"] = static_cast<uint64_t>(a.blockIndex);
    out["deviceAddress"] = static_cast<uint64_t>(a.deviceAddress);
    out["hostPtr"] = reinterpret_cast<uint64_t>(a.hostPtr);
    out["buffer"] = reinterpret_cast<uint64_t>(a.buffer);
    out["memory"] = reinterpret_cast<uint64_t>(a.memory);
    out["hostPtrValid"] = (a.hostPtr != nullptr);
    g_kept.push_back(std::move(a));
    return out;
}

static py::dict allocExport(size_t size, const std::string& name) {
    if (!g_pool) throw std::runtime_error("not initialized");
    vvm::AllocDesc desc;
    desc.size = size;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryUsage = vvm::MemoryUsage::GpuOnly;
    desc.mapped = true;
    desc.exportable = true;
    desc.name = name;
    auto allocOpt = g_pool->allocate(desc);
    if (!allocOpt) throw std::runtime_error("allocation failed");
    auto a = std::move(*allocOpt);
    auto info = g_pool->exportMemory(a, vvm::ExternalHandleType::OpaqueFd);
    if (!info) {
        g_pool->deallocate(std::move(a));
        throw std::runtime_error("exportMemory failed");
    }
    int fd = info->handle.release();
    py::dict out;
    out["size"] = static_cast<uint64_t>(a.size);
    out["offset"] = static_cast<uint64_t>(a.offset);
    out["block"] = static_cast<uint64_t>(a.blockIndex);
    out["deviceAddress"] = static_cast<uint64_t>(a.deviceAddress);
    out["hostPtr"] = reinterpret_cast<uint64_t>(a.hostPtr);
    out["buffer"] = reinterpret_cast<uint64_t>(a.buffer);
    out["memory"] = reinterpret_cast<uint64_t>(a.memory);
    out["hostPtrValid"] = (a.hostPtr != nullptr);
    out["fd"] = fd;
    g_kept.push_back(std::move(a));
    return out;
}

static py::dict stats() {
    if (!g_pool) throw std::runtime_error("not initialized");
    auto s = g_pool->getStats();
    py::dict out;
    out["totalAllocated"] = static_cast<uint64_t>(s.totalAllocated);
    out["totalUsed"] = static_cast<uint64_t>(s.totalUsed);
    out["totalFree"] = static_cast<uint64_t>(s.totalFree);
    out["largestFreeBlock"] = static_cast<uint64_t>(s.largestFreeBlock);
    out["fragmentationRatio"] = s.fragmentationRatio;
    out["blockCount"] = s.blockCount;
    out["allocationCount"] = s.allocationCount;
    return out;
}

static void shutdown() {
    for (auto& a : g_kept) {
        g_pool->deallocate(std::move(a));
    }
    g_kept.clear();
    g_pool.reset();
    if (g_device) vkDestroyDevice(g_device, nullptr);
    if (g_instance) vkDestroyInstance(g_instance, nullptr);
    g_device = VK_NULL_HANDLE;
    g_instance = VK_NULL_HANDLE;
}

PYBIND11_MODULE(vulkanvm_pool_test, m) {
    m.doc() = "Chonk Buffer (UnifiedMemoryPool) smoke test";
    m.def("init", &init, "create instance/device/pool");
    m.def("alloc", &alloc, py::arg("size"), py::arg("name") = "");
    m.def("alloc_keep", &allocKeep, py::arg("size"), py::arg("name") = "");
    m.def("alloc_export", &allocExport, py::arg("size"), py::arg("name") = "");
    m.def("stats", &stats);
    m.def("shutdown", &shutdown);
}
