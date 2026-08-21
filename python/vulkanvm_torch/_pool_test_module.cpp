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

#include <hip/hip_runtime.h>

#include <iostream>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <unistd.h>

namespace py = pybind11;
using namespace vvm;

static VkInstance g_instance = VK_NULL_HANDLE;
static VkDevice g_device = VK_NULL_HANDLE;
static DeviceConfig g_devCfg;
static std::unique_ptr<UnifiedMemoryPool> g_pool;
static std::vector<vvm::Allocation> g_kept;
static py::dict* g_lastInitInfo = nullptr;  // heap-allocated: never destroyed
                                          // at exit (py::dict dtor needs the
                                          // interpreter alive)

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

static py::dict initPool() {
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
    
    PoolConfig cfg = PoolConfig::forAPU(128ull * 1024 * 1024 * 1024);
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

static py::dict init() {
    if (g_pool) {
        throw std::runtime_error("already initialized");
    }
    g_lastInitInfo = new py::dict(initPool());
    return *g_lastInitInfo;
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

static py::dict allocModelWeights(size_t size, const std::string& name) {
    if (!g_pool) throw std::runtime_error("not initialized");
    vvm::AllocDesc desc;
    desc.size = size;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryUsage = vvm::MemoryUsage::GpuOnly;
    desc.mapped = false;
    desc.exportable = false;
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

static py::dict allocOptimizerStates(size_t size, const std::string& name) {
    if (!g_pool) throw std::runtime_error("not initialized");
    vvm::AllocDesc desc;
    desc.size = size;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryUsage = vvm::MemoryUsage::GpuOnly;
    desc.mapped = false;
    desc.exportable = false;
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

static py::dict allocActivations(size_t size, const std::string& name) {
    if (!g_pool) throw std::runtime_error("not initialized");
    vvm::AllocDesc desc;
    desc.size = size;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryUsage = vvm::MemoryUsage::GpuOnly;
    desc.mapped = false;
    desc.exportable = false;
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

static py::dict allocHostVisible(size_t size, const std::string& name) {
    if (!g_pool) throw std::runtime_error("not initialized");
    vvm::AllocDesc desc;
    desc.size = size;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryUsage = vvm::MemoryUsage::CpuToGpu;
    desc.mapped = true;
    desc.exportable = false;
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

// ==========================================================================
// Pool-backed pluggable allocator for PyTorch/HIP.
//
// Replaces torch's HIP caching allocator backing with memory drawn from the
// Chonk Buffer: each segment torch requests is carved from a pool block
// (Vulkan allocation -> dma-buf export -> hipImportExternalMemory). Freed
// segments return to an internal freelist; fully-free blocks beyond a warm
// count are released back to the pool for reuse (KV cache growth, etc.).
// This keeps ONE allocator family over the unified heap instead of interleaved
// HIP segments and Vulkan BOs fragmenting the driver's GTT manager.
//
// Exported with C linkage; loaded by torch.cuda.memory.CUDAPluggableAllocator
// via the old-style 2-function ABI:
//     void* alloc_fn(ssize_t size, int device, void* stream);
//     void  free_fn(void* ptr, size_t size, void* stream);
// ==========================================================================

struct AllocBlock {
    vvm::Allocation alloc;
    int fd = -1;
    hipExternalMemory_t ext = nullptr;
    void* base = nullptr;
    size_t size = 0;
    size_t liveBytes = 0;
    // Free chunks as (offset, size), kept sorted by offset.
    std::vector<std::pair<size_t, size_t>> freeChunks;
};

struct ChonkAllocator {
    std::mutex mtx;
    std::vector<std::unique_ptr<AllocBlock>> blocks;
    std::unordered_map<void*, size_t> liveSizes;
    // Block retention policy:
    //   warmBlocks   - fully-free blocks are kept warm up to this many blocks
    //                  (avoids GTT churn from release/re-create ping-pong)
    //   maxBlocks    - hard cap; when a new block is needed and we are at the
    //                  cap, fully-free blocks are released first to make room
    //                  (bounded GTT commitment - the "better router")
    // Previously blocks were NEVER released (warm=512). That made the
    // monotonic growth of backward-attention temporaries (recompute
    // re-materializes attn weights that grow with kv_len; freed chunks are
    // always smaller than the next request) accumulate exact-fit blocks until
    // the driver's GTT ceiling refused vkAllocateMemory -> zero-storage tensor
    // -> "data is not allocated yet" crash. Bucket-rounded block sizes make a
    // single block absorb the whole growth curve up to its bucket size, and
    // release-on-cap keeps committed GTT bounded.
    size_t warmBlocks = 8;
    size_t maxBlocks = 24;
    size_t minBlocksOnOOM = 4;  // floor kept when releasing under pressure

    static constexpr size_t kAlign = 512;
    static constexpr size_t kMinBlock = 2ull * 1024 * 1024 * 1024;  // 2 GB
    static size_t minBlock() {
        const char* p = getenv("CHONK_MIN_BLOCK_GB");
        if (p) {
            double gb = atof(p);
            if (gb >= 1.0) return (size_t)(gb * 1024.0 * 1024.0 * 1024.0);
        }
        return kMinBlock;
    }
    static size_t envSize(const char* name, size_t def) {
        const char* p = getenv(name);
        if (p) {
            double v = atof(p);
            if (v >= 1.0) return (size_t)v;
        }
        return def;
    }
    // Bucket list (GB) parsed once from CHONK_POOL_BLOCK_SIZES_GB.
    // If the env var is set to "auto", buckets are generated as powers of
    // two from 1 GB up to the device VRAM size (queried at first call).
    // Otherwise, the comma-separated list is used as-is. New blocks are
    // rounded UP to the smallest bucket >= the request so a single block
    // absorbs the monotonic growth of recompute attention temporaries
    // (freed chunk merges back to the full bucket and serves the next,
    // slightly larger, request). Falls back to power-of-two rounding.
    static std::vector<size_t> buckets() {
        static std::vector<size_t> b = [] {
            std::vector<size_t> out;
            const char* p = getenv("CHONK_POOL_BLOCK_SIZES_GB");
            if (p) {
                std::string str(p);
                std::string firstToken = str.substr(0, str.find(','));
                if (firstToken == "auto") {
                    // Graduated ladder: 1 GB steps through 16 GB (where all
                    // transient training traffic lives - peak measured demand
                    // was 8 GiB + 16 MiB), then coarser rungs up to 128 GB
                    // for large model/optimizer-scale requests.
                    static const size_t kLadderGB[] = {
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128,
                    };
                    for (size_t gb : kLadderGB) {
                        out.push_back(gb * 1024ull * 1024ull * 1024ull);
                    }
                    fprintf(stderr, "[allocator] auto-buckets: graduated %zu-rung ladder, 1..128 GB\n",
                            sizeof(kLadderGB) / sizeof(kLadderGB[0]));
                } else {
                    // Parse comma-separated list
                    size_t start = 0;
                    for (;;) {
                        size_t end = str.find(',', start);
                        std::string token = str.substr(start, end == std::string::npos ? std::string::npos : end - start);
                        double gb_val = atof(token.c_str());
                        if (gb_val >= 1.0) out.push_back((size_t)(gb_val * 1024.0 * 1024.0 * 1024.0));
                        if (end == std::string::npos) break;
                        start = end + 1;
                    }
                }
            }
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
            return out;
        }();
        return b;
    }
    static size_t roundToBucket(size_t need) {
        for (size_t b : buckets()) {
            if (b >= need) return b;
        }
        // No bucket fits: round up to the next power of two (min 2 GB).
        size_t b = std::max(minBlock(), need);
        size_t p = 1;
        while (p < b) p <<= 1;
        return p;
    }
};

static ChonkAllocator g_allocator;
static FILE* g_allocLog = nullptr;

static void allocLog(const char* op, void* ptr, size_t sz) {
    if (!g_allocLog) {
        const char* p = getenv("CHONK_ALLOC_LOG");
        if (p) g_allocLog = fopen(p, "w");
    }
    if (g_allocLog) {
        fprintf(g_allocLog, "%s %p %zu\n", op, ptr, sz);
        fflush(g_allocLog);
    }
}

static void* hipImportFromFd(int fd, size_t size, hipExternalMemory_t* outExt) {
    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = size;
    hipExternalMemory_t ext = nullptr;
    if (hipImportExternalMemory(&ext, &desc) != hipSuccess) return nullptr;
    void* base = nullptr;
    hipExternalMemoryBufferDesc buf{};
    buf.offset = 0;
    buf.size = size;
    if (hipExternalMemoryGetMappedBuffer(&base, ext, &buf) != hipSuccess) {
        hipDestroyExternalMemory(ext);
        return nullptr;
    }
    *outExt = ext;
    return base;
}

static void allocatorDestroyBlock(AllocBlock* b);

static bool allocatorCreateBlock(size_t need) {
    if (!g_pool) return false;
    // Proven sizing: exact-fit above the min-block floor (the validated
    // 196K recipe sets CHONK_MIN_BLOCK_GB=16). A 16GB block absorbs the
    // entire recompute-attention growth curve (2.4GB -> 9.7GB): the freed
    // chunk merges back to the full block and the next, slightly larger,
    // request reuses it - no new blocks, flat GTT. Bucket rounding is NOT
    // applied to base allocations (it inflated them by ~10GB). It is only
    // used below as the pressure-relief retry escalation.
    size_t blockSize = std::max(ChonkAllocator::minBlock(),
                                (need + ChonkAllocator::kAlign - 1) / ChonkAllocator::kAlign * ChonkAllocator::kAlign);
    vvm::AllocDesc desc;
    desc.size = blockSize;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryUsage = vvm::MemoryUsage::GpuOnly;
    desc.mapped = true;
    desc.exportable = true;
    desc.name = "torch_segment";
    auto allocOpt = g_pool->allocate(desc);
    if (!allocOpt) {
        // Pressure relief: we are at (or near) the driver's ceiling. Release
        // fully-free blocks down to a small warm floor, then retry once
        // before giving up (auto-allocation under pressure).
        size_t before = g_allocator.blocks.size();
        for (auto it = g_allocator.blocks.begin(); it != g_allocator.blocks.end();) {
            if ((*it)->liveBytes == 0 && g_allocator.blocks.size() > g_allocator.minBlocksOnOOM) {
                allocatorDestroyBlock(it->get());
                it = g_allocator.blocks.erase(it);
            } else {
                ++it;
            }
        }
        if (g_allocator.blocks.size() < before) {
            fprintf(stderr, "[allocator] released %zu empty block(s) under pressure; retrying %zu bytes\n",
                    before - g_allocator.blocks.size(), need);
            allocOpt = g_pool->allocate(desc);
        }
        if (!allocOpt) {
            // Second escalation: re-round the block up to the nearest
            // configured bucket (CHONK_POOL_BLOCK_SIZES_GB; with the
            // graduated auto-ladder this is a tight fit, e.g. 8.02 GB ->
            // 9 GB rung). Bound the overshoot: escalating far past `need`
            // after a driver OOM is guaranteed to fail again and just
            // thrashes the allocator, so skip escalation when the next
            // rung exceeds need + slack.
            size_t bucketSize = ChonkAllocator::roundToBucket(need);
            size_t slack = ChonkAllocator::envSize("CHONK_ESCALATE_SLACK_GB", 2)
                           * 1024ull * 1024ull * 1024ull;
            if (bucketSize > blockSize && bucketSize <= need + slack) {
                fprintf(stderr, "[allocator] pressure: escalating %zu -> %zu bucket\n", blockSize, bucketSize);
                desc.size = bucketSize;
                allocOpt = g_pool->allocate(desc);
            }
        }
    }
    if (!allocOpt) return false;
    vvm::Allocation a = std::move(*allocOpt);
    auto info = g_pool->exportMemory(a, vvm::ExternalHandleType::OpaqueFd);
    if (!info) {
        g_pool->deallocate(std::move(a));
        return false;
    }
    int fd = info->handle.release();
    hipExternalMemory_t ext = nullptr;
    void* base = hipImportFromFd(fd, blockSize, &ext);
    if (!base) {
        g_pool->deallocate(std::move(a));
        return false;
    }
    auto block = std::make_unique<AllocBlock>();
    block->alloc = std::move(a);
    block->fd = fd;
    block->ext = ext;
    block->base = base;
    block->size = blockSize;
    block->freeChunks.push_back({0, blockSize});
    g_allocator.blocks.push_back(std::move(block));
    allocLog("B", base, blockSize);
    return true;
}

static void allocatorDestroyBlock(AllocBlock* b) {
    // After pool.shutdown() the HIP context is gone; skip HIP destruction
    // during teardown (leak the handles; the OS reclaims them).
    if (b->ext && g_pool) hipDestroyExternalMemory(b->ext);
    b->base = nullptr;
    b->ext = nullptr;
    if (b->fd >= 0) close(b->fd);
    b->fd = -1;
    if (g_pool) g_pool->deallocate(std::move(b->alloc));
}

static void allocatorMaybeReleaseEmptyBlock(AllocBlock* blk) {
    if (blk->liveBytes != 0) return;
    // Keep a warm set of empty blocks for slab reuse (avoids GTT churn from
    // release/re-create ping-pong), but never let the block count exceed the
    // hard cap: above it, empty blocks are returned to the pool so committed
    // GTT stays bounded (the old policy never released below 512 blocks, which
    // let monotonically-growing backward temporaries accumulate until the
    // driver refused vkAllocateMemory).
    size_t warm = ChonkAllocator::envSize("CHONK_ALLOCATOR_WARM_BLOCKS", g_allocator.warmBlocks);
    size_t maxb = ChonkAllocator::envSize("CHONK_ALLOCATOR_MAX_BLOCKS", g_allocator.maxBlocks);
    if (g_allocator.blocks.size() <= warm || g_allocator.blocks.size() <= maxb) return;
    for (auto bi = g_allocator.blocks.begin(); bi != g_allocator.blocks.end(); ++bi) {
        if (bi->get() == blk) {
            allocatorDestroyBlock(blk);
            g_allocator.blocks.erase(bi);
            return;
        }
    }
}

extern "C" void* chonk_allocator_alloc(ssize_t size, int device, void* stream) {
    (void)device; (void)stream;
    if (!g_pool) return nullptr;  // pool must be initialized before install
    std::lock_guard<std::mutex> lock(g_allocator.mtx);
    size_t aligned = ((size_t)size + ChonkAllocator::kAlign - 1) &
                     ~(ChonkAllocator::kAlign - 1);
    if (aligned == 0) aligned = ChonkAllocator::kAlign;  // hipMalloc(0) semantics

    // First fit across existing blocks (alignment-aware).
    for (auto& blk : g_allocator.blocks) {
        for (auto it = blk->freeChunks.begin(); it != blk->freeChunks.end(); ++it) {
            size_t off = it->first;
            size_t chunkSz = it->second;
            size_t alignedOff = (off + ChonkAllocator::kAlign - 1) &
                                ~(ChonkAllocator::kAlign - 1);
            size_t headSlack = alignedOff - off;
            if (chunkSz < headSlack + aligned) continue;
            void* ptr = (char*)blk->base + alignedOff;
            blk->liveBytes += aligned;
            g_allocator.liveSizes[ptr] = aligned;
            size_t tail = chunkSz - headSlack - aligned;
            if (headSlack > 0) {
                it->second = headSlack;  // keep head slack as free
                if (tail > 0) {
                    blk->freeChunks.insert(it + 1, {alignedOff + aligned, tail});
                }
            } else if (tail > 0) {
                it->first = alignedOff + aligned;
                it->second = tail;
            } else {
                blk->freeChunks.erase(it);
            }
            allocLog("A", ptr, aligned);
            return ptr;
        }
    }

    // No fit: create a new block sized for the request.
    allocLog("N", nullptr, aligned);
    if (!allocatorCreateBlock(aligned)) return nullptr;
    auto& blk = g_allocator.blocks.back();
    auto it = blk->freeChunks.begin();
    blk->liveBytes += aligned;
    void* ptr = (char*)blk->base;
    g_allocator.liveSizes[ptr] = aligned;
    if (blk->size > aligned) {
        it->first = aligned;
        it->second = blk->size - aligned;
    } else {
        blk->freeChunks.erase(it);
    }
    allocLog("A", ptr, aligned);
    return ptr;
}

extern "C" void chonk_allocator_free(void* ptr, size_t size, void* stream) {
    (void)stream;
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(g_allocator.mtx);
    size_t sz = size;
    auto lsIt = g_allocator.liveSizes.find(ptr);
    if (lsIt != g_allocator.liveSizes.end()) {
        sz = lsIt->second;
        g_allocator.liveSizes.erase(lsIt);
    }
    if (sz == 0) return;

    for (auto& blk : g_allocator.blocks) {
        uintptr_t b = (uintptr_t)blk->base;
        uintptr_t p = (uintptr_t)ptr;
        if (p < b || p + sz > b + blk->size) continue;
        size_t coff = p - b;
        blk->liveBytes -= sz;
        auto& fc = blk->freeChunks;
        auto pos = std::lower_bound(fc.begin(), fc.end(), coff,
            [](const std::pair<size_t, size_t>& c, size_t v) { return c.first < v; });
        // Overlap guards: if the region is already covered by neighbors,
        // this is a double free -- ignore it rather than corrupt the list.
        if (pos != fc.begin()) {
            auto prev = pos - 1;
            if (prev->first + prev->second >= coff + sz) {
                blk->liveBytes += sz;
                allocLog("D", ptr, sz);
                return;
            }
        }
        if (pos != fc.end() && pos->first <= coff) {
            blk->liveBytes += sz;
            allocLog("D", ptr, sz);
            return;
        }
        // Merge with previous chunk.
        if (pos != fc.begin()) {
            auto prev = pos - 1;
            if (prev->first + prev->second == coff) {
                prev->second += sz;
                if (pos != fc.end() && prev->first + prev->second == pos->first) {
                    prev->second += pos->second;
                    fc.erase(pos);
                }
                allocatorMaybeReleaseEmptyBlock(blk.get());
                allocLog("F", ptr, sz);
                return;
            }
        }
        // Merge with next chunk.
        if (pos != fc.end() && coff + sz == pos->first) {
            pos->first = coff;
            pos->second += sz;
        } else {
            fc.insert(pos, {coff, sz});
        }
        allocatorMaybeReleaseEmptyBlock(blk.get());
        allocLog("F", ptr, sz);
        return;
    }
    // Unknown pointer: ignore (torch may free pointers from other allocators).
    allocLog("U", ptr, size);
    fprintf(stderr, "[allocator] WARN: free of UNKNOWN ptr=%p size=%zu\n", ptr, size);
}

PYBIND11_MODULE(vulkanvm_pool_test, m) {
    m.doc() = "Chonk Buffer (UnifiedMemoryPool) smoke test";
    m.def("init", &init, "create instance/device/pool");
    m.def("alloc", &alloc, py::arg("size"), py::arg("name") = "");
    m.def("alloc_keep", &allocKeep, py::arg("size"), py::arg("name") = "");
    m.def("alloc_export", &allocExport, py::arg("size"), py::arg("name") = "");
    m.def("alloc_model_weights", &allocModelWeights, py::arg("size"), py::arg("name") = "");
    m.def("alloc_optimizer_states", &allocOptimizerStates, py::arg("size"), py::arg("name") = "");
    m.def("alloc_activations", &allocActivations, py::arg("size"), py::arg("name") = "");
    m.def("alloc_host_visible", &allocHostVisible, py::arg("size"), py::arg("name") = "");
    m.def("stats", &stats);
    m.def("info", []() {
        if (!g_lastInitInfo) throw std::runtime_error("not initialized");
        return *g_lastInitInfo;
    });
    m.def("shutdown", &shutdown);
}
