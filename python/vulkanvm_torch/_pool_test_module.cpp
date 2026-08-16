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

    PoolConfig cfg = PoolConfig::forAPU(128ull * 1024 * 1024 * 1024);
    cfg.blockSize = 1024ull * 1024 * 1024;
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
    // Never release blocks back to the pool: every block is a dedicated
    // exportable VkDeviceMemory, so releasing only sends pages to the TTM
    // page pool and the next chunk re-creates fresh blocks (GTT churn ->
    // driver memory pressure). Keeping every block warm means torch's
    // empty_cache just returns chunks to the slab and the footprint stays
    // at the first-step peak (stable at ~88GB for the 131K run).
    size_t warmBlocks = 512;

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

static bool allocatorCreateBlock(size_t need) {
    if (!g_pool) return false;
    size_t blockSize = std::max(ChonkAllocator::minBlock(),
                                (need + ChonkAllocator::kAlign - 1) & ~(ChonkAllocator::kAlign - 1));
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
    if (g_allocator.blocks.size() <= g_allocator.warmBlocks) return;
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
