// Shared host arena probe (VK_EXT_external_memory_host): one VirtualAlloc'd
// arena imported as VkDeviceMemory on EVERY discrete GPU. Verifies the
// cross-vendor zero-copy path: AMD XTX writes the arena via GPU DMA, NVIDIA
// Ti reads it via GPU DMA - no CPU memcpy in the loop. Reports per-device
// capability (ext, importable, alignment) + round-trip verdicts + timed
// 256 MiB legs. The Win dma-buf-equivalent candidate that needs no D3D12
// stack and no kernel code. Requires instance 1.3 (1.0 zeroes queries).
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace vvm;

namespace {

struct ArenaDevice {
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkDeviceMemory arenaMem = VK_NULL_HANDLE;
    VkBuffer arenaBuf = VK_NULL_HANDLE;
    VkDeviceMemory stageMem = VK_NULL_HANDLE;
    VkBuffer stageBuf = VK_NULL_HANDLE;
    void* stagePtr = nullptr;
    uint32_t importType = UINT32_MAX;
    VkDeviceSize minAlign = 0;
    bool importable = false;
    std::string name;
};

bool createDeviceFor(const DeviceScore& score, const std::string& name, ArenaDevice& out) {
    out.name = name;
    // Capability queries BEFORE device creation (physical-device level).
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp{};
    hp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &hp;
    vkGetPhysicalDeviceProperties2(score.device, &p2);
    out.minAlign = hp.minImportedHostPointerAlignment;

    VkPhysicalDeviceExternalBufferInfo eb{};
    eb.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    eb.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    eb.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkExternalBufferProperties bp{};
    bp.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(score.device, &eb, &bp);
    out.importable = (bp.externalMemoryProperties.externalMemoryFeatures &
                      VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;

    auto queues = findQueueFamilies(score.device);
    if (!queues.transfer && !queues.graphics) return false;
    out.queueFamily = queues.transfer.value_or(queues.graphics.value());
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = out.queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    const char* exts[] = { "VK_EXT_external_memory_host" };
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = exts;
    if (vkCreateDevice(score.device, &dci, nullptr, &out.device) != VK_SUCCESS) return false;
    VkQueue q = VK_NULL_HANDLE;
    vkGetDeviceQueue(out.device, out.queueFamily, 0, &q);
    out.queue = q;
    return out.queue != VK_NULL_HANDLE;
}

// Import [arena, arena+size) as VkDeviceMemory + bind a TRANSFER|STORAGE
// buffer to it. Returns false when the driver refuses (verdict, logged).
bool importArena(ArenaDevice& d, void* arena, VkDeviceSize size, VkInstance instance) {
    // Resolved dynamically: the loader import lib in this link context does
    // not carry every extension export (found 2026-09); vkGetInstanceProcAddr
    // is the robust route (Vulkan 1.1+ loaders always dispatch it).
    using PFN_GetMemHostPtrProps = PFN_vkGetMemoryHostPointerPropertiesEXT;
    auto getProps = reinterpret_cast<PFN_GetMemHostPtrProps>(
        vkGetInstanceProcAddr(instance, "vkGetMemoryHostPointerPropertiesEXT"));
    if (!getProps) {
        std::cerr << "  loader lacks vkGetMemoryHostPointerPropertiesEXT\n";
        return false;
    }
    VkMemoryHostPointerPropertiesEXT mpp{};
    mpp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    if (getProps(d.device,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, arena, &mpp) != VK_SUCCESS ||
        mpp.memoryTypeBits == 0) {
        std::cerr << "  " << d.name << ": host pointer props refused\n";
        return false;
    }
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < 32; ++i) {
        if (mpp.memoryTypeBits & (1u << i)) { mt = i; break; }
    }
    if (mt == UINT32_MAX) return false;
    d.importType = mt;

    VkImportMemoryHostPointerInfoEXT imp{};
    imp.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = arena;
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = &imp;
    ai.allocationSize = size;
    ai.memoryTypeIndex = mt;
    if (vkAllocateMemory(d.device, &ai, nullptr, &d.arenaMem) != VK_SUCCESS) {
        std::cerr << "  " << d.name << ": vkAllocateMemory(import) failed\n";
        return false;
    }
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(d.device, &bi, nullptr, &d.arenaBuf) != VK_SUCCESS ||
        vkBindBufferMemory(d.device, d.arenaBuf, d.arenaMem, 0) != VK_SUCCESS) {
        std::cerr << "  " << d.name << ": arena buffer create/bind failed\n";
        return false;
    }
    return true;
}

// Cached(+coherent) host-visible staging buffer, raw Vulkan (probe is
// pool-free by design; the SharedHostArena library class comes after).
bool makeStaging(ArenaDevice& d, const DeviceScore& score, VkDeviceSize size) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(score.device, &mp);
    uint32_t mt = UINT32_MAX;
    for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
        const VkMemoryPropertyFlags f = mp.memoryTypes[t].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) { mt = t; break; }
    }
    if (mt == UINT32_MAX) {
        for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
            const VkMemoryPropertyFlags f = mp.memoryTypes[t].propertyFlags;
            if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mt = t; break; }
        }
    }
    if (mt == UINT32_MAX) return false;
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = size;
    ai.memoryTypeIndex = mt;
    if (vkAllocateMemory(d.device, &ai, nullptr, &d.stageMem) != VK_SUCCESS) return false;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(d.device, &bi, nullptr, &d.stageBuf) != VK_SUCCESS ||
        vkBindBufferMemory(d.device, d.stageBuf, d.stageMem, 0) != VK_SUCCESS ||
        vkMapMemory(d.device, d.stageMem, 0, VK_WHOLE_SIZE, 0, &d.stagePtr) != VK_SUCCESS) {
        return false;
    }
    return true;
}

// One-shot device-local copy via a transient command buffer (mirrors
// UnifiedMemoryPool::copyBuffer; the probe is pool-free by design).
bool rawCopy(VkDevice device, VkQueue queue, uint32_t family,
             VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandPoolCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cp.queueFamilyIndex = family;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device, &cp, nullptr, &pool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cba{};
    cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = pool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &cba, &cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(device, pool, nullptr);
        return false;
    }
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(device, &fi, nullptr, &fence);
    bool ok = vkQueueSubmit(queue, 1, &si, fence) == VK_SUCCESS &&
              vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
    return ok;
}

void teardown(ArenaDevice& d) {
    if (d.device == VK_NULL_HANDLE) return;
    if (d.arenaBuf) vkDestroyBuffer(d.device, d.arenaBuf, nullptr);
    if (d.arenaMem) vkFreeMemory(d.device, d.arenaMem, nullptr);
    if (d.stageBuf) vkDestroyBuffer(d.device, d.stageBuf, nullptr);
    if (d.stageMem) {
        if (d.stagePtr) vkUnmapMemory(d.device, d.stageMem);
        vkFreeMemory(d.device, d.stageMem, nullptr);
    }
    vkDestroyDevice(d.device, nullptr);
}

} // namespace

int main() {
    int failures = 0;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM SharedArena Probe";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        std::cerr << "FAIL: vkCreateInstance\n";
        return 1;
    }
    auto devices = enumerateDevices(instance);
    std::cout << "Physical devices: " << devices.size() << "\n";

    // Arena: 64 MiB (a multiple of any sane import alignment; VirtualAlloc
    // base is 64 KB-aligned on Windows).
    const VkDeviceSize kArena = 64ull * 1024 * 1024;
    void* arena = VirtualAlloc(nullptr, static_cast<SIZE_T>(kArena),
                               MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!arena) {
        std::cerr << "FAIL: VirtualAlloc arena\n";
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    // Device + import + staging per DISCRETE GPU (AMD + NVIDIA + Intel).
    std::vector<ArenaDevice> ar;
    for (size_t i = 0; i < devices.size(); ++i) {
        const bool discrete =
            devices[i].props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        if (!discrete) continue;
        const bool isNv = devices[i].vendorID == 0x10DE;
        const bool isAmd = devices[i].vendorID == 0x1002;
        const bool isIntel = devices[i].vendorID == 0x8086;
        if (!isNv && !isAmd && !isIntel) continue;
        ArenaDevice d;
        if (!createDeviceFor(devices[i], devices[i].props.deviceName, d)) {
            std::cerr << "  " << devices[i].props.deviceName << ": device creation failed\n";
            continue;
        }
        std::cout << "  [" << d.name << "] minImportedHostPointerAlignment="
                  << d.minAlign << " importable=" << (d.importable ? "yes" : "NO") << "\n";
        if (!d.importable) {
            std::cout << "    (driver refuses HOST_ALLOCATION import - reported, not failed)\n";
            teardown(d);
            continue;
        }
        if (!importArena(d, arena, kArena, instance)) {
            ++failures;
            teardown(d);
            continue;
        }
        if (!makeStaging(d, devices[i], kArena)) {
            std::cerr << "  " << d.name << ": staging alloc/bind/map failed\n";
            ++failures;
            teardown(d);
            continue;
        }
        ar.push_back(std::move(d));
    }

    if (ar.empty()) {
        std::cout << "\nVERDICT: no GPUs accept HOST_ALLOCATION import - the "
                     "shared-arena path is out on this driver set\n";
        VirtualFree(arena, 0, MEM_RELEASE);
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

    // Pattern tests per device: GPU reads arena (CPU wrote it), GPU writes
    // arena (CPU reads it back). Both directions through the import.
    std::vector<uint8_t> pattern(static_cast<size_t>(kArena));
    for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = static_cast<uint8_t>(i * 7 + 3);
    for (auto& d : ar) {
        std::memcpy(arena, pattern.data(), pattern.size());
        if (!rawCopy(d.device, d.queue, d.queueFamily, d.arenaBuf, d.stageBuf, kArena) ||
            std::memcmp(d.stagePtr, pattern.data(), pattern.size()) != 0) {
            std::cerr << "  " << d.name << ": GPU-read-arena FAIL\n";
            ++failures;
        } else {
            std::cout << "  " << d.name << ": GPU-read-arena PASS\n";
        }
        std::memcpy(d.stagePtr, pattern.data(), pattern.size());
        std::memset(arena, 0, pattern.size());
        if (!rawCopy(d.device, d.queue, d.queueFamily, d.stageBuf, d.arenaBuf, kArena) ||
            std::memcmp(arena, pattern.data(), pattern.size()) != 0) {
            std::cerr << "  " << d.name << ": GPU-write-arena FAIL\n";
            ++failures;
        } else {
            std::cout << "  " << d.name << ": GPU-write-arena PASS\n";
        }
    }

    // THE cross-GPU zero-copy test: GPU0 DMA -> arena, GPU1 DMA <- arena.
    // No CPU memcpy in the loop (the arena IS the shared medium). Both
    // directions + byte verification.
    if (ar.size() >= 2) {
        auto& a = ar[0];
        auto& b = ar[1];
        std::cout << "\nzero-copy cross-GPU: " << a.name << " -> " << b.name << "\n";
        std::memset(a.stagePtr, 0, pattern.size());
        std::memcpy(a.stagePtr, pattern.data(), pattern.size());
        std::memset(arena, 0, pattern.size());
        auto t0 = std::chrono::steady_clock::now();
        const bool w = rawCopy(a.device, a.queue, a.queueFamily, a.stageBuf, a.arenaBuf, kArena);
        const bool r = w && rawCopy(b.device, b.queue, b.queueFamily, b.arenaBuf, b.stageBuf, kArena);
        auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        if (!w || !r || std::memcmp(b.stagePtr, pattern.data(), pattern.size()) != 0) {
            std::cerr << "  FAIL (write=" << w << " read=" << r << ")\n";
            ++failures;
        } else {
            std::cout << "  PASS 64 MiB GPU->arena->GPU in " << sec << " s ("
                      << (double)kArena / (1024.0 * 1024.0 * 1024.0) / sec
                      << " GiB/s, zero CPU memcpy)\n";
        }
        std::cout << "reverse: " << b.name << " -> " << a.name << "\n";
        std::memset(b.stagePtr, 0, pattern.size());
        std::memcpy(b.stagePtr, pattern.data(), pattern.size());
        std::memset(arena, 0, pattern.size());
        const bool w2 = rawCopy(b.device, b.queue, b.queueFamily, b.stageBuf, b.arenaBuf, kArena);
        const bool r2 = w2 && rawCopy(a.device, a.queue, a.queueFamily, a.arenaBuf, a.stageBuf, kArena);
        if (!w2 || !r2 || std::memcmp(a.stagePtr, pattern.data(), pattern.size()) != 0) {
            std::cerr << "  FAIL (write=" << w2 << " read=" << r2 << ")\n";
            ++failures;
        } else {
            std::cout << "  PASS (zero CPU memcpy)\n";
        }

        // Timed 256 MiB legs (4x the arena slab via chunked loops, one sync
        // per leg) vs the staged path's 2.8-3.3 GiB/s.
        const VkDeviceSize kBig = 256ull * 1024 * 1024;
        std::cout << "timed 256 MiB " << a.name << " -> " << b.name << "\n";
        auto t2 = std::chrono::steady_clock::now();
        bool okBig = true;
        for (VkDeviceSize off = 0; off < kBig && okBig; off += kArena) {
            okBig = rawCopy(a.device, a.queue, a.queueFamily, a.stageBuf, a.arenaBuf, kArena) &&
                    rawCopy(b.device, b.queue, b.queueFamily, b.arenaBuf, b.stageBuf, kArena);
        }
        auto t3 = std::chrono::steady_clock::now();
        const double sec2 = std::chrono::duration<double>(t3 - t2).count();
        std::cout << "  " << (okBig ? "OK" : "FAIL") << " 256 MiB in " << sec2 << " s = "
                  << (double)kBig / (1024.0 * 1024.0 * 1024.0) / sec2
                  << " GiB/s (staging fill + drain each leg)\n";
        if (!okBig) ++failures;
    }

    // Library integration: UnifiedMemoryPool::importMemoryHostPointer over a
    // probe device (created WITH the extension). Verifies the pool-level
    // import path + a pool copyBuffer through the arena.
    {
        auto& d = ar[0];
        DeviceConfig cfg{};
        // Physical handle for the SAME device the probe used: re-find it.
        for (size_t i = 0; i < devices.size(); ++i) {
            if (devices[i].props.deviceName == d.name) {
                cfg.physicalDevice = devices[i].device;
                break;
            }
        }
        cfg.device = d.device;
        cfg.graphicsQueueFamily = d.queueFamily;
        cfg.computeQueueFamily = d.queueFamily;
        cfg.transferQueueFamily = d.queueFamily;
        cfg.graphicsQueue = d.queue;
        cfg.computeQueue = d.queue;
        cfg.transferQueue = d.queue;
        PoolConfig pcfg;
        pcfg.blockSize = 64ull * 1024ull * 1024ull;
        pcfg.maxBlocks = 4;
        pcfg.enableHostVisible = true;
        pcfg.enableExternal = false;   // ext already on the device; skip checks
        auto pool = UnifiedMemoryPool::create(cfg, pcfg);
        if (!pool.has_value()) {
            std::cerr << "FAIL: pool create for arena import\n";
            ++failures;
        } else {
            auto imported = pool->importMemoryHostPointer(
                arena, kArena,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            if (!imported.has_value()) {
                std::cerr << "FAIL: pool-level importMemoryHostPointer\n";
                ++failures;
            } else {
                std::cout << "pool-level import: " << imported->size / (1024*1024)
                          << " MB (flags 0x" << std::hex << imported->memoryFlags
                          << std::dec << ", dedicated=" << (imported->blockIndex == UINT32_MAX ? "yes" : "no")
                          << ")\n";
                // Staging via the pool (cached flags), fill, pool copy
                // staging -> imported arena, CPU verify.
                auto st = pool->allocate(kArena,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                if (!st.has_value() || !st->hostPtr) {
                    std::cerr << "FAIL: pool staging alloc\n";
                    ++failures;
                } else {
                    std::memcpy(st->hostPtr, pattern.data(), pattern.size());
                    std::memset(arena, 0, pattern.size());
                    if (!pool->copyBuffer(*st, *imported, 0, 0, kArena) ||
                        std::memcmp(arena, pattern.data(), pattern.size()) != 0) {
                        std::cerr << "FAIL: pool copy staging->arena\n";
                        ++failures;
                    } else {
                        std::cout << "pool copyBuffer staging->arena: PASS\n";
                    }
                    pool->deallocate(std::move(*st));
                }
                pool->deallocate(std::move(*imported));
                const auto stats = pool->getStats();
                std::cout << "after dealloc: dedicated=" << stats.dedicatedCount << "\n";
            }
        }
    }

    // Manager-level integration: MultiGPUPoolManager::createSharedArena
    // (manager-owned VirtualAlloc) + copyDeviceToDeviceArena (two DMA legs,
    // no memcpy) + teardown hygiene (imports freed by pool dtors, arena
    // pages freed last).
    {
        // Rebuild DeviceConfigs for the two importing probe devices.
        std::vector<DeviceConfig> cfgs;
        for (const auto& d : ar) {
            for (size_t i = 0; i < devices.size(); ++i) {
                if (devices[i].props.deviceName == d.name) {
                    DeviceConfig cfg{};
                    cfg.physicalDevice = devices[i].device;
                    cfg.device = d.device;
                    cfg.graphicsQueueFamily = d.queueFamily;
                    cfg.computeQueueFamily = d.queueFamily;
                    cfg.transferQueueFamily = d.queueFamily;
                    cfg.graphicsQueue = d.queue;
                    cfg.computeQueue = d.queue;
                    cfg.transferQueue = d.queue;
                    cfgs.push_back(cfg);
                    break;
                }
            }
        }
        if (cfgs.size() == ar.size() && cfgs.size() >= 2) {
            PoolConfig pcfg;
            pcfg.blockSize = 64ull * 1024ull * 1024ull;
            pcfg.maxBlocks = 4;
            pcfg.enableHostVisible = true;
            pcfg.enableExternal = false;
            auto manager = MultiGPUPoolManager::create(cfgs, pcfg, 0);
            if (!manager) {
                std::cerr << "FAIL: manager create for arena test\n";
                ++failures;
            } else {
                // 256 MiB manager-owned arena (VirtualAlloc inside).
                const VkDeviceSize kMgr = 256ull * 1024 * 1024;
                if (!manager->createSharedArena(kMgr)) {
                    std::cerr << "FAIL: manager createSharedArena\n";
                    ++failures;
                } else {
                    std::cout << "manager arena: " << manager->arenaSize() / (1024*1024)
                              << " MB, pointer " << manager->arenaPointer() << "\n";
                    int importedCount = 0;
                    for (size_t i = 0; i < manager->getInstances().size(); ++i) {
                        if (manager->arenaBuffer((uint32_t)i) != VK_NULL_HANDLE) ++importedCount;
                    }
                    std::cout << "  imports on " << importedCount << "/"
                              << manager->getInstances().size() << " instance(s)\n";
                    if (importedCount < 2) {
                        std::cerr << "FAIL: fewer than 2 manager imports\n";
                        ++failures;
                    } else {
                        // Alloc 64 MiB on each device; XTX -> arena -> Ti
                        // via copyDeviceToDeviceArena, byte verify.
                        const VkDeviceSize kLeg = 64ull * 1024 * 1024;
                        const VkBufferUsageFlags kUsage =
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                        auto s0 = manager->getPool(0).allocate(kLeg, kUsage);
                        auto s1 = manager->getPool(1).allocate(kLeg, kUsage);
                        auto h0 = manager->getPool(0).allocate(kLeg,
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                        auto h1 = manager->getPool(1).allocate(kLeg,
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                        if (!s0 || !s1 || !h0 || !h1 || !h0->hostPtr || !h1->hostPtr) {
                            std::cerr << "FAIL: manager-path allocs\n";
                            ++failures;
                        } else {
                            std::memcpy(h0->hostPtr, pattern.data(), pattern.size());
                            if (!manager->getPool(0).copyBuffer(*h0, *s0, 0, 0, kLeg)) {
                                std::cerr << "FAIL: stage-in\n";
                                ++failures;
                            } else {
                                auto t0 = std::chrono::steady_clock::now();
                                const bool ok = manager->copyDeviceToDeviceArena(0, 1, *s0, *s1, 0, 0, kLeg);
                                auto t1 = std::chrono::steady_clock::now();
                                const double sec = std::chrono::duration<double>(t1 - t0).count();
                                std::memset(h1->hostPtr, 0, pattern.size());
                                if (!ok || !manager->getPool(1).copyBuffer(*s1, *h1, 0, 0, kLeg) ||
                                    std::memcmp(h1->hostPtr, pattern.data(), pattern.size()) != 0) {
                                    std::cerr << "FAIL: arena copy or verify (ok=" << ok << ")\n";
                                    ++failures;
                                } else {
                                    std::cout << "manager arena copy 0->1: PASS 64 MiB in "
                                              << sec << " s ("
                                              << (double)kLeg / (1024.0*1024.0*1024.0) / sec
                                              << " GiB/s, two DMA legs)\n";
                                }
                            }
                            manager->getPool(1).deallocate(std::move(*h1));
                            manager->getPool(0).deallocate(std::move(*h0));
                            manager->getPool(1).deallocate(std::move(*s1));
                            manager->getPool(0).deallocate(std::move(*s0));
                        }
                    }
                }
                // Manager dtor: arena pages freed AFTER pool dtors free the
                // imports (reverse-destruction order). Reset before the
                // caller destroys its VkDevices (multi_gpu_test contract).
                manager.reset();
                std::cout << "manager teardown: clean\n";
            }
        }
    }

    std::cout << "\n=== " << (failures == 0 ? "SHARED ARENA PROBE CLEAN" : "SHARED ARENA PROBE FAILURES")
              << " (" << failures << ") ===\n";
    VirtualFree(arena, 0, MEM_RELEASE);
    for (auto& d : ar) teardown(d);
    vkDestroyInstance(instance, nullptr);
    return failures == 0 ? 0 : 1;
}
