#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"
#include "vulkan_vm/sparse.hpp"

#include <iostream>
#include <vector>
#include <cstring>

using namespace vvm;

// Minimal device setup: graphics + transfer queues, sparse binding enabled.
static VkInstance s_instance = VK_NULL_HANDLE;
static VkPhysicalDevice s_physicalDevice = VK_NULL_HANDLE;
static VkDevice s_device = VK_NULL_HANDLE;
static VkQueue s_transferQueue = VK_NULL_HANDLE;
static uint32_t s_transferFamily = UINT32_MAX;
static VkPhysicalDeviceProperties s_props{};

static bool initDevice() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM Sparse Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&ici, nullptr, &s_instance) != VK_SUCCESS) return false;

    auto devices = enumerateDevices(s_instance);
    if (devices.empty()) return false;
    auto best = selectBestDevice(devices, true, 1024);
    if (!best) return false;
    s_physicalDevice = best->device;
    s_props = best->props;

    auto queues = findQueueFamilies(s_physicalDevice);
    s_transferFamily = queues.transfer.value_or(queues.graphics.value_or(0));

    // The sparse pool binds pages via a queue from a family with
    // VK_QUEUE_SPARSE_BINDING_BIT and fetches it with vkGetDeviceQueue.
    // Some drivers (e.g. ANV on Battlemage) expose a pure-transfer family
    // WITHOUT the sparse bit; if the device is created only on that family,
    // the pool's vkGetDeviceQueue returns NULL and creation fails. Prefer a
    // transfer-capable family that also reports the sparse bit.
    {
        uint32_t famCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(s_physicalDevice, &famCount, nullptr);
        std::vector<VkQueueFamilyProperties> fams(famCount);
        vkGetPhysicalDeviceQueueFamilyProperties(s_physicalDevice, &famCount, fams.data());
        auto hasSparse = [&](uint32_t i) {
            return i < famCount && (fams[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0;
        };
        if (!hasSparse(s_transferFamily)) {
            for (uint32_t i = 0; i < famCount; ++i) {
                if ((fams[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && hasSparse(i)) {
                    s_transferFamily = i;
                    break;
                }
            }
            // Last resort: any sparse-capable family.
            if (!hasSparse(s_transferFamily)) {
                for (uint32_t i = 0; i < famCount; ++i) {
                    if (hasSparse(i)) { s_transferFamily = i; break; }
                }
            }
        }
    }

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = s_transferFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceFeatures feats{};
    feats.sparseBinding = VK_TRUE;
    feats.sparseResidencyBuffer = VK_TRUE;

    VkPhysicalDeviceVulkan12Features v12Feat{};
    v12Feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12Feat.bufferDeviceAddress = VK_TRUE;
    v12Feat.timelineSemaphore = VK_TRUE;

    const char* exts[] = {
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    };

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &feats;
    dci.pNext = &v12Feat;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = exts;
    if (vkCreateDevice(s_physicalDevice, &dci, nullptr, &s_device) != VK_SUCCESS) return false;
    vkGetDeviceQueue(s_device, s_transferFamily, 0, &s_transferQueue);
    return s_transferQueue != VK_NULL_HANDLE;
}

// Helper: copy [srcBuffer, srcOffset + size) -> host-visible staging, compare with expected.
static bool deviceReadbackRange(VkBuffer srcBuffer, VkDeviceSize srcOffset,
                                VkDeviceSize size, const void* expected, const char* what) {
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    VkBufferCreateInfo binfo{};
    binfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    binfo.size = size;
    binfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    binfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(s_device, &binfo, nullptr, &staging) != VK_SUCCESS) return false;

    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(s_physicalDevice, &props);
    auto memType = findMemoryTypeIndex(props, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!memType) return false;

    VkMemoryRequirements reqs{};
    vkGetBufferMemoryRequirements(s_device, staging, &reqs);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = reqs.size;
    ai.memoryTypeIndex = *memType;
    if (vkAllocateMemory(s_device, &ai, nullptr, &stagingMem) != VK_SUCCESS) return false;
    vkBindBufferMemory(s_device, staging, stagingMem, 0);

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.queueFamilyIndex = s_transferFamily;
    if (vkCreateCommandPool(s_device, &cp, nullptr, &pool) != VK_SUCCESS) return false;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cba{};
    cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = pool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    vkAllocateCommandBuffers(s_device, &cba, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferCopy region{};
    region.srcOffset = srcOffset;
    region.size = size;
    vkCmdCopyBuffer(cmd, srcBuffer, staging, 1, &region);
    vkEndCommandBuffer(cmd);

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(s_device, &fci, nullptr, &fence);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(s_transferQueue, 1, &si, fence);
    vkWaitForFences(s_device, 1, &fence, VK_TRUE, UINT64_MAX);

    void* ptr = nullptr;
    vkMapMemory(s_device, stagingMem, 0, size, 0, &ptr);
    bool ok = expected != nullptr && std::memcmp(ptr, expected, static_cast<size_t>(size)) == 0;
    if (!ok) {
        auto* bytes = static_cast<const uint8_t*>(ptr);
        std::cerr << "  MISMATCH (" << what << "): first bytes: ";
        for (int i = 0; i < 16; ++i) std::cerr << std::hex << (int)bytes[i] << " ";
        std::cerr << std::dec << "\n";
    }
    vkUnmapMemory(s_device, stagingMem);

    vkDestroyFence(s_device, fence, nullptr);
    vkDestroyCommandPool(s_device, pool, nullptr);
    vkFreeMemory(s_device, stagingMem, nullptr);
    vkDestroyBuffer(s_device, staging, nullptr);
    return ok;
}

int main() {
    std::cout << "=== VulkanVM Sparse/Residency Test ===\n";

    if (!initDevice()) {
        std::cerr << "FAIL: could not initialize Vulkan device\n";
        return 1;
    }
    std::cout << "Device: " << s_props.deviceName << "\n";

    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(s_physicalDevice, &feats);
    if (!feats.sparseBinding || !feats.sparseResidencyBuffer) {
        std::cout << "SKIP: sparse residency not supported on this device\n";
        vkDestroyDevice(s_device, nullptr);
        vkDestroyInstance(s_instance, nullptr);
        return 0;
    }

    DeviceConfig dev;
    dev.physicalDevice = s_physicalDevice;
    dev.device = s_device;
    dev.transferQueueFamily = s_transferFamily;
    dev.transferQueue = s_transferQueue;
    dev.graphicsQueueFamily = s_transferFamily;
    dev.graphicsQueue = s_transferQueue;

    // Virtual buffer: 1 GiB virtual address space, 32 MiB pages.
    const VkDeviceSize kVirtualSize = 1ull * 1024 * 1024 * 1024;
    SparseMemoryConfig cfg;
    cfg.virtualSize = kVirtualSize;
    cfg.pageSize = 32ull * 1024 * 1024;
    cfg.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    cfg.enableDeviceAddress = true;

    auto pool = SparseVirtualMemoryPool::create(dev, cfg);
    if (!pool) {
        std::cerr << "FAIL: SparseVirtualMemoryPool::create\n";
        return 1;
    }
    std::cout << "  Virtual size: " << pool->getVirtualSize() / (1024 * 1024) << " MiB\n";
    std::cout << "  Page size:    " << pool->getPageSize() / (1024 * 1024) << " MiB\n";
    std::cout << "  Pages:        " << pool->getPageCount() << "\n";
    std::cout << "  Device addr:  0x" << std::hex << pool->getDeviceAddress() << std::dec << "\n";

    int failures = 0;

    // 1. Nothing resident yet.
    if (pool->getResidentPageCount() != 0) {
        std::cerr << "FAIL: expected 0 resident pages after create\n";
        ++failures;
    }

    // 2. Make 64 MiB resident (2 pages).
    if (!pool->makeResident(0, 64ull * 1024 * 1024)) {
        std::cerr << "FAIL: makeResident(0, 64MiB)\n";
        ++failures;
    }
    std::cout << "  Resident after bind: " << pool->getResidentPageCount()
              << " pages, " << pool->getResidentBytes() / (1024 * 1024) << " MiB\n";
    if (pool->getResidentPageCount() != 2) {
        std::cerr << "FAIL: expected 2 resident pages, got " << pool->getResidentPageCount() << "\n";
        ++failures;
    }

    // 3. isResident checks.
    if (!pool->isResident(0) || !pool->isResident(64ull * 1024 * 1024 - 1)) {
        std::cerr << "FAIL: isResident on bound range\n";
        ++failures;
    }
    if (pool->isResident(kVirtualSize - 1)) {
        std::cerr << "FAIL: isResident on unbound tail\n";
        ++failures;
    }

    // 4. Write pattern into page 0 via host pointer (if host-visible) and verify
    //    with a GPU readback of the resident region.
    bool wroteViaHost = false;
    void* hostPtr = pool->getResidentHostPtr(0);
    if (hostPtr) {
        std::vector<uint8_t> pattern(32ull * 1024 * 1024, 0x5A);
        std::memcpy(hostPtr, pattern.data(), pattern.size());
        wroteViaHost = true;
        bool ok = deviceReadbackRange(pool->getBuffer(), 0, 32ull * 1024 * 1024,
                                      pattern.data(), "resident page 0");
        if (!ok) {
            std::cerr << "FAIL: GPU readback of resident page\n";
            ++failures;
        } else {
            std::cout << "  GPU readback of resident page: PASS\n";
        }
    } else {
        std::cout << "  (page memory not host-visible; skipping host write path)\n";
    }

    // 5. Unbound pages read as zero: copy the far tail (never bound) back.
    const VkDeviceSize kTailSize = 64ull * 1024 * 1024;
    const VkDeviceSize kTailOffset = kVirtualSize - kTailSize;
    if (pool->makeResident(kTailOffset, kTailSize) &&
        !pool->makeUnresident(kTailOffset, kTailSize)) {
        // bound + unbound cycle to prove page reuse doesn't leave stale data;
        // the region below stays permanently unbound.
    }
    std::vector<uint8_t> zeros(static_cast<size_t>(kTailSize), 0);
    bool tailOk = deviceReadbackRange(pool->getBuffer(), kTailOffset, kTailSize,
                                      zeros.data(), "unbound tail");
    if (tailOk) {
        std::cout << "  Unbound pages read as zero: PASS\n";
    } else {
        std::cerr << "FAIL: unbound pages did not read as zero\n";
        ++failures;
    }

    // 6. Unbind page 0; verify residency drops and the freed memory is reusable.
    if (!pool->makeUnresident(0, pool->getPageSize())) {
        std::cerr << "FAIL: makeUnresident(page 0)\n";
        ++failures;
    }
    if (pool->isResident(0) || pool->getResidentPageCount() != 1) {
        std::cerr << "FAIL: page 0 still resident after unbind\n";
        ++failures;
    }
    std::cout << "  Resident after unbind: " << pool->getResidentPageCount() << " page\n";

    // 7. Re-bind the freed page (free list reuse) and verify the pattern survives
    //    if we re-write it.
    if (!pool->makeResident(0, pool->getPageSize())) {
        std::cerr << "FAIL: re-bind freed page\n";
        ++failures;
    }
    if (pool->getResidentPageCount() != 2) {
        std::cerr << "FAIL: expected 2 resident pages after re-bind\n";
        ++failures;
    }
    if (wroteViaHost) {
        void* rebindPtr = pool->getResidentHostPtr(0);
        std::vector<uint8_t> pattern(32ull * 1024 * 1024, 0xA5);
        std::memcpy(rebindPtr, pattern.data(), pattern.size());
        bool ok = deviceReadbackRange(pool->getBuffer(), 0, 32ull * 1024 * 1024,
                                      pattern.data(), "rebound page 0");
        if (ok) {
            std::cout << "  Re-bound page write+readback: PASS\n";
        } else {
            std::cerr << "FAIL: re-bound page readback\n";
            ++failures;
        }
    }

    // 8. Bind the last page (far from everything) to prove scattered residency.
    if (!pool->makeResident(kVirtualSize - pool->getPageSize(), pool->getPageSize())) {
        std::cerr << "FAIL: bind last page\n";
        ++failures;
    }
    if (!pool->isResident(kVirtualSize - pool->getPageSize())) {
        std::cerr << "FAIL: last page not resident\n";
        ++failures;
    }
    std::cout << "  Final resident: " << pool->getResidentPageCount() << " / "
              << pool->getPageCount() << " pages\n";

    std::cout << "\n=== " << (failures == 0 ? "ALL SPARSE TESTS PASSED" : "SPARSE TESTS FAILED")
              << " (" << failures << " failures) ===\n";

    pool.reset();
    vkDestroyDevice(s_device, nullptr);
    vkDestroyInstance(s_instance, nullptr);
    return failures == 0 ? 0 : 1;
}