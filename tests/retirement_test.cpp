// retirement_test: GPU-lifetime-safe reclamation (P0-2). Exercises
// pool.retire() / collect() / retireTicket() on a real Vulkan device:
//   A. retire against a never-signaled value -> collect() reclaims nothing,
//      memory stays accounted (no premature reuse).
//   B. retire against a signaled timeline value -> collect() reclaims exactly
//      once, stats return to baseline (no leak, no double-free).
//   C. copyDeviceToDevice with a caller fence is genuinely asynchronous:
//      returns true while the fence is still unsignaled (256 MiB copy takes
//      milliseconds; the status check runs in microseconds), data verifies
//      after the fence, and collect() reclaims the retired teardown.
//   D. backend-neutral CompletionTokens: Ready reclaims with no GPU work,
//      Foreign consults the callback per collect, retireToken() feeds the
//      token overload like B.
// Skips cleanly when no discrete GPU or no timeline-semaphore feature.
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace vvm;

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        ++checks;                                                             \
        if (!(cond)) {                                                        \
            std::printf("FAIL: %s (line %d)\n", msg, __LINE__);               \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

static bool createDeviceForPool(const DeviceScore& score, VkInstance instance,
                                DeviceConfig& out) {
    auto queues = findQueueFamilies(score.device);
    if (!queues.transfer && !queues.graphics) return false;
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

    const char* exts[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#if defined(VVM_PLATFORM_WINDOWS)
        "VK_KHR_external_memory_win32",
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
#endif
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
#ifndef _WIN32
    dci.enabledExtensionCount = 6;
#endif
    dci.ppEnabledExtensionNames = exts;

    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(score.device, &dci, nullptr, &device) != VK_SUCCESS) {
        return false;
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
    return true;
}

static PoolConfig poolCfg() {
    PoolConfig pc;
    pc.blockSize = 64ull * 1024ull * 1024ull;
    pc.minAlignment = 256;
    pc.maxBlocks = 4;
    pc.enableHostVisible = false;
    pc.enableExternal = true;
    pc.enableDeviceAddress = true;
    pc.maxHeapFraction = 0.0f;
    pc.maxPoolBytes = 0;
    return pc;
}

static void signalTicket(VkDevice device, VkQueue queue, uint32_t family,
                         VkSemaphore tl, uint64_t val) {
    VkCommandPoolCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpInfo.queueFamilyIndex = family;
    VkCommandPool tmpPool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device, &cpInfo, nullptr, &tmpPool) != VK_SUCCESS) return;
    VkCommandBufferAllocateInfo cba{};
    cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = tmpPool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &cba, &cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(device, tmpPool, nullptr);
        return;
    }
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &beginInfo) == VK_SUCCESS) {
        vkEndCommandBuffer(cmd);
        VkTimelineSemaphoreSubmitInfo tlInfo{};
        tlInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        tlInfo.signalSemaphoreValueCount = 1;
        tlInfo.pSignalSemaphoreValues = &val;
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.pNext = &tlInfo;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &tl;
        if (vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS) {
            vkQueueWaitIdle(queue);
        }
    }
    vkDestroyCommandPool(device, tmpPool, nullptr);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    std::printf("== retirement_test ==\n");

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM Retirement Test";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        std::printf("FAIL: vkCreateInstance\n");
        return 1;
    }
    auto devices = enumerateDevices(instance);
    int idx = -1;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i].props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            idx = static_cast<int>(i);
            break;
        }
    }
    if (idx < 0) {
        std::printf("SKIP: no discrete GPU\n");
        vkDestroyInstance(instance, nullptr);
        return 0;
    }
    std::printf("device: %s\n", devices[idx].props.deviceName);

    DeviceConfig dc{};
    if (!createDeviceForPool(devices[idx], instance, dc)) {
        std::printf("FAIL: device creation\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    // A: retire against a never-signaled value. Own pool scope: the still-
    // queued item exercises the dtor path at scope end.
    {
        auto pool = UnifiedMemoryPool::create(dc, poolCfg());
        CHECK(pool.has_value(), "pool create");
        if (pool.has_value()) {
            auto [tl, val] = pool->retireTicket();
            if (!tl) {
                std::printf("SKIP A: no timeline support\n");
            } else {
                auto a = pool->allocate(4ull * 1024ull * 1024ull,
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                CHECK(a.has_value(), "A: allocate 4 MiB");
                const uint64_t usedBefore = pool->getStats().totalUsed;
                CHECK(pool->retire(std::move(*a), tl, val + 1000000u),
                      "A: retire accepted");
                CHECK(pool->collect() == 0, "A: collect reclaims nothing pre-signal");
                CHECK(pool->getStats().totalUsed == usedBefore,
                      "A: memory still accounted while retired");
            }
        }
    }

    // B: signal the ticket, then retire -> collect reclaims exactly once.
    {
        auto pool = UnifiedMemoryPool::create(dc, poolCfg());
        CHECK(pool.has_value(), "pool create");
        if (pool.has_value()) {
            const uint64_t baseUsed = pool->getStats().totalUsed;
            auto [tl, val] = pool->retireTicket();
            if (!tl) {
                std::printf("SKIP B: no timeline support\n");
            } else {
                signalTicket(dc.device, dc.transferQueue, dc.transferQueueFamily, tl, val);
                auto b = pool->allocate(4ull * 1024ull * 1024ull,
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                CHECK(b.has_value(), "B: allocate 4 MiB");
                CHECK(pool->retire(std::move(*b), tl, val), "B: retire accepted");
                CHECK(pool->collect() == 1, "B: collect reclaims exactly 1 post-signal");
                CHECK(pool->getStats().totalUsed == baseUsed,
                      "B: stats back to baseline (no leak)");
                CHECK(pool->collect() == 0, "B: second collect reclaims nothing");
            }
        }
    }

    // C: async copyDeviceToDevice via two same-device manager instances.
    {
        auto manager = MultiGPUPoolManager::create({dc, dc}, poolCfg(), 0);
        CHECK(manager.has_value(), "C: manager create (same-device x2)");
        if (manager.has_value()) {
            auto& pool0 = manager->getPool(0);
            auto& pool1 = manager->getPool(1);
            auto [tl, val] = pool1.retireTicket();
            if (!tl) {
                std::printf("SKIP C: no timeline support\n");
            } else {
                constexpr VkDeviceSize kSize = 256ull * 1024ull * 1024ull;
                const VkBufferUsageFlags kUsage =
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                auto src = pool0.allocateDedicatedExportable(kSize, kUsage);
                CHECK(src.has_value(), "C: dedicated src");
                auto dst = pool1.allocate(kSize, kUsage);
                CHECK(dst.has_value(), "C: dst");
                auto stgS = pool0.allocate(kSize, kUsage,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                auto stgD = pool1.allocate(kSize, kUsage,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                CHECK(stgS.has_value() && stgD.has_value(), "C: staging");
                CHECK(stgS->hostPtr != nullptr, "C: src staging mapped");
                CHECK(stgD->hostPtr != nullptr, "C: dst staging mapped");
                std::vector<uint8_t> pattern(static_cast<size_t>(kSize));
                for (size_t i = 0; i < pattern.size(); ++i)
                    pattern[i] = static_cast<uint8_t>(i * 7 + 3);
                std::memcpy(stgS->hostPtr, pattern.data(), pattern.size());
                CHECK(pool0.copyBuffer(*stgS, *src, 0, 0, kSize), "C: stage-in");
                VkFenceCreateInfo fInfo{};
                fInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                VkFence fence = VK_NULL_HANDLE;
                CHECK(vkCreateFence(dc.device, &fInfo, nullptr, &fence) == VK_SUCCESS,
                      "C: fence");
                const auto t0 = std::chrono::steady_clock::now();
                CHECK(manager->copyDeviceToDevice(0, 1, *src, *dst, 0, 0, kSize, fence),
                      "C: async copy accepted");
                // Async proof: 256 MiB takes milliseconds on the GPU; a
                // synchronous implementation could not return this fast.
                const auto dtUs = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count();
                const VkResult st = vkGetFenceStatus(dc.device, fence);
                std::printf("C: copy returned in %lld us, fence %s\n", (long long)dtUs,
                            st == VK_SUCCESS ? "signaled" : "unsignaled");
                CHECK(st != VK_SUCCESS, "C: fence unsignaled at return (truly async)");
                CHECK(vkWaitForFences(dc.device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS,
                      "C: fence wait");
                vkDestroyFence(dc.device, fence, nullptr);
                std::memset(stgD->hostPtr, 0, static_cast<size_t>(kSize));
                CHECK(pool1.copyBuffer(*dst, *stgD, 0, 0, kSize), "C: readback");
                CHECK(std::memcmp(stgD->hostPtr, pattern.data(), pattern.size()) == 0,
                      "C: data verified after async copy");
                uint32_t got = 0;
                for (int i = 0; i < 100 && got == 0; ++i) got = pool1.collect();
                CHECK(got >= 1, "C: collect reclaimed retired teardown");
                pool1.deallocate(std::move(*stgD));
                pool0.deallocate(std::move(*stgS));
                pool1.deallocate(std::move(*dst));
                pool0.deallocate(std::move(*src));
            }
        }
    }

    // D: backend-neutral tokens - Ready needs no GPU work, Foreign
    // consults per collect, retireToken() feeds the token overload.
    {
        auto pool = UnifiedMemoryPool::create(dc, poolCfg());
        CHECK(pool.has_value(), "pool create");
        if (pool.has_value()) {
            const uint64_t baseUsed = pool->getStats().totalUsed;
            auto d1 = pool->allocate(1ull * 1024ull * 1024ull,
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            CHECK(d1.has_value(), "D1: allocate 1 MiB");
            CHECK(pool->retire(std::move(*d1), CompletionToken::ready()),
                  "D1: Ready retire accepted");
            CHECK(pool->collect() == 1, "D1: Ready reclaims on first collect");
            CHECK(pool->getStats().totalUsed == baseUsed,
                  "D1: baseline restored");
            auto d2 = pool->allocate(1ull * 1024ull * 1024ull,
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            CHECK(d2.has_value(), "D2: allocate 1 MiB");
            int polls = 0;
            CHECK(pool->retire(std::move(*d2), CompletionToken::foreign(
                                                  [&] { return ++polls >= 2; })),
                  "D2: Foreign retire accepted");
            CHECK(pool->collect() == 0, "D2: first collect reclaims nothing");
            CHECK(pool->collect() == 1, "D2: second collect reclaims");
            CHECK(polls == 2, "D2: consulted once per collect");
            auto tok = pool->retireToken();
            if (!tok.has_value()) {
                std::printf("SKIP D3: no timeline support\n");
            } else {
                signalTicket(dc.device, dc.transferQueue, dc.transferQueueFamily,
                             tok->timeline, tok->value);
                auto d3 = pool->allocate(1ull * 1024ull * 1024ull,
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                CHECK(d3.has_value(), "D3: allocate 1 MiB");
                CHECK(pool->retire(std::move(*d3), std::move(*tok)),
                      "D3: token retire accepted");
                CHECK(pool->collect() == 1,
                      "D3: token collect reclaims post-signal");
            }
            CHECK(pool->getStats().totalUsed == baseUsed,
                  "D: baseline restored");
        }
    }

    vkDestroyDevice(dc.device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::printf("retirement_test: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
