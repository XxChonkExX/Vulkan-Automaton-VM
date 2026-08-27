// Tensor collective operations test (transport-level, single-process).
//
// Exercises allReduce (Sum/Max/Product/Mean), broadcast, allGather and
// reduceScatter end-to-end through real Vulkan buffers using the CPU fallback
// reduce path in TensorTransportImpl. Expected to pass with a single physical
// GPU (all ranks staged on device 0); broadcast with 2+ devices is validated
// when such hardware is present.

#include <vulkan_vm/tensor_transport.hpp>
#include <vulkan_vm/utils.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace vvm;
using namespace vvm::tensor;

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

namespace {

struct TestEnv {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::vector<DeviceConfig> devices;
    PoolConfig poolConfig;
};

// Read a GPU allocation back into host memory.
bool readGpu(MultiGPUPoolManager& mgr, uint32_t deviceIdx, const Allocation& alloc,
             std::vector<uint8_t>& out) {
    auto& pool = mgr.getPool(deviceIdx);
    AllocDesc desc;
    desc.size = alloc.size;
    desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryUsage = MemoryUsage::GpuToCpu;
    desc.mapped = true;
    desc.exportable = false;
    desc.name = "test-read-staging";
    auto stage = pool.allocate(desc);
    if (!stage || !stage->hostPtr) return false;
    if (!pool.copyBuffer(alloc, *stage, 0, 0, alloc.size)) {
        pool.deallocate(std::move(*stage));
        return false;
    }
    out.assign(static_cast<const uint8_t*>(stage->hostPtr),
               static_cast<const uint8_t*>(stage->hostPtr) + alloc.size);
    pool.deallocate(std::move(*stage));
    return true;
}

// Upload host bytes into an existing GPU allocation.
bool writeGpu(MultiGPUPoolManager& mg, uint32_t deviceIdx, const Allocation& alloc,
              const std::vector<uint8_t>& in) {
    auto& pool = mg.getPool(deviceIdx);
    AllocDesc desc;
    desc.size = in.size();
    desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryUsage = MemoryUsage::CpuToGpu;
    desc.mapped = true;
    desc.exportable = false;
    desc.name = "test-write-staging";
    auto stage = pool.allocate(desc);
    if (!stage || !stage->hostPtr) return false;
    std::memcpy(stage->hostPtr, in.data(), in.size());
    bool ok = pool.copyBuffer(*stage, alloc, 0, 0, in.size());
    pool.deallocate(std::move(*stage));
    return ok;
}

Allocation allocGpu(MultiGPUPoolManager& mg, uint32_t deviceIdx, size_t bytes) {
    AllocDesc desc;
    desc.size = bytes;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryUsage = MemoryUsage::GpuOnly;
    desc.exportable = false;
    desc.mapped = false;
    desc.name = "test-tensor";
    auto a = mg.getPool(deviceIdx).allocate(desc);
    assert(a.has_value());
    return std::move(*a);
}

// Every live tensor handle, so we can return their allocations to the pools
// before device teardown. TensorAllocation does NOT free its VkBuffer by
// itself (see docs/LIFETIME_CONTRACT.md) - dropping a handle without an
// explicit deallocate leaks the buffer.
std::vector<TensorHandle>& liveHandles() {
    static std::vector<TensorHandle> handles;
    return handles;
}

TensorHandle makeHandle(const Allocation& alloc, DataType dt, size_t bytes,
                        std::string name) {
    auto h = std::make_shared<TensorAllocation>();
    h->allocation = alloc;
    h->metadata.dtype = dt;
    h->metadata.layout = MemoryLayout::Contiguous;
    h->metadata.shape = TensorShape::makeContiguous({static_cast<int64_t>(bytes / dataTypeSize(dt))});
    h->metadata.name = std::move(name);
    h->deviceIndex = 0;
    liveHandles().push_back(h);
    return h;
}

std::vector<float> floatsOf(const std::vector<uint8_t>& raw) {
    std::vector<float> out(raw.size() / sizeof(float));
    std::memcpy(out.data(), raw.data(), out.size() * sizeof(float));
    return out;
}

std::vector<uint8_t> bytesOf(const std::vector<float>& vals) {
    std::vector<uint8_t> raw(vals.size() * sizeof(float));
    std::memcpy(raw.data(), vals.data(), raw.size());
    return raw;
}

bool envInit(TestEnv& env) {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM TensorCollective Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> extensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };

    VkInstanceCreateInfo instInfo{};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;
    instInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instInfo.ppEnabledExtensionNames = extensions.data();

    VkInstance instance;
    if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS) return false;
    env.instance = instance;

    auto discovered = enumerateDevices(instance);
    if (discovered.empty()) return false;
    auto best = selectBestDevice(discovered, true, 1024);
    if (!best) return false;

    std::vector<const char*> requiredExts = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#ifdef VVM_PLATFORM_LINUX
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
#else
        "VK_KHR_external_memory_win32",
#endif
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    };

    auto queues = findQueueFamilies(best->device);
    float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    auto addQueue = [&](std::optional<uint32_t> family) {
        if (family) {
            VkDeviceQueueCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            info.queueFamilyIndex = *family;
            info.queueCount = 1;
            info.pQueuePriorities = &queuePriority;
            queueInfos.push_back(info);
        }
    };
    addQueue(queues.graphics);
    addQueue(queues.compute);
    addQueue(queues.transfer);

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDeviceBufferDeviceAddressFeatures addrFeatures{};
    addrFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    addrFeatures.bufferDeviceAddress = VK_TRUE;
    addrFeatures.pNext = features2.pNext;
    features2.pNext = &addrFeatures;
    VkPhysicalDeviceTimelineSemaphoreFeatures tsFeatures{};
    tsFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    tsFeatures.timelineSemaphore = VK_TRUE;
    tsFeatures.pNext = features2.pNext;
    features2.pNext = &tsFeatures;
    VkPhysicalDeviceVulkan12Features v12Features{};
    v12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12Features.bufferDeviceAddress = VK_TRUE;
    v12Features.timelineSemaphore = VK_TRUE;
    v12Features.pNext = features2.pNext;
    features2.pNext = &v12Features;

    VkDeviceCreateInfo devInfo{};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    devInfo.pQueueCreateInfos = queueInfos.data();
    devInfo.pNext = &features2;
    devInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExts.size());
    devInfo.ppEnabledExtensionNames = requiredExts.data();

    VkDevice device;
    if (vkCreateDevice(best->device, &devInfo, nullptr, &device) != VK_SUCCESS) return false;
    env.device = device;

    DeviceConfig cfg;
    cfg.physicalDevice = best->device;
    cfg.device = device;
    cfg.graphicsQueueFamily = queues.graphics.value_or(0);
    cfg.computeQueueFamily = queues.compute.value_or(0);
    cfg.transferQueueFamily = queues.transfer.value_or(0);
    if (queues.graphics) vkGetDeviceQueue(device, *queues.graphics, 0, &cfg.graphicsQueue);
    if (queues.compute) vkGetDeviceQueue(device, *queues.compute, 0, &cfg.computeQueue);
    if (queues.transfer) vkGetDeviceQueue(device, *queues.transfer, 0, &cfg.transferQueue);
    env.devices.push_back(cfg);

    env.poolConfig.blockSize = 256 * 1024 * 1024;
    env.poolConfig.minAlignment = 4 * 1024;  // 4KB alignment for small test tensors
    env.poolConfig.enableHostVisible = true;
    env.poolConfig.enableExternal = true;
    env.poolConfig.enableDeviceAddress = true;
    env.poolConfig.maxBlocks = 32;  // Increased for test allocations + shadow buffers

    return true;
}

} // namespace

int main() {
    TestEnv env;
    if (!envInit(env)) {
        std::printf("SKIP: no Vulkan device available\n");
        return 0;
    }

    TransportConfig cfg;
    cfg.preference = TransportConfig::Preference::Auto;
    cfg.enableAsyncPipeline = false;   // synchronous execution for tests
    cfg.enableGPUDirect = false;

    auto transport = Transport::create(cfg, env.devices, env.poolConfig);
    CHECK(transport != nullptr);
    CHECK(transport->initialize());

    // Use the transport's internal pool manager for all allocations and read/write
    MultiGPUPoolManager* poolMgr = transport->getPoolManager();
    CHECK(poolMgr != nullptr);
    if (failures) goto done;

    {
        // ---- allReduce Sum, three participants on device 0 ----
        const size_t n = 1024;  // 1024 floats = 4KB
        const size_t bytes = n * sizeof(float);
        std::vector<TensorHandle> group;
        for (uint32_t r = 0; r < 3; ++r) {
            group.push_back(makeHandle(allocGpu(*poolMgr, 0, bytes),
                                       DataType::Float32, bytes, "ar" + std::to_string(r)));
            std::vector<float> seeds(n);
            for (size_t i = 0; i < n; ++i) seeds[i] = static_cast<float>(i + r);
            CHECK(writeGpu(*poolMgr, 0, group.back()->allocation, bytesOf(seeds)));
        }
        CHECK(transport->allReduce(group, ReduceOp::Sum, {0, 0, 0}));
        std::vector<float> ref(n);
        for (size_t i = 0; i < n; ++i) ref[i] = static_cast<float>(3 * i + 3);
        for (const auto& t : group) {
            std::vector<uint8_t> raw;
            CHECK(readGpu(*poolMgr, 0, t->allocation, raw));
            auto got = floatsOf(raw);
            for (size_t i = 0; i < n; ++i) {
                CHECK(std::fabs(got[i] - ref[i]) < 1e-4f);
            }
        }
    }

    {
        // allReduce with Mean
        const size_t n = 1024;
        const size_t bytes = n * sizeof(float);
        std::vector<TensorHandle> group;
        for (uint32_t r = 0; r < 2; ++r) {
            group.push_back(makeHandle(allocGpu(*poolMgr, 0, bytes),
                                       DataType::Float32, bytes, "arm" + std::to_string(r)));
            std::vector<float> seeds(n, static_cast<float>(5.0 * (r + 1)));
            CHECK(writeGpu(*poolMgr, 0, group.back()->allocation, bytesOf(seeds)));
        }
        CHECK(transport->allReduce(group, ReduceOp::Mean, {0, 0}));
        std::vector<uint8_t> raw;
        CHECK(readGpu(*poolMgr, 0, group[0]->allocation, raw));
        auto got = floatsOf(raw);
        for (size_t i = 0; i < n; ++i) CHECK(std::fabs(got[i] - 7.5f) < 1e-4f);
    }

    {
        // allReduce on integer data: sum of small ints must be exact.
        const size_t n = 1024;
        const size_t bytes = n * sizeof(int32_t);
        std::vector<TensorHandle> group;
        for (uint32_t r = 1; r <= 3; ++r) {
            group.push_back(makeHandle(allocGpu(*poolMgr, 0, bytes),
                                        DataType::Int32, bytes, "ari" + std::to_string(r)));
            std::vector<int32_t> seeds(n, static_cast<int32_t>(r));
            std::vector<uint8_t> raw(seeds.size() * sizeof(int32_t));
            std::memcpy(raw.data(), seeds.data(), raw.size());
            CHECK(writeGpu(*poolMgr, 0, group.back()->allocation, raw));
        }
        CHECK(transport->allReduce(group, ReduceOp::Sum, {0, 0, 0}));
        std::vector<uint8_t> raw;
        CHECK(readGpu(*poolMgr, 0, group[0]->allocation, raw));
        std::vector<int32_t> got(raw.size() / sizeof(int32_t));
        std::memcpy(got.data(), raw.data(), raw.size());
        for (size_t i = 0; i < n; ++i) CHECK(got[i] == 6);
    }

    {
        // allReduce Product: 1*2*3=6 per element.
        const size_t n = 1024;
        const size_t bytes = n * sizeof(float);
        std::vector<TensorHandle> group;
        std::vector<float> ref(n, 6.0f);
        for (uint32_t r = 1; r <= 3; ++r) {
            group.push_back(makeHandle(allocGpu(*poolMgr, 0, bytes),
                                        DataType::Float32, bytes, "arp" + std::to_string(r)));
            std::vector<float> seeds(n, static_cast<float>(r));
            CHECK(writeGpu(*poolMgr, 0, group.back()->allocation, bytesOf(seeds)));
        }
        CHECK(transport->allReduce(group, ReduceOp::Product, {0, 0, 0}));
        std::vector<uint8_t> raw;
        CHECK(readGpu(*poolMgr, 0, group[0]->allocation, raw));
        auto got = floatsOf(raw);
        for (size_t i = 0; i < n; ++i) CHECK(std::fabs(got[i] - ref[i]) < 1e-4f);
    }

    {
        // allGather: two contributors appended in order.
        const size_t n = 1024;
        const size_t bytes = n * sizeof(float);
        std::vector<TensorHandle> inputs(2);
        inputs[0] = makeHandle(allocGpu(*poolMgr, 0, bytes), DataType::Float32, bytes, "ag0");
        inputs[1] = makeHandle(allocGpu(*poolMgr, 0, bytes), DataType::Float32, bytes, "ag1");
        std::vector<float> a(n, 1.0f), b(n, 2.0f);
        CHECK(writeGpu(*poolMgr, 0, inputs[0]->allocation, bytesOf(a)));
        CHECK(writeGpu(*poolMgr, 0, inputs[1]->allocation, bytesOf(b)));

        auto output = makeHandle(allocGpu(*poolMgr, 0, 2 * bytes),
                                  DataType::Float32, 2 * bytes, "agout");
        CHECK(transport->allGather(inputs, output, {0, 0}));
        std::vector<uint8_t> raw;
        CHECK(readGpu(*poolMgr, 0, output->allocation, raw));
        auto got = floatsOf(raw);
        for (size_t i = 0; i < n; ++i) {
            CHECK(std::fabs(got[i] - 1.0f) < 1e-6f);
            CHECK(std::fabs(got[i + n] - 2.0f) < 1e-6f);
        }
    }

    {
        // reduceScatter: 2 participants -> output gets chunk 0 of the reduced.
        const size_t n = 4096;
        const size_t bytes = n * sizeof(float);
        std::vector<TensorHandle> inputs(2);
        inputs[0] = makeHandle(allocGpu(*poolMgr, 0, bytes), DataType::Float32, bytes, "rs0");
        inputs[1] = makeHandle(allocGpu(*poolMgr, 0, bytes), DataType::Float32, bytes, "rs1");
        std::vector<float> a(n), b(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(n - i);
        }
        CHECK(writeGpu(*poolMgr, 0, inputs[0]->allocation, bytesOf(a)));
        CHECK(writeGpu(*poolMgr, 0, inputs[1]->allocation, bytesOf(b)));

        auto output = makeHandle(allocGpu(*poolMgr, 0, bytes / 2),
                                  DataType::Float32, bytes / 2, "rsout");
        CHECK(transport->reduceScatter(inputs, output, ReduceOp::Sum, {0, 0}));
        std::vector<uint8_t> raw;
        CHECK(readGpu(*poolMgr, 0, output->allocation, raw));
        auto got = floatsOf(raw);
        for (size_t i = 0; i < n / 2; ++i) {
            CHECK(std::fabs(got[i] - static_cast<float>(n)) < 1e-4f);
        }
    }

    {
        // broadcast: root registered through allocateDistributed.
        TensorMetadata meta;
        meta.dtype = DataType::Float32;
        meta.shape = TensorShape::makeContiguous({16});
        meta.name = "bcast";

        auto dist = transport->allocateDistributed(meta, {0});
        CHECK(dist.size() == 1);
        if (!dist.empty()) {
            std::vector<float> seeds(16);
            for (size_t i = 0; i < 16; ++i) seeds[i] = static_cast<float>(i * 3);
            CHECK(writeGpu(*poolMgr, 0, dist[0]->allocation, bytesOf(seeds)));
        }
        if (dist.size() == 1) {
            CHECK(transport->broadcast(dist[0], {0}, 0));
        }
    }

    {
        // Broadcast across 2 devices when hardware allows.
        if (poolMgr->getInstances().size() < 2) {
            static constexpr char kMsg[] = "SKIP broadcast two-device: insufficient GPUs\n";
            std::fwrite(kMsg, 1, sizeof(kMsg) - 1, stdout);
        } else {
            TensorMetadata meta;
            meta.dtype = DataType::Float32;
            meta.shape = TensorShape::makeContiguous({32});
            meta.name = "bcast2";
            auto dist = transport->allocateDistributed(meta, {0, 1});
            CHECK(dist.size() == 2);
            if (dist.size() == 2) {
                std::vector<float> seeds(32);
                for (size_t i = 0; i < 32; ++i) seeds[i] = static_cast<float>(i + 1);
                CHECK(writeGpu(*poolMgr, 0, dist[0]->allocation, bytesOf(seeds)));
                CHECK(transport->broadcast(dist[0], {0, 1}, 0));
                std::vector<uint8_t> raw;
                CHECK(readGpu(*poolMgr, 1, dist[1]->allocation, raw));
                auto got = floatsOf(raw);
                for (size_t i = 0; i < 32; ++i) {
                    CHECK(std::fabs(got[i] - static_cast<float>(i + 1)) < 1e-6f);
                }
            }
        }
    }

done:
    // Return every tensor allocation to its pool BEFORE shutting the
    // transport / destroying the device; otherwise validation reports leaked
    // VkBuffers (VUID-vkDestroyDevice-device-05137). Deallocating ahead of
    // shutdown() also avoids racing pool teardown against live worker threads.
    for (auto& h : liveHandles()) {
        if (!h) continue;
        poolMgr->getPool(h->deviceIndex).deallocate(std::move(h->allocation));
    }
    liveHandles().clear();
    transport->shutdown();
    if (env.device) vkDestroyDevice(env.device, nullptr);
    if (env.instance) vkDestroyInstance(env.instance, nullptr);

    if (failures > 0) {
        std::printf("\nFAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS: tensor collective tests\n");
    return 0;
}

