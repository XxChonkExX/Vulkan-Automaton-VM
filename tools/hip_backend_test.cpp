// HipMemoryBackend smoke test: dynamic loader + allocate/free/budget plane,
// PLUS a full UnifiedMemoryPool created over the HIP backend (backend-driven
// discovery path). Build: cl hip_backend_test.cpp (links the vulkan_vm
// import lib; the HIP runtime itself is loaded dynamically at runtime).
#include "vulkan_vm/mem_backend.hpp"
#include "vulkan_vm/hip_mem_backend.hpp"
#include "vulkan_vm/vulkan_vm.hpp"

#include <cstdio>
#include <vector>

using namespace vvm;

int main() {
    int failures = 0;

    // 1. Factory: HIP via dynamic loader. Device 1 = XTX (0 = iGPU on the
    //    reference box); fall back to 0 when only one HIP device exists.
    std::unique_ptr<HipMemoryBackend> hip = HipMemoryBackend::create(1);
    if (!hip) {
        std::printf("hip: device 1 unavailable, trying device 0\n");
        hip = HipMemoryBackend::create(0);
    }
    if (!hip) {
        std::printf("FAIL: could not create HIP backend\n");
        return 1;
    }
    std::printf("ok: hip backend created (%s)\n", hip->name());

    // 2. Discovery
    const auto heaps = hip->heaps();
    const auto types = hip->memoryTypes();
    std::printf("ok: %zu heap(s), %zu type(s)\n", heaps.size(), types.size());
    if (heaps.empty() || types.empty()) { std::printf("FAIL: empty discovery\n"); ++failures; }

    // 3. Live budget
    const BackendBudget b = hip->heapBudget(0);
    std::printf("ok: budget %llu MB (used %llu MB) valid=%d\n",
                (unsigned long long)(b.budgetBytes / (1024 * 1024)),
                (unsigned long long)(b.usedBytes / (1024 * 1024)),
                (int)b.valid);
    if (!b.valid) { std::printf("FAIL: budget invalid\n"); ++failures; }

    // 4. Allocate / write-pattern / free cycle (three sizes)
    struct Case { uint64_t size; const char* name; };
    const Case cases[] = {
        { 1024ull * 1024ull, "1 MiB" },
        { 64ull * 1024ull * 1024ull, "64 MiB" },
        { 1073741824ull, "1 GiB" },
    };
    for (const Case& c : cases) {
        int err = 0;
        const BackendMemory mem = hip->allocate({c.size, 0, false, false, 0.0f, 0}, &err);
        if (mem == 0) {
            std::printf("FAIL: allocate %s (err %d)\n", c.name, err);
            ++failures;
            continue;
        }
        const BackendBuffer buf = hip->create_buffer(mem, 0, c.size,
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                     false, &err);
        if (buf != mem) {   // HIP: buffer echoes the pointer
            std::printf("FAIL: buffer echo for %s\n", c.name);
            ++failures;
        }
        const uint64_t addr = hip->buffer_device_address(buf);
        if (addr != mem) {
            std::printf("FAIL: device address echo for %s\n", c.name);
            ++failures;
        }
        void* mapped = hip->map(mem, nullptr);
        if (mapped != nullptr) {
            std::printf("FAIL: device memory should not map\n");
            ++failures;
        }
        hip->destroy_buffer(buf);   // no-op: ownership follows memory
        hip->free(mem);
        std::printf("ok: %s allocate/echo/free\n", c.name);
    }

    // 5. Budget reflects the freed 1 GiB (allocation returned)
    const BackendBudget after = hip->heapBudget(0);
    std::printf("ok: budget after frees %llu MB used\n",
                (unsigned long long)(after.usedBytes / (1024 * 1024)));

    // 6. Vulkan backend still constructs (factory dispatch unaffected)
    DeviceConfig vkCfg{};   // null handles: constructor only caches properties
    auto vk = create_memory_backend(MemBackendKind::Vulkan, vkCfg);
    std::printf("%s: vulkan backend construct %s\n",
                vk ? "ok" : "FAIL", vk ? vk->name() : "(null)");
    if (!vk) ++failures;

    // 7. FULL POOL over the HIP backend: backend-driven discovery, buddy
    //    sub-allocation, budget policy - the Part-A end-to-end gate.
    DeviceConfig hipCfg{};
    hipCfg.backendDeviceIndex = 0;   // sole visible HIP device (the XTX)
    PoolConfig pcfg;
    pcfg.blockSize = 64ull * 1024ull * 1024ull;      // small blocks for the test
    pcfg.maxBlocks = 4;
    pcfg.enableHostVisible = false;
    auto pool = UnifiedMemoryPool::create(hipCfg, pcfg);
    if (!pool.has_value()) {
        std::printf("FAIL: pool create over HIP backend\n");
        ++failures;
    } else {
        std::printf("ok: UnifiedMemoryPool created over HIP backend\n");
        // Sub-allocate through the pool's policy layer (AllocDesc path).
        vvm::AllocDesc desc;
        desc.size = 8ull * 1024ull * 1024ull;
        desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        desc.memoryUsage = vvm::MemoryUsage::GpuOnly;
        auto a1 = pool->allocate(desc);
        if (!a1.has_value()) {
            std::printf("FAIL: pool allocate 8 MiB\n");
            ++failures;
        } else {
            const uint64_t ptr = reinterpret_cast<uint64_t>(a1->buffer);
            std::printf("ok: pool allocate 8 MiB -> device ptr %p (echo %s)\n",
                        (void*)ptr, ptr == reinterpret_cast<uint64_t>(a1->memory) ? "yes" : "no");
            if (a1->deviceAddress != 0) {
                std::printf("ok: device address %llu\n", (unsigned long long)a1->deviceAddress);
            }
            pool->deallocate(std::move(*a1));
            std::printf("ok: pool deallocate\n");
        }
        const PoolStats st = pool->getStats();
        std::printf("ok: pool stats: blocks=%u used=%llu MB cap=%llu MB\n",
                    st.blockCount,
                    (unsigned long long)(st.totalUsed / (1024 * 1024)),
                    (unsigned long long)(st.totalCapacity / (1024 * 1024)));

        // Budget reservation (KV pattern): hold 1 GiB for a late-arriving
        // giant, verify small allocs still land, release, then land the
        // giant itself as a dedicated allocation. Unreserve saturates.
        constexpr uint64_t kHold = 1ull * 1024 * 1024 * 1024;
        if (!pool->reserve(kHold)) {
            std::printf("FAIL: reserve 1 GiB\n");
            ++failures;
        } else if (pool->reservedBytes() != kHold ||
                   pool->getStats().reservedBytes != kHold) {
            std::printf("FAIL: reservation not visible in stats\n");
            ++failures;
        } else {
            std::printf("ok: reserve 1 GiB held\n");
        }
        if (pool->reserve(100ull * 1024 * 1024 * 1024 * 1024)) {
            std::printf("FAIL: absurd reservation accepted\n");
            ++failures;
        } else if (pool->reservedBytes() != kHold) {
            std::printf("FAIL: failed reserve mutated the hold\n");
            ++failures;
        } else {
            std::printf("ok: absurd reservation refused, hold intact\n");
        }
        vvm::AllocDesc rd;
        rd.size = 8ull * 1024ull * 1024ull;
        rd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        rd.memoryUsage = vvm::MemoryUsage::GpuOnly;
        auto ra = pool->allocate(rd);
        if (!ra.has_value()) {
            std::printf("FAIL: 8 MiB alongside reservation\n");
            ++failures;
        } else {
            pool->deallocate(std::move(*ra));
            std::printf("ok: 8 MiB alongside reservation\n");
        }
        pool->unreserve(kHold);
        pool->unreserve(kHold);  // double-release must saturate, not underflow
        if (pool->reservedBytes() != 0) {
            std::printf("FAIL: unreserve did not clear\n");
            ++failures;
        } else {
            std::printf("ok: unreserve clears + saturates\n");
        }
        vvm::AllocDesc gd;
        gd.size = kHold;
        gd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        gd.memoryUsage = vvm::MemoryUsage::GpuOnly;
        auto ga = pool->allocate(gd);
        if (!ga.has_value() || ga->blockIndex != UINT32_MAX) {
            std::printf("FAIL: 1 GiB giant did not land dedicated\n");
            ++failures;
            if (ga.has_value()) pool->deallocate(std::move(*ga));
        } else {
            const PoolStats gs = pool->getStats();
            if (gs.dedicatedCount < 1) {
                std::printf("FAIL: giant missing from dedicated tracking\n");
                ++failures;
            } else {
                std::printf("ok: 1 GiB giant landed dedicated\n");
            }
            pool->deallocate(std::move(*ga));
            const PoolStats gs2 = pool->getStats();
            if (gs2.dedicatedCount != 0 || gs2.reservedBytes != 0) {
                std::printf("FAIL: teardown leftovers (dedicated=%u reserved=%llu)\n",
                            gs2.dedicatedCount,
                            (unsigned long long)gs2.reservedBytes);
                ++failures;
            } else {
                std::printf("ok: giant freed, stats clean\n");
            }
        }
    }

    if (failures == 0) {
        std::printf("ALL HIP BACKEND TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
