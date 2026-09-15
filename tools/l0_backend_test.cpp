// L0MemoryBackend smoke test: dynamic ze_loader + allocate/free plane
// against the real Intel device (B70 on the reference box), PLUS a full
// UnifiedMemoryPool created over the L0 backend (backend-driven discovery).
#include "vulkan_vm/mem_backend.hpp"
#include "vulkan_vm/l0_mem_backend.hpp"
#include "vulkan_vm/vulkan_vm.hpp"

#include <cstdio>

using namespace vvm;

int main() {
    int failures = 0;

    auto l0 = L0MemoryBackend::create(0);
    if (!l0) {
        std::printf("FAIL: could not create L0 backend\n");
        return 1;
    }
    std::printf("ok: l0 backend created (%s)\n", l0->name());

    const auto heaps = l0->heaps();
    const auto types = l0->memoryTypes();
    std::printf("ok: %zu heap(s) = %llu MB, %zu type(s)\n",
                heaps.size(),
                heaps.empty() ? 0ull : (unsigned long long)(heaps[0].size / (1024 * 1024)),
                types.size());
    if (heaps.empty() || types.empty()) { std::printf("FAIL: empty discovery\n"); ++failures; }

    struct Case { uint64_t size; const char* name; };
    const Case cases[] = {
        { 1024ull * 1024ull, "1 MiB" },
        { 64ull * 1024ull * 1024ull, "64 MiB" },
        { 1024ull * 1024ull * 1024ull, "1 GiB" },
    };
    for (const Case& c : cases) {
        int err = 0;
        const BackendMemory mem = l0->allocate({c.size, 0, false, false, 0.0f, 0}, &err);
        if (mem == 0) {
            std::printf("FAIL: allocate %s (err %d)\n", c.name, err);
            ++failures;
            continue;
        }
        const BackendBuffer buf = l0->create_buffer(mem, 0, c.size,
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                     false, &err);
        if (buf != mem) { std::printf("FAIL: buffer echo for %s\n", c.name); ++failures; }
        if (l0->buffer_device_address(buf) != mem) {
            std::printf("FAIL: device address echo for %s\n", c.name);
            ++failures;
        }
        if (l0->map(mem, nullptr) != nullptr) {
            std::printf("FAIL: device memory should not map\n");
            ++failures;
        }
        l0->destroy_buffer(buf);
        l0->free(mem);
        std::printf("ok: %s allocate/echo/free\n", c.name);
    }

    // FULL POOL over the L0 backend: explicit kind hint, backend-driven
    // discovery, buddy sub-allocation.
    DeviceConfig l0Cfg{};
    l0Cfg.memBackendKind = static_cast<int32_t>(MemBackendKind::Level0);
    l0Cfg.backendDeviceIndex = 0;   // the B70
    PoolConfig pcfg;
    pcfg.blockSize = 64ull * 1024ull * 1024ull;
    pcfg.maxBlocks = 4;
    pcfg.enableHostVisible = false;
    auto pool = UnifiedMemoryPool::create(l0Cfg, pcfg);
    if (!pool.has_value()) {
        std::printf("FAIL: pool create over L0 backend\n");
        ++failures;
    } else {
        std::printf("ok: UnifiedMemoryPool created over L0 backend\n");
        AllocDesc desc;
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
            pool->deallocate(std::move(*a1));
            std::printf("ok: pool deallocate\n");
        }
        const PoolStats st = pool->getStats();
        std::printf("ok: pool stats: blocks=%u used=%llu MB cap=%llu MB\n",
                    st.blockCount,
                    (unsigned long long)(st.totalUsed / (1024 * 1024)),
                    (unsigned long long)(st.totalCapacity / (1024 * 1024)));
    }

    if (failures == 0) {
        std::printf("ALL L0 BACKEND TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
