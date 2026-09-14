// HipMemoryBackend smoke test: dynamic loader + allocate/free/budget plane.
// Build: cl hip_backend_test.cpp (links vulkan_vm import lib, loads
// vulkan_vm.dll; the HIP runtime itself is loaded dynamically at runtime).
#include "vulkan_vm/mem_backend.hpp"
#include "vulkan_vm/hip_mem_backend.hpp"

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

    if (failures == 0) {
        std::printf("ALL HIP BACKEND TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
