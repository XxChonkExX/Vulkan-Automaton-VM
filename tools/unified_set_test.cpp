// UnifiedPoolSet test: enumerate all runtimes, create HIP + L0 pools in
// ONE process, allocate on each, verify distinct devices + JSON aggregate.
#include "vulkan_vm/device_registry.hpp"

#include <cstdio>

using namespace vvm;

int main() {
    int failures = 0;

    const auto devices = enumerate_all_devices();
    std::printf("enumerated %zu device(s):\n", devices.size());
    for (const auto& d : devices) {
        std::printf("  '%s' vendor=0x%04x total=%llu MB src=%d preferred=%s%s\n",
                    d.name, d.vendorId,
                    (unsigned long long)(d.totalMem / (1024 * 1024)),
                    (int)d.source, backend_kind_name(d.preferred),
                    d.integrated ? " (integrated)" : "");
    }
    if (devices.empty()) {
        std::printf("FAIL: no devices enumerated\n");
        return 1;
    }

    PoolConfig pcfg;
    pcfg.blockSize = 64ull * 1024ull * 1024ull;
    pcfg.maxBlocks = 2;
    pcfg.enableHostVisible = false;

    auto set = UnifiedPoolSet::create(pcfg);
    if (!set.has_value()) {
        std::printf("FAIL: UnifiedPoolSet::create\n");
        return 1;
    }

    // Allocate 8 MiB on every pooled device; verify the pointers are live
    // and DIFFERENT (different physical devices, not aliases).
    uint64_t firstPtr = 0;
    int pooled = 0;
    for (const auto& entry : set->devices()) {
        if (!entry.pool) {
            std::printf("info: '%s' pooled=false (needs app VkDevice)\n", entry.info.name);
            continue;
        }
        AllocDesc desc;
        desc.size = 8ull * 1024ull * 1024ull;
        desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        desc.memoryUsage = vvm::MemoryUsage::GpuOnly;
        auto a = entry.pool->allocate(desc);
        if (!a.has_value()) {
            std::printf("FAIL: allocate on '%s'\n", entry.info.name);
            ++failures;
            continue;
        }
        const uint64_t ptr = reinterpret_cast<uint64_t>(a->buffer);
        std::printf("ok: '%s' [%s] allocate -> %p\n",
                    entry.info.name, backend_kind_name(entry.info.preferred), (void*)ptr);
        if (firstPtr == 0) {
            firstPtr = ptr;
        } else if (ptr == firstPtr) {
            std::printf("FAIL: aliased pointer across devices\n");
            ++failures;
        }
        entry.pool->deallocate(std::move(*a));
        ++pooled;
    }
    if (pooled == 0) {
        std::printf("FAIL: no pooled devices (expected HIP + L0 pools)\n");
        ++failures;
    }

    std::printf("aggregate: %s\n", set->aggregate_stats_json());

    // Plan-driven routing: synthetic plan, no device needed. Mirrors the
    // llama pick_from_plan contract (positional expert indexing).
    {
        PlacementPlan plan;
        plan.denseBackend = MemBackendKind::Hip;
        plan.denseDeviceIndex = 1;
        plan.experts.resize(4);
        for (int i = 0; i < 4; ++i) {
            plan.experts[i].layer = i;
            plan.experts[i].onCpu = (i >= 2);
            plan.experts[i].backend = MemBackendKind::Hip;
            plan.experts[i].deviceIndex = 1;
        }
        auto r = route_plan(plan, TensorClass::LookupTable, -1);
        if (!r.wantCpu) { std::printf("FAIL: PLE must route CPU\n"); ++failures; }
        r = route_plan(plan, TensorClass::Expert, 0);
        if (r.wantCpu || r.kind != MemBackendKind::Hip || r.vendorIndex != 1) {
            std::printf("FAIL: GPU expert misrouted\n"); ++failures;
        }
        r = route_plan(plan, TensorClass::Expert, 3);
        if (!r.wantCpu) { std::printf("FAIL: CPU expert must route CPU\n"); ++failures; }
        r = route_plan(plan, TensorClass::Expert, 9);
        if (!r.wantCpu) { std::printf("FAIL: out-of-plan expert must route CPU\n"); ++failures; }
        r = route_plan(plan, TensorClass::Expert, -1);
        if (!r.wantCpu) { std::printf("FAIL: layerless expert must route CPU\n"); ++failures; }
        r = route_plan(plan, TensorClass::Dense, -1);
        if (r.wantCpu || r.kind != MemBackendKind::Hip || r.vendorIndex != 1) {
            std::printf("FAIL: dense misrouted\n"); ++failures;
        }
        r = route_plan(plan, TensorClass::Attention, -1);
        if (r.wantCpu || r.vendorIndex != 1) {
            std::printf("FAIL: attention must follow dense\n"); ++failures;
        }
        PlacementPlan empty;
        r = route_plan(empty, TensorClass::Dense, -1);
        if (!r.wantCpu) { std::printf("FAIL: missing dense must route CPU\n"); ++failures; }
        std::printf("route_plan: synthetic checks done\n");
    }

    if (failures == 0) {
        std::printf("ALL UNIFIED SET TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
