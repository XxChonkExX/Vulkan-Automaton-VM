// pool_test: dedicated coverage for UnifiedMemoryPool — the 2.3k-line
// allocator at the heart of the Chonk Buffer.
//
// Coverage targets (all CPU-side policy, no kernel work):
//   - create/discovery over a creatable backend (probes HIP then L0;
//     skips cleanly when neither runtime is present)
//   - sub-allocation: distinct offsets in one block, deallocate returns
//     space, stats bookkeeping
//   - exhaustion: maxBlocks=1 + oversized request -> dedicated fallback
//   - budget: maxPoolBytes hard cap refuses growth
//   - chunk-tier config accepted + serves small allocs

#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/mem_backend.hpp"

#include <cstdio>
#include <optional>
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

// Probe HIP then Level Zero; return the first kind that can actually create
// (device present + runtime loadable). nullopt when nothing is creatable.
static std::optional<MemBackendKind> creatableBackend() {
    for (MemBackendKind k : {MemBackendKind::Hip, MemBackendKind::Level0}) {
        DeviceConfig probe{};
        probe.backendDeviceIndex = 0;
        probe.memBackendKind = static_cast<int32_t>(k);
        auto pool = UnifiedMemoryPool::create(probe, PoolConfig{});
        if (pool.has_value()) {
            return k;
        }
    }
    return std::nullopt;
}

static PoolConfig smallCfg() {
    PoolConfig pc;
    pc.blockSize = 64ull * 1024ull * 1024ull;   // 64 MiB blocks
    pc.maxBlocks = 4;
    pc.enableHostVisible = false;
    pc.enableExternal = false;
    pc.enableDeviceAddress = false;
    return pc;
}

static DeviceConfig cfgFor(MemBackendKind kind) {
    DeviceConfig dc{};
    dc.memBackendKind = static_cast<int32_t>(kind);
    dc.backendDeviceIndex = 0;
    return dc;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    std::printf("== pool_test ==\n");

    const auto kind = creatableBackend();
    if (!kind.has_value()) {
        std::printf("SKIP: no creatable memory backend on this box\n");
        return 0;
    }
    std::printf("using backend: %s\n",
                kind == MemBackendKind::Hip ? "hip" : "level0");

    // ------------------------------------------------------------------
    // 1. create + discovery sanity
    // ------------------------------------------------------------------
    {
        auto pool = UnifiedMemoryPool::create(cfgFor(*kind), smallCfg());
        CHECK(pool.has_value(), "pool create");
        if (pool.has_value()) {
            const PoolStats st = pool->getStats();
            CHECK(st.blockCount >= 1, "initial block allocated");
            CHECK(st.totalCapacity >= 64ull * 1024ull * 1024ull, "capacity >= blockSize");
        }
    }

    // ------------------------------------------------------------------
    // 2. sub-allocation lifecycle
    // ------------------------------------------------------------------
    auto poolOpt = UnifiedMemoryPool::create(cfgFor(*kind), smallCfg());
    if (!poolOpt.has_value()) {
        std::printf("FAIL: could not create pool for lifecycle section\n");
        return 1;
    }
    {
        auto & pool = *poolOpt;

        AllocDesc d8{}, d16{};
        d8.size = 8ull * 1024ull * 1024ull;
        d8.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        d8.memoryUsage = vvm::MemoryUsage::GpuOnly;
        d16 = d8;
        d16.size = 16ull * 1024ull * 1024ull;

        auto a = pool.allocate(d8);
        CHECK(a.has_value(), "allocate 8 MiB");
        auto b = pool.allocate(d16);
        CHECK(b.has_value(), "allocate 16 MiB");
        if (a && b) {
            CHECK(a->buffer != b->buffer, "distinct sub-allocations");
            CHECK(a->offset != b->offset, "distinct offsets");
            CHECK(a->blockIndex == b->blockIndex, "packed into same 64 MiB block");
        }

        const PoolStats mid = pool.getStats();
        CHECK(mid.allocationCount >= 2, "stats count sub-allocations");
        CHECK(mid.totalUsed >= 24ull * 1024ull * 1024ull, "stats track used bytes");

        if (a) pool.deallocate(std::move(*a));
        if (b) pool.deallocate(std::move(*b));
        const PoolStats end = pool.getStats();
        CHECK(end.allocationCount == 0 || end.totalUsed < mid.totalUsed,
              "freed space visible in stats");

        auto c = pool.allocate(d8);
        CHECK(c.has_value(), "re-allocate 8 MiB after free");
        if (c) pool.deallocate(std::move(*c));
    }

    // ------------------------------------------------------------------
    // 3. exhaustion -> dedicated fallback
    // ------------------------------------------------------------------
    {
        DeviceConfig dc1 = cfgFor(*kind);
        PoolConfig pc = smallCfg();
        pc.maxBlocks = 1;                        // single 64 MiB block
        auto p1 = UnifiedMemoryPool::create(dc1, pc);
        CHECK(p1.has_value(), "exhaustion pool create");
        if (p1.has_value()) {
            AllocDesc big{};
            big.size = 128ull * 1024ull * 1024ull;   // > block size
            big.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            big.memoryUsage = vvm::MemoryUsage::GpuOnly;
            auto dedicated = p1->allocate(big);
            CHECK(dedicated.has_value(), "oversized -> dedicated fallback");
            if (dedicated) {
                CHECK(dedicated->blockIndex == UINT32_MAX, "marked dedicated");
                p1->deallocate(std::move(*dedicated));
            }
        }
    }

    // ------------------------------------------------------------------
    // 4. budget: maxPoolBytes hard cap
    // ------------------------------------------------------------------
    {
        DeviceConfig dc2 = cfgFor(*kind);
        PoolConfig pc = smallCfg();
        pc.maxPoolBytes = 96ull * 1024ull * 1024ull;   // 96 MiB hard cap
        auto p2 = UnifiedMemoryPool::create(dc2, pc);
        CHECK(p2.has_value(), "budget pool create");
        if (p2.has_value()) {
            AllocDesc d64{}, d64b{};
            d64.size = 64ull * 1024ull * 1024ull;
            d64.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            d64.memoryUsage = vvm::MemoryUsage::GpuOnly;
            d64b = d64;
            auto x = p2->allocate(d64);
            CHECK(x.has_value(), "first 64 MiB under cap");
            auto y = p2->allocate(d64b);
            // Second 64 MiB would take the pool to 128 MiB > 96 MiB cap.
            // Refusal is the CORRECT outcome (budget discipline).
            CHECK(!y.has_value(), "second 64 MiB refused by maxPoolBytes");
        }
    }

    // ------------------------------------------------------------------
    // 5. chunk-tier config accepted + serves small allocs
    // ------------------------------------------------------------------
    {
        DeviceConfig dc3 = cfgFor(*kind);
        PoolConfig pc = smallCfg();
        pc.smallAllocThreshold = 16ull * 1024ull * 1024ull;
        pc.chunkBlockSize = 16ull * 1024ull * 1024ull;
        auto p3 = UnifiedMemoryPool::create(dc3, pc);
        CHECK(p3.has_value(), "chunk-tier pool create");
        if (p3.has_value()) {
            AllocDesc small{};
            small.size = 2ull * 1024ull * 1024ull;
            small.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            small.memoryUsage = vvm::MemoryUsage::GpuOnly;
            auto s = p3->allocate(small);
            CHECK(s.has_value(), "2 MiB via chunk tiers");
            if (s) p3->deallocate(std::move(*s));
        }
    }

    std::printf("\npool_test: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
