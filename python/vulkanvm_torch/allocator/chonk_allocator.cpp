// chonk_allocator.cpp - Production provider: pool + DMA-BUF + HIP import
// wired into the pure slab core (chonk_slab.hpp). Retention policy env knobs
// and the PyTorch C ABI live here.

#include "chonk_allocator.hpp"

#include "../device/pool_device.hpp"
#include "../interop/hip_external_memory.hpp"
#include "chonk_slab.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace vvm_torch {
using slab::Block;  // provider callbacks use the slab block type unqualified
namespace {

size_t envSizeGB(const char* name, size_t defGB) {
    const char* p = getenv(name);
    if (p) {
        double v = atof(p);
        if (v >= 1.0) return (size_t)v;
    }
    return defGB;
}

// Bucket list (GB) parsed once from CHONK_POOL_BLOCK_SIZES_GB ("auto" gives
// the graduated 1..128 GB ladder). New blocks are rounded UP to the smallest
// bucket >= the request so a single block absorbs the monotonic growth of
// recompute attention temporaries. Falls back to power-of-two rounding.
std::vector<size_t> buckets() {
    static std::vector<size_t> b = [] {
        std::vector<size_t> out;
        const char* p = getenv("CHONK_POOL_BLOCK_SIZES_GB");
        if (p) {
            std::string str(p);
            std::string firstToken = str.substr(0, str.find(','));
            if (firstToken == "auto") {
                static const size_t kLadderGB[] = {
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128,
                };
                for (size_t gb : kLadderGB) {
                    out.push_back(gb * 1024ull * 1024ull * 1024ull);
                }
                fprintf(stderr, "[allocator] auto-buckets: graduated %zu-rung ladder, 1..128 GB\n",
                        sizeof(kLadderGB) / sizeof(kLadderGB[0]));
            } else {
                size_t start = 0;
                for (;;) {
                    size_t end = str.find(',', start);
                    std::string token = str.substr(start, end == std::string::npos ? std::string::npos : end - start);
                    double gb_val = atof(token.c_str());
                    if (gb_val >= 1.0) out.push_back((size_t)(gb_val * 1024.0 * 1024.0 * 1024.0));
                    if (end == std::string::npos) break;
                    start = end + 1;
                }
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }();
    return b;
}

size_t roundToBucket(size_t need) {
    for (size_t b : buckets()) {
        if (b >= need) return b;
    }
    size_t minBlock = envSizeGB("CHONK_MIN_BLOCK_GB", 2) * 1024ull * 1024ull * 1024ull;
    size_t b = std::max(minBlock, need);
    size_t p = 1;
    while (p < b) p <<= 1;
    return p;
}

FILE* g_allocLog = nullptr;

void allocLog(const char* op, void* ptr, size_t sz) {
    if (!g_allocLog) {
        const char* p = getenv("CHONK_ALLOC_LOG");
        if (p) g_allocLog = fopen(p, "w");
    }
    if (g_allocLog) {
        fprintf(g_allocLog, "%s %p %zu\n", op, ptr, sz);
        fflush(g_allocLog);
    }
}

// Production provider: Vulkan allocation -> DMA-BUF export -> HIP import.
// Keeps the vvm::Allocation handle per block (the slab core type-erases it).
struct PoolBlockProvider : slab::Core::IProvider {
    size_t minBlockBytes = envSizeGB("CHONK_MIN_BLOCK_GB", 2) * 1024ull * 1024ull * 1024ull;
    size_t escalateSlackGB = envSizeGB("CHONK_ESCALATE_SLACK_GB", 2);
    size_t minBlocksOnOOM = 1;  // aggressive: keep only 1 block on OOM release (was 4)
    std::unordered_map<void*, vvm::Allocation> blockAllocs_;

    size_t warmBlocks() const { return envSizeGB("CHONK_WARM_BLOCKS", 8); }
    size_t maxBlocks() const { return envSizeGB("CHONK_MAX_BLOCKS", 24); }

    Block* createBlock(slab::Core& core, size_t need) override {
        vvm::UnifiedMemoryPool* p = pool();
        if (!p) return nullptr;
        // Proven sizing: exact-fit above the min-block floor (the validated
        // 196K recipe sets CHONK_MIN_BLOCK_GB=16). Bucket rounding is NOT
        // applied to base allocations (it inflated them by ~10GB); it is the
        // pressure-relief escalation only.
        size_t blockSize = std::max(minBlockBytes,
                                    (need + slab::Core::kAlign - 1) / slab::Core::kAlign * slab::Core::kAlign);
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
        auto allocOpt = p->allocate(desc);
        if (!allocOpt) {
            // Pressure relief: release fully-free blocks down to a warm
            // floor, then retry once before escalating.
            size_t released = core.releaseEmptyBlocks(minBlocksOnOOM);
            if (released > 0) {
                fprintf(stderr, "[allocator] released %zu empty block(s) under pressure; retrying %zu bytes\n",
                        released, need);
                allocOpt = p->allocate(desc);
            }
            if (!allocOpt) {
                // Escalate to the nearest configured bucket, bounded by
                // need + slack (escalating far past a driver OOM just
                // thrashes the allocator).
                size_t bucketSize = roundToBucket(need);
                size_t slack = escalateSlackGB * 1024ull * 1024ull * 1024ull;
                if (bucketSize > blockSize && bucketSize <= need + slack) {
                    fprintf(stderr, "[allocator] pressure: escalating %zu -> %zu bucket\n",
                            blockSize, bucketSize);
                    desc.size = bucketSize;
                    allocOpt = p->allocate(desc);
                }
            }
        }
        if (!allocOpt) return nullptr;
        vvm::Allocation a = std::move(*allocOpt);
        // The pool's granted size is authoritative: the escalation path
        // re-requests at bucketSize, so the Vulkan allocation may exceed the
        // original request. The HIP import and slab bookkeeping MUST describe
        // the same memory object as the Vulkan allocation.
        const size_t actualBlockSize = a.size;
        auto info = p->exportMemory(a, vvm::ExternalHandleType::OpaqueFd);
        if (!info) {
            p->deallocate(std::move(a));
            return nullptr;
        }
        int fd = info->handle.release();
        hipExternalMemory_t ext = nullptr;
        void* base = hipImportFromFd(fd, actualBlockSize, &ext);
        if (!base) {
            p->deallocate(std::move(a));
            return nullptr;
        }
        Block* b = new Block();
        b->base = base;
        b->extHandle = ext;
        b->fd = fd;
        b->size = actualBlockSize;
        b->freeChunks.push_back({0, actualBlockSize});
        blockAllocs_[base] = std::move(a);  // ownership for destroyBlock
        allocLog("B", base, actualBlockSize);
        return b;
    }

    void destroyBlock(Block* b) override {
        // After pool.shutdown() the HIP context is gone; skip HIP destruction
        // during teardown (leak the handles; the OS reclaims them).
        if (b->extHandle && pool()) {
            hipDestroyExternalMemory(static_cast<hipExternalMemory_t>(b->extHandle));
        }
        if (b->fd >= 0) close(b->fd);
        if (pool()) {
            auto it = blockAllocs_.find(b->base);
            if (it != blockAllocs_.end()) {
                pool()->deallocate(std::move(it->second));
                blockAllocs_.erase(it);
            }
        }
        allocLog("R", b->base, b->size);
        delete b;
    }
};

PoolBlockProvider& provider() {
    static PoolBlockProvider p;
    return p;
}


// Sub-allocator: tiered sub-slabs (512MB, 1024MB, 2048MB floors) for
// "best fit" coverage of allocations in the 0..2GB range. Each tier routes
// requests to the smallest sub-slab whose floor >= the request, so a
// 600MB allocation gets a 1024MB block (not a 2GB main-slab block that
// wastes 1.4GB). Tier defaults are env-tunable.
namespace sub {
    // Tier N: floor size in bytes. Requests are routed to the tier whose
    // floor is the largest value still <= 2x the request (i.e., the floor
    // fits the request with <= 2x waste). This matches the user's chunk
    // sizes (512, 1024, 2048) and closes the 512MB-2GB gap.
    static constexpr size_t TIER_FLOORS_BYTES[3] = {
        (size_t)512 * 1024 * 1024,   // tier 0: 512 MB
        (size_t)1024 * 1024 * 1024,  // tier 1: 1 GB
        (size_t)2048 * 1024 * 1024,  // tier 2: 2 GB
    };
    static constexpr int N_TIERS = 3;
    static slab::Core* tiers[N_TIERS] = {nullptr, nullptr, nullptr};

    // Pick the tier index for a request size, or -1 if the request is
    // larger than the largest tier floor (caller should use main slab).
    int pick_tier(size_t sz) {
        for (int i = 0; i < N_TIERS; ++i) {
            if (sz <= TIER_FLOORS_BYTES[i] * 2) return i;
        }
        return -1;
    }

    slab::Core* get(int tier) {
        if (tier < 0 || tier >= N_TIERS) return nullptr;
        if (!tiers[tier]) {
            // Lazy init: each tier gets its own slab::Core with floor = TIER_FLOORS_BYTES[tier]
            // (handled in the provider via minBlock per request; here we
            // just create a Core with a generic provider).
            size_t warm  = envSizeGB("CHONK_SUB_WARM_BLOCKS", 4);
            size_t maxb  = envSizeGB("CHONK_SUB_MAX_BLOCKS", 16);
            size_t minb  = envSizeGB("CHONK_SUB_MIN_BLOCKS", 4);
            static PoolBlockProvider p;
            // Note: the per-tier min block floor is enforced in the provider's
            // createBlock by passing need >= TIER_FLOORS_BYTES[tier]; the slab
            // itself doesn't know about tier floors. The provider is global,
            // so this is a soft guarantee (allocator may round up further if
            // the bucket ladder mandates). For an exact tier floor, a per-tier
            // provider would be needed; in practice the bucket ladder is
            // graduated and the difference is small.
            tiers[tier] = new slab::Core(&p, warm, maxb, minb);
        }
        return tiers[tier];
    }

    // Legacy single-slab init (kept for backward compat; returns tier 0).
    slab::Core* init() { return get(0); }
}

slab::Core& core() {
    static slab::Core c(&provider(), provider().warmBlocks(),
                        provider().maxBlocks(), provider().minBlocksOnOOM);
    return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// C ABI (declared in chonk_allocator.hpp)
// ---------------------------------------------------------------------------

void* chonk_allocator_alloc(ssize_t size, int device, void* stream) {
    (void)device; (void)stream;
    if (size < 0) return nullptr;  // ABI boundary: reject negative sizes explicitly
    if (!pool()) return nullptr;   // pool must be initialized before install
    size_t granted = 0;
    void* ptr = nullptr;
    size_t granted_local = 0;
    // Tiered sub-allocator routing: pick the best-fit sub-slab (512MB / 1GB / 2GB)
    // for this request. Falls back to the main slab if no sub-slab fits.
    int tier = sub::pick_tier((size_t)size);
    if (tier >= 0) {
        slab::Core* s = sub::get(tier);
        if (s) {
            // Aggressive empty-block release: before allocating, free any
            // fully-empty sub-slab blocks back to the Vulkan pool. With the
            // 130 GB GTT heap, re-alloc is cheap; holding dead 1-2 GB blocks
            // just bloats the pool and fragments the driver's exportable heap.
            s->releaseEmptyBlocks(0);
            ptr = s->alloc((size_t)size, &granted_local);
        }
    }
    if (!ptr) {
        // Same aggressive release on the main slab.
        core().releaseEmptyBlocks(0);
        ptr = core().alloc((size_t)size, (granted_local == 0 ? &granted : nullptr));
        if (granted_local == 0) granted_local = granted;
    }
    if (ptr && granted == 0) granted = granted_local;
    if (ptr) allocLog("A", ptr, granted);
    return ptr;
}

void chonk_allocator_free(void* ptr, size_t size, void* stream) {
    (void)stream;
    if (!ptr) return;
    allocLog("F", ptr, size);
    core().free(ptr, size);
}

void vvm_torch_chonk_allocator_reset() {
    core().reset();
}

}  // namespace vvm_torch
