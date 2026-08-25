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
    size_t minBlocksOnOOM = 4;
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
    void* ptr = core().alloc((size_t)size, &granted);
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
