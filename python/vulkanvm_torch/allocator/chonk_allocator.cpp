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

// Bucket ladder from 1 MB to 512 GB. New blocks are rounded UP to the
// smallest bucket >= the request. One unified freelist across all block
// sizes (slab::Core best-fit) so 1 KB to 131K-context allocations share
// the same pool, with no tier-routing layer in between. Defaults to
// "auto" (the full ladder). Override with CHONK_POOL_BLOCK_SIZES_GB.
//
// Ladder layout (per user spec, no shortcuts):
//   MB scale:
//     1, 1.5, 2, 3, 4 MB                          (fine for activations, gradients)
//     6, 8, 10, 12, ..., 16382 MB (2 MB steps)   (continuous to 16 GB)
//   GB scale:
//     16, 18, 20, ..., 128 GB (2 GB steps)        (KV cache, p/scores)
//   Standard sizing:
//     128, 192, 256, 384, 512 GB                 (1.5x; huge contexts)
//
// The 1.5 MB rung catches the "odd duck" between 1 and 2 MB. From 4 MB
// upward, 2-unit MB steps give <= 2 MB of waste. From 16 GB upward,
// 2-unit GB steps give <= 2 GB of waste. Beyond 128 GB, 1.5x standard
// sizing (matches the GTT heap's power-of-2 fragment).
std::vector<size_t> buckets() {
    static std::vector<size_t> b = [] {
        std::vector<size_t> out;
        const char* p = getenv("CHONK_POOL_BLOCK_SIZES_GB");
        if (p && std::string(p) != "auto") {
            // Custom ladder, in GB, parsed like "1,2,4,8,16".
            std::string str(p);
            size_t start = 0;
            for (;;) {
                size_t end = str.find(',', start);
                std::string token = str.substr(start, end == std::string::npos ? std::string::npos : end - start);
                double gb_val = atof(token.c_str());
                if (gb_val >= 1.0) out.push_back((size_t)(gb_val * 1024.0 * 1024.0 * 1024.0));
                if (end == std::string::npos) break;
                start = end + 1;
            }
        } else {
            // Default: the full user-specified ladder.
            //   1, 1.5, 2, 3, 4 MB  (fine MB rungs)
            out.push_back(1 * 1024 * 1024);
            out.push_back(1 * 1024 * 1024 + 512 * 1024);  // 1.5 MB
            out.push_back(2 * 1024 * 1024);
            out.push_back(3 * 1024 * 1024);
            out.push_back(4 * 1024 * 1024);
            //   6, 8, 10, ..., 16382 MB  (2 MB steps, continuous to 16 GB)
            for (size_t mb = 6; mb <= 16382; mb += 2) {
                out.push_back(mb * 1024 * 1024);
            }
            //   16, 18, 20, ..., 128 GB  (2 GB steps)
            for (size_t gb = 16; gb <= 128; gb += 2) {
                out.push_back((size_t)gb * 1024 * 1024 * 1024);
            }
            //   192, 256, 384, 512 GB  (1.5x standard, beyond 128)
            out.push_back((size_t)192 * 1024 * 1024 * 1024);
            out.push_back((size_t)256 * 1024 * 1024 * 1024);
            out.push_back((size_t)384 * 1024 * 1024 * 1024);
            out.push_back((size_t)512 * 1024 * 1024 * 1024);
            fprintf(stderr,
                    "[allocator] auto-buckets: user-spec ladder, %zu rungs, 1 MB .. 512 GB "
                    "(1,1.5,2,3,4,6,8..16GB step 2MB, 16..128GB step 2GB, 192..512GB 1.5x)\n",
                    out.size());
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }();
    return b;
}

// Read CHONK_MIN_BLOCK_GB and (override) CHONK_MIN_BLOCK_MB. Default 1 MB.
static size_t computeMinBlockBytes() {
    if (const char* mb = getenv("CHONK_MIN_BLOCK_MB")) {
        double v = atof(mb);
        if (v >= 0.5) return (size_t)(v * 1024.0 * 1024.0);
    }
    return envSizeGB("CHONK_MIN_BLOCK_GB", 0.001) * 1024ull * 1024ull * 1024ull;  // 1 MB
}

// Round a request size UP to the smallest ladder rung that fits it. Falls
// back to max(minBlock, need) rounded up to the next power of two if the
// request exceeds all rungs (defensive; should not happen for our workloads).
size_t roundToBucket(size_t need) {
    for (size_t b : buckets()) {
        if (b >= need) return b;
    }
    size_t minBlock = computeMinBlockBytes();
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
    size_t minBlockBytes = computeMinBlockBytes();  // default 1 MB
    size_t escalateSlackGB = envSizeGB("CHONK_ESCALATE_SLACK_GB", 2);
    size_t minBlocksOnOOM = 2;  // pressure relief keeps 2 warm blocks (validated design)
    std::unordered_map<void*, vvm::Allocation> blockAllocs_;

    size_t warmBlocks() const { return envSizeGB("CHONK_WARM_BLOCKS", 8); }
    size_t maxBlocks() const { return envSizeGB("CHONK_MAX_BLOCKS", 24); }

    Block* createBlock(slab::Core& core, size_t need) override {
        vvm::UnifiedMemoryPool* p = pool();
        if (!p) return nullptr;
        // Size the new block to the nearest ladder rung >= max(need, minBlock).
        // For a 1 KB request with min=1 MB, we get a 1 MB block; for a 700 MB
        // request, a 1 GB block; for 14 GB, a 16 GB block. The slab's best-fit
        // then finds the smallest free chunk across all blocks (any size),
        // so the ladder is just the set of block sizes the pool may grow to.
        size_t blockSize = roundToBucket(
            std::max(minBlockBytes,
                     (need + slab::Core::kAlign - 1) / slab::Core::kAlign * slab::Core::kAlign));
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
    // NO per-alloc empty-block release here. Every slab block is a dedicated
    // exportable allocation (vkAllocateMemory + dma-buf + HIP import), so
    // releasing and re-creating blocks on the hot path round-trips the kernel
    // and driver per chunk — the block churn that destabilized long runs.
    // Warm blocks are retained; pressure relief happens only inside
    // createBlock's retry path (releaseEmptyBlocks(minBlocksOnOOM)).
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
