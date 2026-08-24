// chonk_allocator.cpp - Slab allocator implementation over Chonk Buffer
// blocks. See chonk_allocator.hpp for the architecture description.

#include "chonk_allocator.hpp"

#include "../device/pool_device.hpp"
#include "../interop/hip_external_memory.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vvm_torch {

size_t ChonkAllocator::minBlock() {
    const char* p = getenv("CHONK_MIN_BLOCK_GB");
    if (p) {
        double gb = atof(p);
        if (gb >= 1.0) return (size_t)(gb * 1024.0 * 1024.0 * 1024.0);
    }
    return kMinBlock;
}

size_t ChonkAllocator::envSize(const char* name, size_t def) {
    const char* p = getenv(name);
    if (p) {
        double v = atof(p);
        if (v >= 1.0) return (size_t)v;
    }
    return def;
}

// Bucket list (GB) parsed once from CHONK_POOL_BLOCK_SIZES_GB.
// If the env var is set to "auto", buckets are generated as a graduated
// ladder (1 GB steps through 16 GB - where all transient training traffic
// lives, peak measured demand was 8 GiB + 16 MiB - then coarser rungs up to
// 128 GB). Otherwise the comma-separated list is used as-is. New blocks are
// rounded UP to the smallest bucket >= the request so a single block absorbs
// the monotonic growth of recompute attention temporaries. Falls back to
// power-of-two rounding.
std::vector<size_t> ChonkAllocator::buckets() {
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

size_t ChonkAllocator::roundToBucket(size_t need) {
    for (size_t b : buckets()) {
        if (b >= need) return b;
    }
    // No bucket fits: round up to the next power of two (min 2 GB).
    size_t b = std::max(minBlock(), need);
    size_t p = 1;
    while (p < b) p <<= 1;
    return p;
}

static ChonkAllocator g_allocator;
static FILE* g_allocLog = nullptr;

static void allocLog(const char* op, void* ptr, size_t sz) {
    if (!g_allocLog) {
        const char* p = getenv("CHONK_ALLOC_LOG");
        if (p) g_allocLog = fopen(p, "w");
    }
    if (g_allocLog) {
        fprintf(g_allocLog, "%s %p %zu\n", op, ptr, sz);
        fflush(g_allocLog);
    }
}

static void allocatorDestroyBlock(AllocBlock* b) {
    // After pool.shutdown() the HIP context is gone; skip HIP destruction
    // during teardown (leak the handles; the OS reclaims them).
    if (b->ext && pool()) hipDestroyExternalMemory(b->ext);
    b->base = nullptr;
    b->ext = nullptr;
    if (b->fd >= 0) close(b->fd);
    b->fd = -1;
    if (pool()) pool()->deallocate(std::move(b->alloc));
}

static bool allocatorCreateBlock(size_t need) {
    vvm::UnifiedMemoryPool* p = pool();
    if (!p) return false;
    // Proven sizing: exact-fit above the min-block floor (the validated
    // 196K recipe sets CHONK_MIN_BLOCK_GB=16). A 16GB block absorbs the
    // entire recompute-attention growth curve (2.4GB -> 9.7GB): the freed
    // chunk merges back to the full block and the next, slightly larger,
    // request reuses it - no new blocks, flat GTT. Bucket rounding is NOT
    // applied to base allocations (it inflated them by ~10GB). It is only
    // used below as the pressure-relief retry escalation.
    size_t blockSize = std::max(ChonkAllocator::minBlock(),
                                (need + ChonkAllocator::kAlign - 1) / ChonkAllocator::kAlign * ChonkAllocator::kAlign);
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
        // Pressure relief: we are at (or near) the driver's ceiling. Release
        // fully-free blocks down to a small warm floor, then retry once
        // before giving up (auto-allocation under pressure).
        size_t before = g_allocator.blocks.size();
        for (auto it = g_allocator.blocks.begin(); it != g_allocator.blocks.end();) {
            if ((*it)->liveBytes == 0 && g_allocator.blocks.size() > g_allocator.minBlocksOnOOM) {
                allocatorDestroyBlock(it->get());
                it = g_allocator.blocks.erase(it);
            } else {
                ++it;
            }
        }
        if (g_allocator.blocks.size() < before) {
            fprintf(stderr, "[allocator] released %zu empty block(s) under pressure; retrying %zu bytes\n",
                    before - g_allocator.blocks.size(), need);
            allocOpt = p->allocate(desc);
        }
        if (!allocOpt) {
            // Second escalation: re-round the block up to the nearest
            // configured bucket (CHONK_POOL_BLOCK_SIZES_GB; with the
            // graduated auto-ladder this is a tight fit, e.g. 8.02 GB ->
            // 9 GB rung). Bound the overshoot: escalating far past `need`
            // after a driver OOM is guaranteed to fail again and just
            // thrashes the allocator, so skip escalation when the next
            // rung exceeds need + slack.
            size_t bucketSize = ChonkAllocator::roundToBucket(need);
            size_t slack = ChonkAllocator::envSize("CHONK_ESCALATE_SLACK_GB", 2)
                           * 1024ull * 1024ull * 1024ull;
            if (bucketSize > blockSize && bucketSize <= need + slack) {
                fprintf(stderr, "[allocator] pressure: escalating %zu -> %zu bucket\n", blockSize, bucketSize);
                desc.size = bucketSize;
                allocOpt = p->allocate(desc);
            }
        }
    }
    if (!allocOpt) return false;
    vvm::Allocation a = std::move(*allocOpt);
    // The pool's granted size is authoritative: the pressure-escalation path
    // above re-requests at bucketSize, so the Vulkan allocation may be larger
    // than the original `blockSize`. The HIP import and slab bookkeeping MUST
    // describe the same memory object as the Vulkan allocation.
    const size_t actualBlockSize = a.size;
    auto info = p->exportMemory(a, vvm::ExternalHandleType::OpaqueFd);
    if (!info) {
        p->deallocate(std::move(a));
        return false;
    }
    int fd = info->handle.release();
    hipExternalMemory_t ext = nullptr;
    void* base = hipImportFromFd(fd, actualBlockSize, &ext);
    if (!base) {
        p->deallocate(std::move(a));
        return false;
    }
    auto block = std::make_unique<AllocBlock>();
    block->alloc = std::move(a);
    block->fd = fd;
    block->ext = ext;
    block->base = base;
    block->size = actualBlockSize;
    block->freeChunks.push_back({0, actualBlockSize});
    g_allocator.blocks.push_back(std::move(block));
    allocLog("B", base, actualBlockSize);
    return true;
}

static void allocatorMaybeReleaseEmptyBlock(AllocBlock* blk) {
    if (blk->liveBytes != 0) return;
    if (g_allocator.blocks.size() <= g_allocator.warmBlocks) return;
    for (auto it = g_allocator.blocks.begin(); it != g_allocator.blocks.end(); ++it) {
        if (it->get() == blk) {
            allocatorDestroyBlock(blk);
            g_allocator.blocks.erase(it);
            return;
        }
    }
}

void* chonk_allocator_alloc_impl(ssize_t size, int device, void* stream) {
    (void)device; (void)stream;
    if (size < 0) return nullptr;  // ABI boundary: reject negative sizes explicitly
    if (!pool()) return nullptr;  // pool must be initialized before install
    std::lock_guard<std::mutex> lock(g_allocator.mtx);
    size_t aligned = ((size_t)size + ChonkAllocator::kAlign - 1) &
                     ~(ChonkAllocator::kAlign - 1);
    if (aligned == 0) aligned = ChonkAllocator::kAlign;  // hipMalloc(0) semantics

    // First fit across existing blocks (alignment-aware).
    for (auto& blk : g_allocator.blocks) {
        for (auto it = blk->freeChunks.begin(); it != blk->freeChunks.end(); ++it) {
            size_t off = it->first;
            size_t chunkSz = it->second;
            size_t alignedOff = (off + ChonkAllocator::kAlign - 1) &
                                ~(ChonkAllocator::kAlign - 1);
            size_t headSlack = alignedOff - off;
            if (chunkSz < headSlack + aligned) continue;
            void* ptr = (char*)blk->base + alignedOff;
            blk->liveBytes += aligned;
            g_allocator.liveSizes[ptr] = aligned;
            size_t tail = chunkSz - headSlack - aligned;
            if (headSlack > 0) {
                it->second = headSlack;  // keep head slack as free
                if (tail > 0) {
                    blk->freeChunks.insert(it + 1, {alignedOff + aligned, tail});
                }
            } else if (tail > 0) {
                it->first = alignedOff + aligned;
                it->second = tail;
            } else {
                blk->freeChunks.erase(it);
            }
            allocLog("A", ptr, aligned);
            return ptr;
        }
    }

    // No fit: create a new block sized for the request.
    allocLog("N", nullptr, aligned);
    if (!allocatorCreateBlock(aligned)) return nullptr;
    auto& blk = g_allocator.blocks.back();
    auto it = blk->freeChunks.begin();
    blk->liveBytes += aligned;
    void* ptr = (char*)blk->base;
    g_allocator.liveSizes[ptr] = aligned;
    if (blk->size > aligned) {
        it->first = aligned;
        it->second = blk->size - aligned;
    } else {
        blk->freeChunks.erase(it);
    }
    allocLog("A", ptr, aligned);
    return ptr;
}

void chonk_allocator_free_impl(void* ptr, size_t size, void* stream) {
    (void)stream;
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(g_allocator.mtx);
    size_t sz = size;
    auto lsIt = g_allocator.liveSizes.find(ptr);
    if (lsIt != g_allocator.liveSizes.end()) {
        sz = lsIt->second;
        g_allocator.liveSizes.erase(lsIt);
    }
    if (sz == 0) return;

    for (auto& blk : g_allocator.blocks) {
        uintptr_t b = (uintptr_t)blk->base;
        uintptr_t p = (uintptr_t)ptr;
        if (p < b || p + sz > b + blk->size) continue;
        size_t coff = p - b;
        blk->liveBytes -= sz;
        auto& fc = blk->freeChunks;
        auto pos = std::lower_bound(fc.begin(), fc.end(), coff,
            [](const std::pair<size_t, size_t>& c, size_t v) { return c.first < v; });
        // Overlap guards: if the region is already covered by neighbors,
        // this is a double free -- ignore it rather than corrupt the list.
        if (pos != fc.begin()) {
            auto prev = pos - 1;
            if (prev->first + prev->second >= coff + sz) {
                blk->liveBytes += sz;
                allocLog("D", ptr, sz);
                return;
            }
        }
        if (pos != fc.end() && pos->first <= coff) {
            blk->liveBytes += sz;
            allocLog("D", ptr, sz);
            return;
        }
        // Merge with previous chunk.
        if (pos != fc.begin()) {
            auto prev = pos - 1;
            if (prev->first + prev->second == coff) {
                prev->second += sz;
                if (pos != fc.end() && prev->first + prev->second == pos->first) {
                    prev->second += pos->second;
                    fc.erase(pos);
                }
                allocatorMaybeReleaseEmptyBlock(blk.get());
                allocLog("F", ptr, sz);
                return;
            }
        }
        // Merge with next chunk.
        if (pos != fc.end() && coff + sz == pos->first) {
            pos->first = coff;
            pos->second += sz;
        } else {
            fc.insert(pos, {coff, sz});
        }
        allocatorMaybeReleaseEmptyBlock(blk.get());
        allocLog("F", ptr, sz);
        return;
    }
    // Unknown pointer: ignore (torch may free pointers from other allocators).
    allocLog("U", ptr, size);
    fprintf(stderr, "[allocator] WARN: free of UNKNOWN ptr=%p size=%zu\n", ptr, size);
}

void vvm_torch_chonk_allocator_reset() {
    vvm_torch::chonk_allocator_reset_impl();
}

void vvm_torch::chonk_allocator_reset_impl() {
    std::lock_guard<std::mutex> lock(g_allocator.mtx);
    g_allocator.blocks.clear();
    g_allocator.liveSizes.clear();
}

}  // namespace vvm_torch

// C ABI (declared extern "C" in chonk_allocator.hpp)
void* chonk_allocator_alloc(ssize_t size, int device, void* stream) {
    return vvm_torch::chonk_allocator_alloc_impl(size, device, stream);
}

void chonk_allocator_free(void* ptr, size_t size, void* stream) {
    vvm_torch::chonk_allocator_free_impl(ptr, size, stream);
}


