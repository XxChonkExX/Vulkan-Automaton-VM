// chonk_slab.cpp - Slab allocator core implementation. See chonk_slab.hpp.

#include "chonk_slab.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

namespace vvm_torch {
namespace slab {

Core::Core(IProvider* provider, size_t warmBlocks, size_t maxBlocks,
           size_t minBlocksOnOOM)
    : provider_(provider),
      warmBlocks_(warmBlocks),
      maxBlocks_(maxBlocks),
      minBlocksOnOOM_(minBlocksOnOOM) {}

void* Core::alloc(size_t size, size_t* grantedSize) {
    size_t aligned = ((size + kAlign - 1) / kAlign) * kAlign;
    if (aligned == 0) aligned = kAlign;  // hipMalloc(0) semantics
    if (grantedSize) *grantedSize = aligned;

    // First fit across existing blocks (alignment-aware).
    for (Block* blk : blocks_) {
        for (auto it = blk->freeChunks.begin(); it != blk->freeChunks.end(); ++it) {
            const size_t off = it->first;
            const size_t chunkSz = it->second;
            const size_t alignedOff = ((off + kAlign - 1) / kAlign) * kAlign;
            const size_t headSlack = alignedOff - off;
            if (chunkSz < headSlack + aligned) continue;
            void* ptr = static_cast<char*>(blk->base) + alignedOff;
            blk->liveBytes += aligned;
            liveSizes_[ptr] = aligned;
            const size_t tail = chunkSz - headSlack - aligned;
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
            return ptr;
        }
    }

    // No fit: ask the provider for a new block.
    if (!provider_) return nullptr;
    Block* blk = provider_->createBlock(*this, aligned);
    if (blk == nullptr) return nullptr;
    // Block bases MUST be kAlign-aligned: all intra-block offsets are aligned
    // relative to base, so an unaligned base silently misaligns every
    // allocation in the block. Reject loudly instead of corrupting alignment.
    if (reinterpret_cast<uintptr_t>(blk->base) % kAlign != 0) {
        std::fprintf(stderr,
                     "chonk_slab: createBlock rejected - provider base %p is not "
                     "%zu-byte aligned\n",
                     blk->base, kAlign);
        provider_->destroyBlock(blk);
        return nullptr;
    }
    blocks_.push_back(blk);
    auto& fc = blk->freeChunks;
    if (fc.empty() || fc.front().second < aligned) {
        // Provider handed us a block smaller than the request: fail cleanly.
        // (Should not happen with well-formed providers.)
        provider_->destroyBlock(blk);
        blocks_.pop_back();
        return nullptr;
    }
    blk->liveBytes += aligned;
    void* ptr = static_cast<char*>(blk->base);
    liveSizes_[ptr] = aligned;
    if (blk->size > aligned) {
        fc.front().first = aligned;
        fc.front().second = blk->size - aligned;
    } else {
        fc.erase(fc.begin());
    }
    return ptr;
}

void Core::free(void* ptr, size_t sizeHint) {
    if (!ptr) return;
    size_t sz = sizeHint;
    auto lsIt = liveSizes_.find(ptr);
    if (lsIt != liveSizes_.end()) {
        sz = lsIt->second;
        liveSizes_.erase(lsIt);
    }
    if (sz == 0) return;

    for (Block* blk : blocks_) {
        const uintptr_t b = reinterpret_cast<uintptr_t>(blk->base);
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        if (p < b || p + sz > b + blk->size) continue;
        const size_t coff = p - b;
        blk->liveBytes -= sz;
        auto& fc = blk->freeChunks;
        auto pos = std::lower_bound(fc.begin(), fc.end(), coff,
            [](const std::pair<size_t, size_t>& c, size_t v) { return c.first < v; });
        // Double-free guard: region already covered by neighbors -> ignore.
        if (pos != fc.begin()) {
            auto prev = pos - 1;
            if (prev->first + prev->second >= coff + sz) {
                blk->liveBytes += sz;
                return;
            }
        }
        if (pos != fc.end() && pos->first <= coff) {
            blk->liveBytes += sz;
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
        return;
    }
    // Unknown pointer: ignore (torch may free pointers from other allocators).
}

size_t Core::releaseEmptyBlocks(size_t keepFloor) {
    size_t released = 0;
    for (auto it = blocks_.begin(); it != blocks_.end();) {
        if ((*it)->liveBytes == 0 && blocks_.size() > keepFloor) {
            provider_->destroyBlock(*it);
            it = blocks_.erase(it);
            ++released;
        } else {
            ++it;
        }
    }
    return released;
}

bool Core::checkInvariants(std::string* err, bool deep) const {
    auto fail = [&](const std::string& msg) {
        if (err) *err = msg;
        return false;
    };

    uint64_t globalLive = 0;
    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        const Block* blk = blocks_[bi];
        const std::string tag = "[block " + std::to_string(bi) + "] ";

        // I4: sum(free) + liveBytes == size
        size_t freeSum = 0;
        for (size_t ci = 0; ci < blk->freeChunks.size(); ++ci) {
            const auto& c = blk->freeChunks[ci];
            // I1: sorted, non-empty, in range
            if (c.second == 0) return fail(tag + "zero-size free chunk");
            if (c.first + c.second > blk->size) {
                return fail(tag + "free chunk out of range");
            }
            if (ci > 0) {
                const auto& prev = blk->freeChunks[ci - 1];
                // I1: sorted
                if (c.first <= prev.first) return fail(tag + "free chunks not sorted");
                // I2: no overlap
                if (c.first < prev.first + prev.second) {
                    return fail(tag + "overlapping free chunks");
                }
                // I3: fully coalesced
                if (c.first == prev.first + prev.second) {
                    return fail(tag + "adjacent free chunks not coalesced");
                }
            }
            freeSum += c.second;
        }
        if (freeSum + blk->liveBytes != blk->size) {
            return fail(tag + "free(" + std::to_string(freeSum) + ") + live(" +
                        std::to_string(blk->liveBytes) + ") != size(" +
                        std::to_string(blk->size) + ")");
        }
        globalLive += blk->liveBytes;
    }

    if (!deep) return true;
    // I5: every live pointer inside exactly one block; per-block sums match.
    std::vector<size_t> perBlock(blocks_.size(), 0);
    for (const auto& kv : liveSizes_) {
        const uintptr_t p = reinterpret_cast<uintptr_t>(kv.first);
        const uintptr_t end = p + kv.second;
        size_t owners = 0;
        size_t ownerIdx = 0;
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            const uintptr_t b = reinterpret_cast<uintptr_t>(blocks_[bi]->base);
            if (p >= b && end <= b + blocks_[bi]->size) {
                ++owners;
                ownerIdx = bi;
                if (kv.first == blocks_[bi]->base) {
                    // base pointer allocation is fine
                }
            }
        }
        if (owners != 1) {
            return fail("live pointer in " + std::to_string(owners) +
                        " blocks (expected 1)");
        }
        perBlock[ownerIdx] += kv.second;
    }
    for (size_t bi = 0; bi < blocks_.size(); ++bi) {
        if (perBlock[bi] != blocks_[bi]->liveBytes) {
            return fail("[block " + std::to_string(bi) + "] liveBytes(" +
                        std::to_string(blocks_[bi]->liveBytes) + ") != live map sum(" +
                        std::to_string(perBlock[bi]) + ")");
        }
    }
    globalLive = 0;
    (void)globalLive;
    return true;
}

Core::Stats Core::stats() const {
    Stats s;
    s.blocks = blocks_.size();
    s.allocations = liveSizes_.size();
    for (const Block* blk : blocks_) {
        s.capacityBytes += blk->size;
        s.liveBytes += blk->liveBytes;
        for (const auto& c : blk->freeChunks) s.freeBytes += c.second;
    }
    return s;
}

void Core::reset() {
    for (Block* blk : blocks_) {
        provider_->destroyBlock(blk);
    }
    blocks_.clear();
    liveSizes_.clear();
}

}  // namespace slab
}  // namespace vvm_torch



