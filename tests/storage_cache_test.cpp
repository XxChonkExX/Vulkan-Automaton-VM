// storage_cache_test: L1 cache core invariants + fuzz (CPU-only).
// Exercises the residency state machine, atomic pinning, and all three
// eviction policies. Mirrors the chonk_slab_test invariant-checking style.

#include "vulkan_vm/storage/cache.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace vvm::storage::cache;

static void testBasic() {
    CacheConfig cfg;
    cfg.capacity = 2;
    ShardCache c(cfg);

    auto p0 = c.probe(100);
    assert(p0.kind == ShardCache::Lookup::Miss);
    c.complete(p0.slot);
    c.pin(p0.slot);
    assert(c.state(p0.slot) == LineState::Valid);
    assert(c.refcount(p0.slot) == 1);

    auto p0again = c.probe(100);
    assert(p0again.kind == ShardCache::Lookup::Hit);
    assert(p0again.slot == p0.slot);

    c.unpin(p0.slot);
    assert(c.refcount(p0.slot) == 0);
    std::cout << "  basic miss/hit/pin ok\n";
}

static void testInFlight() {
    ShardCache c({2, 20, EvictionKind::Clock});
    auto p0 = c.probe(42);
    assert(p0.kind == ShardCache::Lookup::Miss);
    // Before complete(), a second probe of the same key must report InFlight
    // (coalescing seam), not a second Miss.
    auto p1 = c.probe(42);
    assert(p1.kind == ShardCache::Lookup::InFlight);
    assert(p1.slot == p0.slot);
    c.complete(p0.slot);
    assert(c.probe(42).kind == ShardCache::Lookup::Hit);
    std::cout << "  in-flight coalescing ok\n";
}

static void testEvictionPinned() {
    ShardCache c({2, 20, EvictionKind::Clock});
    auto a = c.probe(1); c.complete(a.slot); c.pin(a.slot);
    auto b = c.probe(2); c.complete(b.slot);
    // capacity 2, line 1 pinned. probing 3 must evict line 2, not line 1.
    auto x = c.probe(3);
    assert(x.kind == ShardCache::Lookup::Miss);
    assert(x.evictedKey == 2);
    assert(c.state(a.slot) == LineState::Valid);   // pinned survives
    c.complete(x.slot);
    // line 2 now gone, line 1 + line 3 resident.
    assert(c.probe(1).kind == ShardCache::Lookup::Hit);
    assert(c.probe(3).kind == ShardCache::Lookup::Hit);
    c.unpin(a.slot);
    std::cout << "  pinned-line eviction skip ok\n";
}

static void testDirty() {
    ShardCache c({4, 20, EvictionKind::Lru});
    auto a = c.probe(10);
    c.complete(a.slot);
    assert(!c.isDirty(a.slot));
    c.markDirty(a.slot);
    assert(c.isDirty(a.slot));
    c.markClean(a.slot);
    assert(!c.isDirty(a.slot));
    std::cout << "  dirty mark/clean ok\n";
}

static void testPolicies() {
    for (auto k : {EvictionKind::Clock, EvictionKind::Lru, EvictionKind::Fifo}) {
        ShardCache c({3, 20, k});
        uint32_t s0 = c.probe(0).slot; c.complete(s0);
        uint32_t s1 = c.probe(1).slot; c.complete(s1);
        uint32_t s2 = c.probe(2).slot; c.complete(s2);
        auto x = c.probe(3);
        assert(x.kind == ShardCache::Lookup::Miss); // some victim was evicted
        assert(x.evictedKey != UINT64_MAX);
        c.complete(x.slot);
        // exactly 3 of 4 keys resident
        assert(c.residentCount() == 3);
    }
    std::cout << "  all policies evict ok\n";
}

static void testFuzz() {
    std::mt19937 rng(12345);
    ShardCache c({8, 20, EvictionKind::Clock});

    for (int step = 0; step < 20000; ++step) {
        const uint64_t key = rng() % 32; // 32 distinct shard ids
        uint8_t op = static_cast<uint8_t>(rng() % 4);
        switch (op) {
            case 0: { // probe (and complete)
                auto p = c.probe(key);
                if (p.kind == ShardCache::Lookup::Miss) c.complete(p.slot);
                break;
            }
            case 1: { // pin on hit
                auto p = c.probe(key);
                if (p.kind == ShardCache::Lookup::Hit) c.pin(p.slot);
                break;
            }
            case 2: { // unpin any valid slot
                for (uint32_t s = 0; s < c.capacity(); ++s)
                    if (c.state(s) == LineState::Valid && c.refcount(s) > 0) c.unpin(s);
                break;
            }
            case 3: { // explicit evict of some valid slot
                for (uint32_t s = 0; s < c.capacity(); ++s)
                    if (c.state(s) == LineState::Valid && c.refcount(s) == 0) { c.evict(s); break; }
                break;
            }
        }

        // invariants
        uint32_t valid = 0;
        for (uint32_t s = 0; s < c.capacity(); ++s) {
            assert(c.refcount(s) <= 64); // no runaway pinning in this harness
            if (c.state(s) == LineState::Valid) ++valid;
        }
        assert(valid == c.residentCount());
    }
    std::cout << "  fuzz invariants ok\n";
}

int main() {
    std::cout << "storage_cache_test (CPU-only)\n";
    testBasic();
    testInFlight();
    testEvictionPinned();
    testDirty();
    testPolicies();
    testFuzz();
    std::cout << "All storage_cache tests passed!\n";
    return 0;
}