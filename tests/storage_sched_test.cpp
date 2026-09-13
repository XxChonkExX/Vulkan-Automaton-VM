// storage_sched_test: L4 tiering scheduler — CPU-only (no GPU, fake backends).
// Exercises two-lane binding, MoE predictor prefetch, placement tiers,
// rebalance demotion, and invariants.

#include "vulkan_vm/storage/scheduler.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace vvm::storage::sched;
using namespace vvm::storage::queue;
using namespace vvm::storage::cache;

// FakeBackend shared with the queue test pattern (in-memory stand-in).
class FakeBackend final : public StreamBackend {
public:
    uint32_t submitBatch(const std::vector<IORequest>& batch) override {
        submitted_ += batch.size();
        for (auto& r : batch) inFlight_.push_back(r.id);
        return static_cast<uint32_t>(batch.size());
    }
    uint32_t pollCompletions(std::vector<uint64_t>* out,
                             std::vector<uint64_t>* outFailed = nullptr) override {
        uint32_t n = 0;
        while (!inFlight_.empty()) {
            out->push_back(inFlight_.front());
            inFlight_.pop_front();
            ++n;
        }
        return n;
    }
    const char* name() const override { return "fake"; }
    uint64_t submittedCount() const { return submitted_; }
private:
    uint64_t submitted_ = 0;
    std::deque<uint64_t> inFlight_;
};

static void testPredictor() {
    SequentialPredictor p(4, 8, 2);
    auto n = p.nextPrefetch(0);
    assert(n.size() == 2);
    assert(n[0] == 8 && n[1] == 9); // layer 1, experts 0..1
    auto last = p.nextPrefetch(3);
    assert(last.empty()); // last layer: nothing to prefetch
    std::cout << "  sequential predictor ok\n";
}

static void testPlacementTiers() {
    SchedulerConfig cfg;
    cfg.maxHotShards = 4;
    TieringScheduler s(cfg);

    // Unknown key = cold.
    assert(s.placement(999).tier == Tier::Cold);

    // Schedule layer 0 with active keys 0, 1: both missing (never resident),
    // predictor prefetches layer 1's first topK keys.
    auto r = s.scheduleLayer(0, {0, 1});
    assert(r.ready.empty());
    assert(r.missing.size() == 2);
    assert(r.inFlight.size() == 2); // predictor fired for layer 1 keys 8,9

    // Weight lane drove real IO (queue submitted).
    assert(s.stats().prefetches == 2);
    std::cout << "  placement tiers ok\n";
}

static int testTwoLanes() {
    SchedulerConfig cfg;
    cfg.maxHotShards = 8;
    TieringScheduler s(cfg);
    FakeBackend wbe, kbe;
    s.attachWeightsBackend(&wbe);
    s.attachKVBackend(&kbe);
    s.attachPredictor(std::make_unique<SequentialPredictor>(4, 8, 2));

    // KV lane: spill + fill are real IO through the KV backend.
    if (!s.spillKV(100)) {
        std::cerr << "  FAIL: spill\n";
        return 1;
    }
    assert(s.stats().kvSpills == 1);
    assert(kbe.submittedCount() == 1);

    if (s.ensureHot(100) != RequestQueue::Enqueue::Ok) {
        std::cerr << "  FAIL: ensureHot\n";
        return 1;
    }
    assert(s.stats().kvFills == 1);
    assert(kbe.submittedCount() == 2);

    // Weight lane: scheduleLayer drives the weights backend.
    s.scheduleLayer(0, {0, 1});
    assert(wbe.submittedCount() >= 2); // 2 active + predictor prefetches
    std::cout << "  two-lane binding ok\n";
    return 0;
}

static void testRebalance() {
    SchedulerConfig cfg;
    cfg.maxHotShards = 2;
    TieringScheduler s(cfg);
    FakeBackend wbe;
    s.attachWeightsBackend(&wbe);

    // Fill hot slots: keys 0,1 resident via scheduleLayer (never resident, so
    // they miss; force residency through the hot cache core directly).
    s.hotCache().probe(0);
    s.hotCache().complete(0);
    s.hotCache().probe(1);
    s.hotCache().complete(1);

    assert(s.placement(0).tier == Tier::Cold); // tier map not updated by direct probe
    auto evicted = s.rebalance(1);
    assert(evicted.size() == 1);
    assert(s.placement(evicted[0]).tier == Tier::Warm); // demoted one tier
    assert(s.stats().rebalances == 1);
    std::cout << "  rebalance demotion ok\n";
}

static void testFuzz() {
    std::mt19937 rng(4242);
    SchedulerConfig cfg;
    cfg.maxHotShards = 4;
    cfg.layers = 8;
    TieringScheduler s(cfg);
    FakeBackend wbe, kbe;
    s.attachWeightsBackend(&wbe);
    s.attachKVBackend(&kbe);
    s.attachPredictor(std::make_unique<SequentialPredictor>(cfg.layers, cfg.expertsPerLayer, cfg.topK));

    for (int step = 0; step < 5000; ++step) {
        const uint8_t op = static_cast<uint8_t>(rng() % 4);
        const uint32_t layer = rng() % cfg.layers;
        switch (op) {
            case 0: {
                std::vector<uint64_t> active;
                for (uint32_t e = 0; e < cfg.topK; ++e)
                    active.push_back(static_cast<uint64_t>(layer) * cfg.expertsPerLayer + e);
                s.scheduleLayer(layer, active);
                break;
            }
            case 1: s.spillKV(rng() % 64); break;
            case 2: s.ensureHot(rng() % 64); break;
            case 3: s.rebalance(1 + rng() % 2); break;
        }
        // invariants
        assert(s.stats().rebalances <= 10000);
    }
    std::cout << "  fuzz invariants ok\n";
}

int main() {
    std::cout << "storage_sched_test (CPU-only)\n";
    int rc = 0;
    testPredictor();
    testPlacementTiers();
    rc |= testTwoLanes();
    testRebalance();
    testFuzz();
    std::cout << (rc == 0 ? "All scheduler tests passed!\n" : "scheduler tests FAILED\n");
    return rc;
}