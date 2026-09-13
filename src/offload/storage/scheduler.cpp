#include "vulkan_vm/storage/scheduler.hpp"

#include <algorithm>
#include <cstring>

namespace vvm {
namespace storage {
namespace sched {

using queue::IORequest;      // lane request type (vvm::storage::queue)
using queue::RequestQueue;

// ===========================================================================
// Predictors
// ===========================================================================
SequentialPredictor::SequentialPredictor(uint32_t layers, uint32_t expertsPerLayer, uint32_t topK)
    : layers_(layers), expertsPerLayer_(expertsPerLayer), topK_(topK == 0 ? 1 : topK) {}

std::vector<uint64_t> SequentialPredictor::nextPrefetch(uint32_t layerDone) {
    std::vector<uint64_t> out;
    const uint32_t next = layerDone + 1;
    if (next >= layers_) return out; // last layer: nothing to prefetch
    out.reserve(topK_);
    for (uint32_t e = 0; e < topK_ && e < expertsPerLayer_; ++e)
        out.push_back(static_cast<uint64_t>(next) * expertsPerLayer_ + e);
    return out;
}

// ===========================================================================
// TieringScheduler
// ===========================================================================
TieringScheduler::TieringScheduler(SchedulerConfig cfg)
    : cfg_(cfg),
      hot_(cache::CacheConfig{cfg.maxHotShards, 20, cfg.policy}),
      weightQueue_(queue::QueueConfig{}),
      kvQueue_(queue::QueueConfig{}) {}

void TieringScheduler::attachWeightsBackend(queue::StreamBackend* be) { weightsBe_ = be; }
void TieringScheduler::attachKVBackend(queue::StreamBackend* be) { kvBe_ = be; }
void TieringScheduler::attachPredictor(std::unique_ptr<Predictor> p) { predictor_ = std::move(p); }

Placement TieringScheduler::placement(uint64_t key) const {
    auto it = tier_.find(key);
    if (it != tier_.end()) return {it->second, key};
    return {Tier::Cold, key};
}

TieringScheduler::ScheduleResult TieringScheduler::scheduleLayer(
    uint32_t layer, const std::vector<uint64_t>& activeKeys) {
    ScheduleResult out;

    // 1) Resident check: hot shards are ready; refresh their clock bit.
    for (uint64_t key : activeKeys) {
        if (hot_.state(static_cast<uint32_t>(key % cfg_.maxHotShards)) == cache::LineState::Valid) {
            // Residency is by slot; verify the key maps to this slot.
            const uint32_t slot = static_cast<uint32_t>(key % cfg_.maxHotShards);
            if (hot_.keyAt(slot) == key) {
                hot_.pin(slot);          // hold while the shader runs (R5: no evict under use)
                out.ready.push_back(key);
                tier_[key] = Tier::Hot;
                ++stats_.hotHits;
                continue;
            }
        }
        out.missing.push_back(key);
    }

    // 2) Prefetch: predictor-adjacent keys (weight lane), skipping residents.
    if (predictor_) {
        for (uint64_t key : predictor_->nextPrefetch(layer)) {
            if (std::find(out.ready.begin(), out.ready.end(), key) != out.ready.end()) continue;
            const uint32_t slot = static_cast<uint32_t>(key % cfg_.maxHotShards);
            if (hot_.keyAt(slot) == key &&
                hot_.state(slot) == cache::LineState::Valid) continue;

            auto p = hot_.probe(key);
            if (p.kind == cache::ShardCache::Lookup::Miss) {
                // weight-lane IO: bulk read into the slot's backend staging
                IORequest req;
                req.id = 0; // RequestQueue assigns
                req.shardKey = key;
                req.dstSlot = p.slot;
                req.isRead = true;
                req.priority = 1;
                if (weightQueue_.enqueue(req) == queue::RequestQueue::Enqueue::Ok) {
                    weightQueue_.submit(weightsBe_);
                    out.inFlight.push_back(key);
                    tier_[key] = Tier::Hot; // will be Valid on completion
                    ++stats_.prefetches;
                } else {
                    hot_.evict(p.slot); // give the slot back; cold for now
                }
            }
        }
    }

    // 3) Reap weight-lane completions -> cache lines Valid + unpinned.
    weightQueue_.poll(weightsBe_);
    // Completed shards: mark their lines Valid (the backend filled the slot).
    for (uint64_t key : out.inFlight) {
        const uint32_t slot = static_cast<uint32_t>(key % cfg_.maxHotShards);
        if (hot_.state(slot) == cache::LineState::InFlight && hot_.keyAt(slot) == key) {
            hot_.complete(slot);
            hot_.pin(slot);
        }
    }

    return out;
}

Result TieringScheduler::spillKV(uint64_t key, uint32_t priority) {
    IORequest req;
    req.id = 0;
    req.shardKey = key;
    req.size = cfg_.kvBlockBytes;
    req.isRead = false;
    req.priority = priority;
    if (kvQueue_.enqueue(req) != queue::RequestQueue::Enqueue::Ok)
        return Result::error(ErrorCode::OutOfMemory, "KV spill queue full");
    kvQueue_.submit(kvBe_);
    tier_[key] = Tier::Warm; // written to the shadow; cold on flush
    ++stats_.kvSpills;
    return Result::success();
}

queue::RequestQueue::Enqueue TieringScheduler::ensureHot(uint64_t key, uint32_t priority) {
    IORequest req;
    req.id = 0;
    req.shardKey = key;
    req.size = cfg_.kvBlockBytes;
    req.isRead = true;
    req.priority = priority;
    auto r = kvQueue_.enqueue(req);
    if (r == queue::RequestQueue::Enqueue::Ok) {
        kvQueue_.submit(kvBe_);
        ++stats_.kvFills;
    }
    return r;
}

std::vector<uint64_t> TieringScheduler::rebalance(uint32_t count) {
    std::vector<uint64_t> evicted;
    // Scan hot slots with the clock hand; demote unpinned Valid lines.
    const uint32_t cap = cfg_.maxHotShards;
    for (uint32_t i = 0; i < cap && evicted.size() < count; ++i) {
        const uint32_t slot = (clock_ % cap);
        ++clock_;
        if (hot_.refcount(slot) != 0) continue; // pinned: in use (R5)
        if (hot_.state(slot) != cache::LineState::Valid) continue;
        const uint64_t key = hot_.keyAt(slot);
        auto e = hot_.evict(slot);
        if (e) {
            // Demote one tier: hot -> warm (host shadow); cold stays cold.
            tier_[key] = Tier::Warm;
            evicted.push_back(key);
            ++stats_.rebalances;
        }
    }
    return evicted;
}

TieringScheduler::Stats TieringScheduler::stats() const { return stats_; }

} // namespace sched
} // namespace storage
} // namespace vvm