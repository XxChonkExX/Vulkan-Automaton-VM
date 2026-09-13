#pragma once

// L4 — tiering scheduler: two lanes bound to backends (blk-mq HCTX split).
//
//   Weight lane  — bulk sequential shard streams (MoE experts), bound to the
//                  zero-copy backend when available, predictor-prefetched.
//   KV lane      — fine-grained spill/fill (attention cache), bound to the
//                  read+write backend, clock-evicted, SSD as third tier below
//                  the pool's offloadToHost (DMA tier).
//
// Placement is FlexGen-style: hot shards resident in VRAM (L1 cache core),
// warm in the host shadow (offload tier), cold on SSD (this module's packs).
// Placement is solved once at load and re-evaluated on pressure.

#include "vulkan_vm/storage/request_queue.hpp"
#include "vulkan_vm/storage/cache.hpp"
#include "vulkan_vm/core.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace vvm {
namespace storage {
namespace sched {

enum class Lane : uint8_t { Weights = 0, KV = 1 };
enum class Tier : uint8_t { Cold = 0, Warm = 1, Hot = 2 }; // SSD / host shadow / VRAM

struct Placement {
    Tier tier = Tier::Cold;
    uint64_t key = 0;
};

// Predicts the next layer's expert set for MoE prefetch. Pluggable: the
// default is a layer-sequential predictor (next layer's top-k set); a router
// logits predictor can be injected for real models.
class VVM_API Predictor {
public:
    virtual ~Predictor() = default;
    // Given the layer just scheduled, return the shard keys to prefetch next.
    virtual std::vector<uint64_t> nextPrefetch(uint32_t layerDone) = 0;
};

// Sequential layer predictor: prefetches layer+1's expert set. `layers` x
// `expertsPerLayer` key space, key = layer * expertsPerLayer + expert.
class VVM_API SequentialPredictor final : public Predictor {
public:
    SequentialPredictor(uint32_t layers, uint32_t expertsPerLayer, uint32_t topK);
    std::vector<uint64_t> nextPrefetch(uint32_t layerDone) override;
private:
    uint32_t layers_ = 0;
    uint32_t expertsPerLayer_ = 0;
    uint32_t topK_ = 1;
};

struct SchedulerConfig {
    uint32_t layers = 32;
    uint32_t expertsPerLayer = 8;
    uint32_t topK = 2;               // active experts per token
    uint32_t maxHotShards = 16;      // VRAM residency cap (L1 cache capacity)
    uint32_t prefetchDepth = 4;      // shards to prefetch per layer
    uint32_t kvBlockBytes = 1u << 20; // KV block IO size (queue rejects zero)
    cache::EvictionKind policy = cache::EvictionKind::Clock;
};

// ---------------------------------------------------------------------------
// TieringScheduler
// ---------------------------------------------------------------------------
class VVM_API TieringScheduler {
public:
    explicit TieringScheduler(SchedulerConfig cfg);

    // Backends: the weight lane gets the zero-copy backend when one exists
    // (DStorage bridge), the KV lane the read+write backend (IoRing floor).
    // Either may be null (queue-only driving for tests).
    void attachWeightsBackend(queue::StreamBackend* be);
    void attachKVBackend(queue::StreamBackend* be);
    void attachPredictor(std::unique_ptr<Predictor> p);

    // ---- placement query ----
    // Where a shard lives right now. Hot = L1 cache has it resident.
    Placement placement(uint64_t key) const;

    // ---- weight lane ----
    // Schedule the next layer: return keys resident (Hit), start prefetch for
    // predictor-adjacent keys, and return the set the caller must wait on.
    // Keys not resident and not prefetchable move to the caller's miss list.
    struct ScheduleResult {
        std::vector<uint64_t> ready;      // resident in VRAM (Hit)
        std::vector<uint64_t> inFlight;   // prefetch submitted this call
        std::vector<uint64_t> missing;    // neither resident nor prefetchable
    };
    ScheduleResult scheduleLayer(uint32_t layer, const std::vector<uint64_t>& activeKeys);

    // ---- KV lane ----
    // Spill a KV block to the warm tier (host shadow) or cold tier (SSD pack).
    // The KV lane is the only writer; reads come back through ensureHot.
    Result spillKV(uint64_t key, uint32_t priority = 3);
    // Bring a KV block hot (blocking on the caller's part via the queue).
    queue::RequestQueue::Enqueue ensureHot(uint64_t key, uint32_t priority = 1);

    // ---- re-evaluation ----
    // On VRAM pressure: demote cold-est hot shards (LRU-est) one tier down.
    // Returns evicted keys (they now live warm or cold).
    std::vector<uint64_t> rebalance(uint32_t count);

    // ---- accounting ----
    struct Stats {
        uint64_t hotHits = 0;         // scheduleLayer found resident
        uint64_t prefetches = 0;      // weight-lane prefetches submitted
        uint64_t kvSpills = 0;        // KV-lane spills submitted
        uint64_t kvFills = 0;         // KV-lane fills submitted
        uint64_t rebalances = 0;      // shards demoted
    };
    Stats stats() const;

    // L1 cache core exposure (residency state machine, pin/unpin).
    cache::ShardCache& hotCache() { return hot_; }

private:
    SchedulerConfig cfg_;
    cache::ShardCache hot_;
    queue::RequestQueue weightQueue_;
    queue::RequestQueue kvQueue_;
    queue::StreamBackend* weightsBe_ = nullptr; // not owned
    queue::StreamBackend* kvBe_ = nullptr;      // not owned
    std::unique_ptr<Predictor> predictor_;
    std::unordered_map<uint64_t, Tier> tier_;   // key -> where it lives
    mutable uint32_t clock_ = 0;                // rebalance scan hand
    Stats stats_;
};

} // namespace sched
} // namespace storage
} // namespace vvm