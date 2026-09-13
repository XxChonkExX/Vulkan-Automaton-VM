#pragma once

// L1 — software cache core (BaM §3.4, ASPLOS'23), platform-neutral.
//
// A fixed-capacity set of cache "lines" (one shard each), with:
//   * per-line state machine: Invalid -> InFlight -> Valid
//   * atomic reference counting (pinning) so consumers hold lines without a
//     global lock, and eviction skips pinned lines (no use-after-free)
//   * a swappable eviction policy (Clock / LRU / FIFO) behind one interface —
//     the same "best tool for the job" seam the network layer uses for
//     TCP / verbs / NDK / UCX.
//
// The cache is storage-backend-agnostic: it tracks *residency*, never bytes.
// A backend fills an InFlight line from SSD and calls complete(); the shard
// bytes live in a backend-owned staging/pool slot keyed by the line slot.

#include <atomic>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#ifndef VVM_API
#if defined(VVM_BUILD_SHARED) && defined(VVM_EXPORT)
#  if defined(_MSC_VER)
#    define VVM_API __declspec(dllexport)
#  else
#    define VVM_API __attribute__((visibility("default")))
#  endif
#elif defined(VVM_BUILD_SHARED) && defined(_MSC_VER)
#  define VVM_API __declspec(dllimport)
#else
#  define VVM_API
#endif
#endif

namespace vvm {
namespace storage {
namespace cache {

enum class LineState : uint8_t { Invalid = 0, InFlight = 1, Valid = 2 };

enum class EvictionKind : uint32_t { Clock = 0, Lru = 1, Fifo = 2 };

struct CacheConfig {
    uint32_t capacity = 0;        // number of resident shard slots
    uint32_t shardLog2 = 20;      // nominal line size (informational)
    EvictionKind policy = EvictionKind::Clock;
};

// ---------------------------------------------------------------------------
// Eviction policy — one strategy interface, swappable (mirrors RdmaTransport).
// `victim()` is handed a snapshot of per-slot refcounts (index == slot) and
// MUST skip any slot with refcount != 0. Returns nullopt only if every slot is
// pinned.
// ---------------------------------------------------------------------------
class VVM_API EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;
    virtual void reset(uint32_t capacity) = 0;
    virtual void onInsert(uint32_t slot) = 0;
    virtual void onAccess(uint32_t slot) = 0;
    virtual void onEvict(uint32_t slot) = 0;
    virtual std::optional<uint32_t> victim(const std::vector<uint32_t>& refcounts) = 0;
};

// BaM clock (Corbato '68): global slot hand + per-line reference bit. Parallel
// victim selection, O(1) typical. Default policy.
class VVM_API ClockPolicy final : public EvictionPolicy {
public:
    void reset(uint32_t capacity) override;
    void onInsert(uint32_t slot) override;
    void onAccess(uint32_t slot) override;
    void onEvict(uint32_t slot) override;
    std::optional<uint32_t> victim(const std::vector<uint32_t>& refcounts) override;
private:
    std::vector<uint8_t> referenced_;
    uint32_t hand_ = 0;
    uint32_t capacity_ = 0;
};

class VVM_API LruPolicy final : public EvictionPolicy {
public:
    void reset(uint32_t capacity) override;
    void onInsert(uint32_t slot) override;
    void onAccess(uint32_t slot) override;
    void onEvict(uint32_t slot) override;
    std::optional<uint32_t> victim(const std::vector<uint32_t>& refcounts) override;
private:
    std::vector<std::list<uint32_t>::iterator> iter_; // slot -> list pos (end() == absent)
    std::list<uint32_t> order_;                       // front = most-recent
};

class VVM_API FifoPolicy final : public EvictionPolicy {
public:
    void reset(uint32_t capacity) override;
    void onInsert(uint32_t slot) override;
    void onAccess(uint32_t slot) override;
    void onEvict(uint32_t slot) override;
    std::optional<uint32_t> victim(const std::vector<uint32_t>& refcounts) override;
private:
    std::deque<uint32_t> order_;
};

VVM_API std::unique_ptr<EvictionPolicy> makePolicy(EvictionKind kind);

// ---------------------------------------------------------------------------
// ShardCache — the residency table.
// ---------------------------------------------------------------------------
class VVM_API ShardCache {
public:
    explicit ShardCache(CacheConfig cfg);

    ShardCache(const ShardCache&) = delete;
    ShardCache& operator=(const ShardCache&) = delete;

    enum class Lookup : uint8_t { Hit, Miss, InFlight, Full };

    struct Probe {
        Lookup kind = Lookup::Miss;
        uint32_t slot = 0;
        // Valid when kind == Miss and a resident line was evicted to make room.
        uint64_t evictedKey = UINT64_MAX;
    };

    // Resolve a shard key:
    //   Hit      -> resident, policy onAccess fired, caller pins slot
    //   InFlight -> another thread is fetching; caller coalesces/waits
    //   Miss     -> line reserved (InFlight), caller fetches then complete()
    //   Full     -> capacity exhausted and every line pinned
    Probe probe(uint64_t key);

    // InFlight -> Valid. Call after the backend finished filling the line.
    void complete(uint32_t slot);

    // Atomic pin/unpin. Pinning blocks eviction (BaM refcount semantic).
    void pin(uint32_t slot);
    void unpin(uint32_t slot);
    uint32_t refcount(uint32_t slot) const;

    bool isDirty(uint32_t slot) const;
    void markDirty(uint32_t slot);
    void markClean(uint32_t slot);

    // Force a line back to Invalid; returns its key if it was resident.
    std::optional<uint64_t> evict(uint32_t slot);

    uint32_t capacity() const { return cfg_.capacity; }
    uint32_t residentCount() const;
    LineState state(uint32_t slot) const;
    uint64_t keyAt(uint32_t slot) const;

private:
    struct Line {
        uint64_t key = UINT64_MAX;
        std::atomic<uint32_t> refcount{0};
        std::atomic<LineState> state{LineState::Invalid};
        std::atomic<bool> dirty{false};
        bool referenced = false; // clock reference bit; guarded by mutex_
    };

    CacheConfig cfg_;
    mutable std::mutex mutex_;               // structural changes + `referenced`
    std::vector<Line> lines_;
    std::unordered_map<uint64_t, uint32_t> index_; // key -> slot
    std::unique_ptr<EvictionPolicy> policy_;
};

} // namespace cache
} // namespace storage
} // namespace vvm