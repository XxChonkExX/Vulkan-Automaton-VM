#pragma once

// L2 — request queue: Little's Law depth sizing + submit/poll split.
//
// The queue is a state machine over IORequest records (Pending -> InFlight ->
// Done/Failed). It never touches bytes: the backend (L3) does actual I/O and
// reports completions; the queue orchestrates ordering, backpressure, and
// accounting. This keeps it CPU-testable with a fake backend on any host.
//
// Rules taken from the frontier systems (docs/STORAGE_STREAM_ARCHITECTURE.md):
//   * Little's Law (BaM §2.2): in-flight depth needed to saturate a link is
//     depth = throughput x latency / ioSize. 12 GB/s x 100 us / 1 MiB = ~2
//     concurrent 1 MiB IOs; the same link at 4 KiB IOs needs ~293. The queue
//     computes and enforces this window.
//   * Submit/poll split (uGDS, DirectStorage EnqueueRequests): enqueue is
//     decoupled from submit, submit from poll. One flush submits a batch.
//   * Priority weighting (DirectStorage): 0 = highest .. 3 = lowest; submit
//     drains highest priority first, FIFO within a priority.
//   * Coalescing (uGDS/DirectStorage submission threads): consecutive
//     contiguous IOs merge into larger device commands at the backend
//     boundary; `coalesceRuns()` is the pure helper for that.

#include <cstdint>
#include <deque>
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
namespace queue {

// ---------------------------------------------------------------------------
// Request model (64-bit ids/offsets — best fit; 32-bit sizes — IOs < 4 GiB)
// ---------------------------------------------------------------------------

enum class RequestState : uint8_t { Pending = 0, InFlight = 1, Done = 2, Failed = 3 };

// Priority buckets (DirectStorage-style). 0 = highest.
constexpr uint32_t kPriorities = 4;

struct IORequest {
    uint64_t id = 0;         // monotonic ticket, assigned by the queue
    uint64_t shardKey = 0;   // L1 cache line key this IO fills (opaque here)
    uint64_t fileOffset = 0; // absolute offset in the pack
    uint32_t size = 0;       // bytes (4 KiB-aligned)
    uint32_t dstSlot = 0;    // destination staging/pool slot (opaque to queue)
    bool isRead = true;      // reads stream in; writes spill (KV path)
    uint32_t priority = 0;   // 0 highest .. 3 lowest
};

struct QueueConfig {
    // Little's Law inputs.
    uint64_t targetBytesPerSec = 26ull * 1024 * 1024 * 1024; // PCIe Gen4 x16 ~26 GB/s
    double deviceLatencyUs = 100.0;                          // NVMe read latency
    uint32_t ioSize = 1u << 20;                              // 1 MiB IO granularity
    uint32_t maxDepth = 0;                                   // 0 = auto (Little's Law), else cap

    // Batch behavior.
    uint32_t batchThreshold = 64;   // auto-submit hint when >= N pending
    uint32_t maxBatchBytes = 32u << 20; // cap for one submit batch

    bool validate() const {
        return ioSize != 0 && targetBytesPerSec != 0 && deviceLatencyUs > 0.0
            && batchThreshold != 0 && maxBatchBytes != 0;
    }
};

// ---------------------------------------------------------------------------
// Backend seam (L3 implements this; mirrors RdmaTransport/EvictionPolicy swap)
// ---------------------------------------------------------------------------

class StreamBackend {
public:
    virtual ~StreamBackend() = default;
    // Hand a batch to the device. Returns the number accepted (<= batch size).
    virtual uint32_t submitBatch(const std::vector<IORequest>& batch) = 0;
    // Non-blocking completion reap: append finished request ids to outIds and
    // failed request ids to outFailed (may be null). Returns the number
    // appended across both.
    virtual uint32_t pollCompletions(std::vector<uint64_t>* outIds,
                                     std::vector<uint64_t>* outFailed = nullptr) = 0;
    virtual const char* name() const = 0;
};

// Pure helper: merge contiguous same-direction runs into larger device
// commands (fewer doorbells). Input batch must be sorted by (fileOffset,
// isRead). Returns the merged batch; merged entry keeps the FIRST id and its
// dstSlot, `size` grows. Ids of absorbed entries are appended to `absorbed`.
VVM_API std::vector<IORequest> coalesceRuns(const std::vector<IORequest>& batch,
                                            uint32_t maxMergedBytes,
                                            std::vector<uint64_t>* absorbed = nullptr);

// ---------------------------------------------------------------------------
// RequestQueue
// ---------------------------------------------------------------------------

class VVM_API RequestQueue {
public:
    explicit RequestQueue(QueueConfig cfg);

    // Little's Law: depth = ceil(throughput x latency / ioSize), capped.
    static uint32_t littleLawDepth(const QueueConfig& cfg);
    uint32_t depth() const { return depth_; }

    enum class Enqueue : uint8_t { Ok, Full, Invalid };

    // Add a request. Fails Full when outstanding (pending + in-flight) has
    // reached the backpressure cap (depth x kBackpressureMultiplier).
    Enqueue enqueue(const IORequest& req);

    // Flush: pending -> InFlight, highest priority first, FIFO within a
    // priority, up to the in-flight window. Hands accepted entries to the
    // backend when one is attached. Returns the number submitted.
    uint32_t submit(StreamBackend* backend = nullptr);

    // Reap completions. With a backend, polls it; otherwise the driver calls
    // complete() directly. Returns the ids that finished this call.
    std::vector<uint64_t> poll(StreamBackend* backend = nullptr);

    // Mark an in-flight request finished (queue-only driving, tests).
    bool complete(uint64_t id, bool failed = false);

    RequestState state(uint64_t id) const;
    bool allSettled() const; // no pending, no in-flight

    uint32_t pendingCount() const;
    uint32_t inFlightCount() const;
    uint64_t outstandingCap() const { return outstandingCap_; }

    struct Stats {
        uint64_t enqueued = 0;
        uint64_t submitted = 0;
        uint64_t completed = 0;
        uint64_t failed = 0;
        uint64_t bytesSubmitted = 0;
    };
    Stats stats() const;

private:
    QueueConfig cfg_;
    uint32_t depth_ = 1;
    uint64_t outstandingCap_ = 0;
    uint64_t nextId_ = 1;

    struct Record {
        IORequest req;
        RequestState state = RequestState::Pending;
    };
    std::deque<Record> records_;                          // insertion order
    std::unordered_map<uint64_t, size_t> index_;          // id -> records_ pos

    uint32_t statsInFlightLocked() const;

    // Structurally: enqueue/submit/poll/complete mutate records_ + index_.
    // Guarded by one mutex; contended only across loader/poller threads.
    mutable std::mutex mutex_;
    Stats stats_;
};

} // namespace queue
} // namespace storage
} // namespace vvm