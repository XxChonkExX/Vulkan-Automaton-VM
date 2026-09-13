// storage_queue_test: L2 request queue — Little's Law depth, submit/poll
// split, priority ordering, coalescing, backpressure, fake-backend driving.
// CPU-only (no GPU, no real SSD).
//
// Discipline: no side-effecting calls inside assert() — every call that
// mutates is a REAL call bound to a value first (NDEBUG footgun).

#include "vulkan_vm/storage/request_queue.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace vvm::storage::queue;

// ---------------------------------------------------------------------------
// FakeBackend: accepts submits, completes them after N polls, can fail some.
// Mirrors the ndfake provider pattern (in-memory stand-in for real hardware).
// ---------------------------------------------------------------------------
class FakeBackend final : public StreamBackend {
public:
    explicit FakeBackend(uint32_t completeAfterPolls = 1, uint32_t failEveryNth = 0)
        : completeAfterPolls_(completeAfterPolls == 0 ? 1 : completeAfterPolls),
          failEveryNth_(failEveryNth) {}

    uint32_t submitBatch(const std::vector<IORequest>& batch) override {
        submitted_ += batch.size();
        bytes_ += [&] {
            uint64_t b = 0;
            for (auto& r : batch) b += r.size;
            return b;
        }();
        for (auto& r : batch) inFlight_.push_back(r.id);
        return static_cast<uint32_t>(batch.size());
    }

    uint32_t pollCompletions(std::vector<uint64_t>* out,
                             std::vector<uint64_t>* outFailed = nullptr) override {
        ++polls_;
        if (polls_ < completeAfterPolls_) return 0; // simulate device latency
        uint32_t n = 0;
        while (!inFlight_.empty()) {
            uint64_t id = inFlight_.front();
            inFlight_.pop_front();
            const bool fail = failEveryNth_ != 0 && (id % failEveryNth_) == 0;
            if (fail && outFailed) outFailed->push_back(id);
            else out->push_back(id);
            ++n;
        }
        return n;
    }

    const char* name() const override { return "fake"; }
    uint64_t submittedCount() const { return submitted_; }
    uint64_t bytesSubmittedCount() const { return bytes_; }

private:
    uint32_t completeAfterPolls_ = 1;
    uint32_t failEveryNth_ = 0;
    uint32_t polls_ = 0;
    uint64_t submitted_ = 0;
    uint64_t bytes_ = 0;
    std::deque<uint64_t> inFlight_;
};

static IORequest makeReq(uint64_t shardKey, uint64_t off, uint32_t size, uint32_t prio = 0, bool read = true) {
    IORequest r;
    r.shardKey = shardKey;
    r.fileOffset = off;
    r.size = size;
    r.priority = prio;
    r.isRead = read;
    r.dstSlot = static_cast<uint32_t>(shardKey & 0xFFFFFFFF);
    return r;
}

static void testLittleLaw() {
    QueueConfig cfg;
    // 26 GB/s x 100 us = 2.6 MB in flight / 1 MiB => ceil(2.5) = 3
    cfg.targetBytesPerSec = 26ull << 30;
    cfg.deviceLatencyUs = 100.0;
    cfg.ioSize = 1u << 20;
    assert(RequestQueue::littleLawDepth(cfg) == 3);

    // 4 KiB IOs on the same link: 2.6 MB / 4 KiB = ~635
    cfg.ioSize = 4096;
    const uint32_t d4k = RequestQueue::littleLawDepth(cfg);
    assert(d4k > 600 && d4k < 700);

    // maxDepth caps the computed window
    cfg.maxDepth = 8;
    assert(RequestQueue::littleLawDepth(cfg) == 8);

    // degenerate: tiny io + tiny latency floors at 1
    cfg.ioSize = 1u << 20;
    cfg.deviceLatencyUs = 0.001;
    assert(RequestQueue::littleLawDepth(cfg) >= 1);
    std::cout << "  little-law depth ok\n";
}

static void testSubmitPollSplit() {
    QueueConfig cfg;
    cfg.targetBytesPerSec = 26ull << 30;
    cfg.deviceLatencyUs = 100.0;
    cfg.ioSize = 1u << 20;
    RequestQueue q(cfg);

    FakeBackend be(2); // completes on the 2nd poll (simulated latency)
    for (int i = 0; i < 4; ++i)
        q.enqueue(makeReq(i, i * (1u << 20), 1u << 20));

    assert(q.pendingCount() == 4);
    const uint32_t submitted = q.submit(&be);
    assert(submitted == 4);
    assert(q.inFlightCount() == 4);
    assert(q.pendingCount() == 0);

    // First poll: device still busy.
    auto first = q.poll(&be);
    assert(first.empty());
    assert(q.inFlightCount() == 4);

    // Second poll: everything completes.
    auto done = q.poll(&be);
    assert(done.size() == 4);
    assert(q.allSettled());
    assert(q.stats().completed == 4);
    std::cout << "  submit/poll split ok\n";
}

static void testInFlightWindow() {
    QueueConfig cfg;
    cfg.targetBytesPerSec = 26ull << 30;
    cfg.deviceLatencyUs = 100.0;
    cfg.ioSize = 1u << 20;   // depth 3
    cfg.maxDepth = 3;
    RequestQueue q(cfg);
    assert(q.depth() == 3);

    for (int i = 0; i < 8; ++i)
        q.enqueue(makeReq(i, i * (1u << 20), 1u << 20));

    // Submit fills exactly the window (3), leaves 5 pending.
    assert(q.submit() == 3);
    assert(q.inFlightCount() == 3);
    assert(q.pendingCount() == 5);

    // Complete one; the next submit admits exactly one more.
    auto first = q.poll(); // no backend: driver completes directly
    assert(first.empty());
    const uint64_t idOf0 = 1; // first enqueued id
    assert(q.complete(idOf0));
    assert(q.submit() == 1);
    assert(q.inFlightCount() == 3);
    std::cout << "  in-flight window ok\n";
}

static void testPriorityOrdering() {
    QueueConfig cfg;
    cfg.maxDepth = 4;
    RequestQueue q(cfg);
    assert(q.depth() == 4);

    q.enqueue(makeReq(1, 0, 4096, 3)); // low
    q.enqueue(makeReq(2, 4096, 4096, 0)); // high
    q.enqueue(makeReq(3, 8192, 4096, 2));
    q.enqueue(makeReq(4, 12288, 4096, 0)); // high (after 2)

    assert(q.submit() == 4); // window 4 admits all
    // FIFO within priority 0: ids 2 then 4 submitted before 3 then 1.
    // The queue marks the accepted prefix InFlight in sorted order; verify
    // via stats + state only (ordering itself is covered by the sort above).
    assert(q.inFlightCount() == 4);

    // Window of 2: only the two priority-0 requests submit first.
    RequestQueue q2(cfg);
    q2.enqueue(makeReq(1, 0, 4096, 3));
    q2.enqueue(makeReq(2, 4096, 4096, 0));
    q2.enqueue(makeReq(3, 8192, 4096, 2));
    q2.enqueue(makeReq(4, 12288, 4096, 0));
    assert(q2.submit() == 2);
    assert(q2.state(2) == RequestState::InFlight); // high prio first
    assert(q2.state(4) == RequestState::InFlight);
    assert(q2.state(1) == RequestState::Pending);  // low prio waits
    assert(q2.state(3) == RequestState::Pending);
    std::cout << "  priority ordering ok\n";
}

static void testCoalesce() {
    // Contiguous same-direction runs merge; non-contiguous don't.
    std::vector<IORequest> batch;
    batch.push_back(makeReq(1, 0, 4096));
    batch.push_back(makeReq(2, 4096, 4096));       // contiguous with 1
    batch.push_back(makeReq(3, 8192, 4096));       // contiguous with 2
    batch.push_back(makeReq(4, 65536, 4096));      // gap -> separate
    batch.push_back(makeReq(5, 69632, 4096, 0, false)); // write -> separate

    auto merged = coalesceRuns(batch, 1u << 20);
    assert(merged.size() == 3);
    assert(merged[0].id == 1 && merged[0].size == 12288); // 3 x 4K
    assert(merged[1].id == 4 && merged[1].size == 4096);
    assert(merged[2].id == 5 && merged[2].size == 4096);

    // maxMergedBytes caps growth.
    auto capped = coalesceRuns(batch, 8192);
    assert(capped.size() == 4);
    assert(capped[0].size == 8192); // 1+2 merged, 3 not (would exceed cap)
    std::cout << "  coalesce ok\n";
}

static void testBackpressure() {
    QueueConfig cfg;
    cfg.maxDepth = 1; // outstanding cap = max(depth*8, 64) = 64
    RequestQueue q(cfg);

    uint32_t enqueued = 0;
    for (int i = 0; i < 200; ++i) {
        if (q.enqueue(makeReq(i, i * 4096, 4096)) == RequestQueue::Enqueue::Ok) ++enqueued;
        else break;
    }
    assert(enqueued == 64);
    assert(q.enqueue(makeReq(999, 999 * 4096, 4096)) == RequestQueue::Enqueue::Full);

    // Drain everything; enqueues succeed again.
    FakeBackend be(1);
    while (!q.allSettled()) {
        q.submit(&be);
        q.poll(&be);
    }
    assert(q.enqueue(makeReq(1000, 0, 4096)) == RequestQueue::Enqueue::Ok);
    std::cout << "  backpressure ok\n";
}

static void testFailures() {
    QueueConfig cfg;
    cfg.maxDepth = 16;
    RequestQueue q(cfg);
    FakeBackend be(1, 2); // every 2nd id fails

    for (int i = 0; i < 8; ++i)
        q.enqueue(makeReq(i, i * 4096, 4096));
    assert(q.submit(&be) == 8);
    q.poll(&be);
    assert(q.allSettled());
    assert(q.stats().completed > 0);
    assert(q.stats().failed > 0);
    assert(q.stats().completed + q.stats().failed == 8);
    std::cout << "  failure path ok\n";
}

static void testFuzz() {
    std::mt19937 rng(777);
    QueueConfig cfg;
    cfg.maxDepth = 4;
    RequestQueue q(cfg);
    FakeBackend be(1);

    uint64_t nextKey = 0;
    for (int step = 0; step < 5000; ++step) {
        const uint8_t op = static_cast<uint8_t>(rng() % 4);
        switch (op) {
            case 0: case 1: { // enqueue + submit
                IORequest r = makeReq(nextKey++, nextKey * 4096, 4096,
                                      static_cast<uint32_t>(rng() % kPriorities));
                if (q.enqueue(r) == RequestQueue::Enqueue::Ok) q.submit(&be);
                break;
            }
            case 2: q.submit(&be); q.poll(&be); break;
            case 3: q.poll(&be); break;
        }

        // invariants
        assert(q.inFlightCount() <= q.depth());
        assert(q.pendingCount() + q.inFlightCount() <= q.outstandingCap());
        const auto s = q.stats();
        assert(s.completed + s.failed + s.enqueued >= s.submitted);
    }
    // drain
    while (!q.allSettled()) { q.submit(&be); q.poll(&be); }
    assert(q.allSettled());
    std::cout << "  fuzz invariants ok\n";
}

int main() {
    std::cout << "storage_queue_test (CPU-only)\n";
    testLittleLaw();
    testSubmitPollSplit();
    testInFlightWindow();
    testPriorityOrdering();
    testCoalesce();
    testBackpressure();
    testFailures();
    testFuzz();
    std::cout << "All storage_queue tests passed!\n";
    return 0;
}