#include "vulkan_vm/storage/request_queue.hpp"

#include <algorithm>

namespace vvm {
namespace storage {
namespace queue {

// ===========================================================================
// coalesceRuns (pure helper)
// ===========================================================================
std::vector<IORequest> coalesceRuns(const std::vector<IORequest>& batch,
                                    uint32_t maxMergedBytes,
                                    std::vector<uint64_t>* absorbed) {
    std::vector<IORequest> out;
    out.reserve(batch.size());
    if (absorbed) absorbed->clear();

    for (const auto& r : batch) {
        // Merge into the previous entry when: contiguous in file space, same
        // direction, same destination slot, same priority, and the merged
        // size stays within the device command cap.
        if (!out.empty()) {
            IORequest& prev = out.back();
            const bool contiguous = prev.isRead == r.isRead
                                 && prev.dstSlot == r.dstSlot
                                 && prev.priority == r.priority
                                 && prev.fileOffset + prev.size == r.fileOffset
                                 && static_cast<uint64_t>(prev.size) + r.size <= maxMergedBytes;
            if (contiguous) {
                prev.size += r.size;
                if (absorbed) absorbed->push_back(r.id);
                continue;
            }
        }
        out.push_back(r);
    }
    return out;
}

// ===========================================================================
// RequestQueue
// ===========================================================================
uint32_t RequestQueue::littleLawDepth(const QueueConfig& cfg) {
    if (!cfg.validate()) return 1;
    // depth = throughput x latency / ioSize  (BaM §2.2)
    const double bytesInFlight =
        static_cast<double>(cfg.targetBytesPerSec) * cfg.deviceLatencyUs * 1e-6;
    double d = bytesInFlight / static_cast<double>(cfg.ioSize);
    if (d < 1.0) d = 1.0;
    uint32_t depth = static_cast<uint32_t>(d + 0.999); // ceil
    if (cfg.maxDepth != 0 && depth > cfg.maxDepth) depth = cfg.maxDepth;
    return depth;
}

RequestQueue::RequestQueue(QueueConfig cfg)
    : cfg_(cfg), depth_(littleLawDepth(cfg)) {
    // Outstanding cap: the in-flight window is the hardware constraint
    // (depth); pending CPU-side staging is cheap, so allow a generous
    // multiple of the window before backpressure engages.
    outstandingCap_ = static_cast<uint64_t>(depth_) * 8;
    if (outstandingCap_ < 64) outstandingCap_ = 64;
}

RequestQueue::Enqueue RequestQueue::enqueue(const IORequest& req) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (req.size == 0) return Enqueue::Invalid;

    uint64_t outstanding = 0;
    for (const auto& rec : records_)
        if (rec.state == RequestState::Pending || rec.state == RequestState::InFlight)
            ++outstanding;
    if (outstanding >= outstandingCap_) return Enqueue::Full;

    Record rec;
    rec.req = req;
    // Correlation id: preserve the caller's when provided (id != 0) — the CQE
    // UserData must map back to the caller's table. Only assign when unset.
    if (rec.req.id == 0) rec.req.id = nextId_++;
    rec.state = RequestState::Pending;
    index_[rec.req.id] = records_.size();
    records_.push_back(std::move(rec));
    ++stats_.enqueued;
    return Enqueue::Ok;
}

uint32_t RequestQueue::submit(StreamBackend* backend) {
    std::lock_guard<std::mutex> lk(mutex_);

    // Gather pending, priority-ordered (stable: FIFO within a priority).
    std::vector<size_t> order;
    for (size_t i = 0; i < records_.size(); ++i)
        if (records_[i].state == RequestState::Pending) order.push_back(i);
    if (order.empty()) return 0;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return records_[a].req.priority < records_[b].req.priority;
    });

    // Cap the batch to the free in-flight window and byte budget.
    uint32_t freeWindow = depth_ >= statsInFlightLocked() ? depth_ - statsInFlightLocked() : 0;
    if (freeWindow == 0) return 0;

    std::vector<IORequest> batch;
    uint64_t batchBytes = 0;
    uint32_t taken = 0;
    for (size_t idx : order) {
        if (taken >= freeWindow) break;
        const IORequest& r = records_[idx].req;
        if (batchBytes + r.size > cfg_.maxBatchBytes && !batch.empty()) break;
        batch.push_back(r);
        batchBytes += r.size;
        ++taken;
    }
    if (batch.empty()) return 0;

    uint32_t accepted = taken;
    if (backend) {
        accepted = backend->submitBatch(batch);
        if (accepted > taken) accepted = taken;
    }

    // Mark the accepted prefix InFlight (batch order == order vector prefix).
    for (uint32_t i = 0; i < accepted; ++i) {
        records_[order[i]].state = RequestState::InFlight;
        ++stats_.submitted;
        stats_.bytesSubmitted += records_[order[i]].req.size;
    }
    return accepted;
}

std::vector<uint64_t> RequestQueue::poll(StreamBackend* backend) {
    std::vector<uint64_t> done;
    if (backend) {
        std::vector<uint64_t> ids, failed;
        backend->pollCompletions(&ids, &failed);
        for (uint64_t id : ids) {
            if (complete(id)) done.push_back(id);
        }
        for (uint64_t id : failed) complete(id, true);
    }
    return done;
}

bool RequestQueue::complete(uint64_t id, bool failed) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = index_.find(id);
    if (it == index_.end()) return false;
    Record& rec = records_[it->second];
    if (rec.state != RequestState::InFlight) return false;
    rec.state = failed ? RequestState::Failed : RequestState::Done;
    if (failed) ++stats_.failed; else ++stats_.completed;
    return true;
}

RequestState RequestQueue::state(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = index_.find(id);
    if (it == index_.end()) return RequestState::Pending; // unknown -> not ours
    return records_[it->second].state;
}

bool RequestQueue::allSettled() const {
    std::lock_guard<std::mutex> lk(mutex_);
    for (const auto& rec : records_)
        if (rec.state == RequestState::Pending || rec.state == RequestState::InFlight)
            return false;
    return true;
}

uint32_t RequestQueue::pendingCount() const {
    std::lock_guard<std::mutex> lk(mutex_);
    uint32_t n = 0;
    for (const auto& rec : records_) if (rec.state == RequestState::Pending) ++n;
    return n;
}

uint32_t RequestQueue::inFlightCount() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return statsInFlightLocked();
}

uint32_t RequestQueue::statsInFlightLocked() const {
    uint32_t n = 0;
    for (const auto& rec : records_) if (rec.state == RequestState::InFlight) ++n;
    return n;
}

RequestQueue::Stats RequestQueue::stats() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return stats_;
}

} // namespace queue
} // namespace storage
} // namespace vvm