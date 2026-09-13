#include "vulkan_vm/storage/cache.hpp"

namespace vvm {
namespace storage {
namespace cache {

// ===========================================================================
// Clock policy
// ===========================================================================
void ClockPolicy::reset(uint32_t capacity) {
    capacity_ = capacity;
    referenced_.assign(capacity, 0);
    hand_ = 0;
}
void ClockPolicy::onInsert(uint32_t slot) { referenced_[slot] = 0; }
void ClockPolicy::onAccess(uint32_t slot) { referenced_[slot] = 1; }
void ClockPolicy::onEvict(uint32_t slot)  { referenced_[slot] = 0; }

std::optional<uint32_t> ClockPolicy::victim(const std::vector<uint32_t>& refcounts) {
    for (uint32_t i = 0; i < capacity_; ++i) {
        uint32_t cand = (hand_ % capacity_);
        ++hand_;
        if (refcounts[cand] != 0) continue;   // pinned — never evict
        if (referenced_[cand]) {              // second chance
            referenced_[cand] = 0;
            continue;
        }
        return cand;
    }
    return std::nullopt; // all referenced or pinned
}

// ===========================================================================
// LRU policy
// ===========================================================================
void LruPolicy::reset(uint32_t capacity) {
    order_.clear();
    iter_.assign(capacity, order_.end());
}
void LruPolicy::onInsert(uint32_t slot) {
    order_.push_front(slot);
    iter_[slot] = order_.begin();
}
void LruPolicy::onAccess(uint32_t slot) {
    if (iter_[slot] == order_.end()) return;
    order_.erase(iter_[slot]);
    order_.push_front(slot);
    iter_[slot] = order_.begin();
}
void LruPolicy::onEvict(uint32_t slot) {
    if (iter_[slot] != order_.end()) {
        order_.erase(iter_[slot]);
        iter_[slot] = order_.end();
    }
}
std::optional<uint32_t> LruPolicy::victim(const std::vector<uint32_t>& refcounts) {
    for (auto it = order_.rbegin(); it != order_.rend(); ++it) {
        if (refcounts[*it] == 0) return *it;
    }
    return std::nullopt;
}

// ===========================================================================
// FIFO policy
// ===========================================================================
void FifoPolicy::reset(uint32_t /*capacity*/) { order_.clear(); }
void FifoPolicy::onInsert(uint32_t slot) { order_.push_back(slot); }
void FifoPolicy::onAccess(uint32_t /*slot*/) {}
void FifoPolicy::onEvict(uint32_t /*slot*/) {}
std::optional<uint32_t> FifoPolicy::victim(const std::vector<uint32_t>& refcounts) {
    // Rotate past pinned slots; the front age-order is preserved.
    for (size_t i = 0; i < order_.size(); ++i) {
        uint32_t f = order_.front();
        if (refcounts[f] == 0) return f;
        order_.pop_front();
        order_.push_back(f);
    }
    return std::nullopt;
}

std::unique_ptr<EvictionPolicy> makePolicy(EvictionKind kind) {
    switch (kind) {
        case EvictionKind::Clock: return std::make_unique<ClockPolicy>();
        case EvictionKind::Lru:   return std::make_unique<LruPolicy>();
        case EvictionKind::Fifo:  return std::make_unique<FifoPolicy>();
    }
    return std::make_unique<ClockPolicy>();
}

// ===========================================================================
// ShardCache
// ===========================================================================
ShardCache::ShardCache(CacheConfig cfg)
    : cfg_(cfg), lines_(cfg.capacity), policy_(makePolicy(cfg.policy)) {
    policy_->reset(cfg.capacity);
}

ShardCache::Probe ShardCache::probe(uint64_t key) {
    std::lock_guard<std::mutex> lk(mutex_);

    auto it = index_.find(key);
    if (it != index_.end()) {
        const uint32_t s = it->second;
        const LineState st = lines_[s].state.load(std::memory_order_acquire);
        if (st == LineState::Valid) {
            policy_->onAccess(s);
            return {Lookup::Hit, s, UINT64_MAX};
        }
        return {Lookup::InFlight, s, UINT64_MAX};
    }

    // Miss: claim a slot (free Invalid first, else policy victim).
    uint32_t slot = UINT32_MAX;
    for (uint32_t s = 0; s < cfg_.capacity; ++s) {
        if (lines_[s].state.load(std::memory_order_acquire) == LineState::Invalid &&
            lines_[s].refcount.load(std::memory_order_acquire) == 0) {
            slot = s;
            break;
        }
    }
    uint64_t evictedKey = UINT64_MAX;
    if (slot == UINT32_MAX) {
        std::vector<uint32_t> refs(cfg_.capacity);
        for (uint32_t s = 0; s < cfg_.capacity; ++s)
            refs[s] = lines_[s].refcount.load(std::memory_order_acquire);
        auto v = policy_->victim(refs);
        if (!v) return {Lookup::Full, 0, UINT64_MAX};
        slot = *v;
        evictedKey = lines_[slot].key;
        if (index_.count(evictedKey)) index_.erase(evictedKey);
        policy_->onEvict(slot);
    }

    // Reclaim the slot for `key` as InFlight.
    lines_[slot].key = key;
    lines_[slot].refcount.store(0, std::memory_order_release);
    lines_[slot].dirty.store(false, std::memory_order_release);
    lines_[slot].referenced = false;
    lines_[slot].state.store(LineState::InFlight, std::memory_order_release);
    index_[key] = slot;
    policy_->onInsert(slot);

    return {Lookup::Miss, slot, evictedKey};
}

void ShardCache::complete(uint32_t slot) {
    lines_[slot].state.store(LineState::Valid, std::memory_order_release);
}

void ShardCache::pin(uint32_t slot)   { lines_[slot].refcount.fetch_add(1, std::memory_order_acq_rel); }
void ShardCache::unpin(uint32_t slot) { lines_[slot].refcount.fetch_sub(1, std::memory_order_acq_rel); }
uint32_t ShardCache::refcount(uint32_t slot) const { return lines_[slot].refcount.load(std::memory_order_acquire); }

bool ShardCache::isDirty(uint32_t slot) const { return lines_[slot].dirty.load(std::memory_order_acquire); }
void ShardCache::markDirty(uint32_t slot) { lines_[slot].dirty.store(true, std::memory_order_release); }
void ShardCache::markClean(uint32_t slot) { lines_[slot].dirty.store(false, std::memory_order_release); }

std::optional<uint64_t> ShardCache::evict(uint32_t slot) {
    std::lock_guard<std::mutex> lk(mutex_);
    const LineState st = lines_[slot].state.load(std::memory_order_acquire);
    const uint64_t key = lines_[slot].key;
    if (index_.count(key)) index_.erase(key);
    policy_->onEvict(slot);
    lines_[slot].state.store(LineState::Invalid, std::memory_order_release);
    lines_[slot].dirty.store(false, std::memory_order_release);
    if (st == LineState::Invalid) return std::nullopt;
    return key;
}

uint32_t ShardCache::residentCount() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<uint32_t>(index_.size());
}

LineState ShardCache::state(uint32_t slot) const {
    return lines_[slot].state.load(std::memory_order_acquire);
}
uint64_t ShardCache::keyAt(uint32_t slot) const { return lines_[slot].key; }

} // namespace cache
} // namespace storage
} // namespace vvm