#include "vulkan_vm/placement.hpp"
#include "vulkan_vm/network/multi_node_manager.hpp"
#include "vulkan_vm/network/model_registry.hpp"
#include "vulkan_vm/offload.hpp"
#include <algorithm>
#include <mutex>
#include <chrono>

namespace vvm::placement {

namespace {

struct ShardState {
    std::string shardId;
    std::string contentHash;
    MemTier tier = MemTier::DeviceLocal;
    enum class State { Fetching, Ready, Failed, Evicted } state = State::Fetching;
    std::unique_ptr<Allocation> allocation; // for DeviceLocal/HostOffload
    std::string cachePath; // for DiskCache
    VkDeviceSize bytes = 0;
    mutable std::mutex mutex;
};

class ShardTable {
public:
    ShardState* get(const std::string& shardId) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = table_.find(shardId);
        return it != table_.end() ? it->second.get() : nullptr;
    }

    ShardState& emplace(const std::string& shardId, const std::string& contentHash, MemTier tier, VkDeviceSize bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = table_.find(shardId);
        if (it == table_.end()) {
            auto ptr = std::make_unique<ShardState>();
            ptr->shardId = shardId;
            ptr->contentHash = contentHash;
            ptr->tier = tier;
            ptr->bytes = 0;
            auto [it2, inserted] = table_.emplace(shardId, std::move(ptr));
            return *it2->second;
        }
        // Already exists - update fields
        ShardState& state = *it->second;
        state.contentHash = contentHash;
        state.tier = tier;
        return state;
    }

    void erase(const std::string& shardId) {
        std::lock_guard<std::mutex> lock(mutex_);
        table_.erase(shardId);
    }

    std::vector<std::string> allShardIds() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> ids;
        ids.reserve(table_.size());
        for (const auto& [k, v] : table_) ids.push_back(k);
        return ids;
    }

private:
    std::unordered_map<std::string, std::unique_ptr<ShardState>> table_;
    mutable std::mutex mutex_;
};

} // namespace

class PlacementExecutor::Impl {
public:
    Impl(::vvm::network::MultiNodePoolManager& node, ::vvm::network::ModelHub* hub)
        : node_(node), hub_(hub) {
        nodeId_ = node_.getLocalNodeId().toString();
    }

    ExecuteResult executeLocal(const ModelManifest& model,
                               const PlacementPlan& plan,
                               const ExecuteOptions& opt) {
        ExecuteResult result;

        if (!plan.status) {
            result.status = plan.status;
            return result;
        }

        // Revalidate capacity if requested (stale check)
        if (policy_.revalidateCapacity) {
            // Could gather current capacity and check stillFits()
            // For v0, skip detailed check
        }

        // Extract this node's assignments
        std::vector<ShardPlacement> myAssignments;
        for (const auto& a : plan.assignments) {
            if (a.nodeId == nodeId_) myAssignments.push_back(a);
        }

        if (myAssignments.empty()) {
            result.status = Status::ok();
            return result;
        }

        // Execute each shard
        for (const auto& ap : myAssignments) {
            // Find the shard spec
            const ShardSpec* spec = nullptr;
            for (const auto& s : model.shards) {
                if (s.shardId == ap.shardId) { spec = &s; break; }
            }
            if (!spec) {
                result.status.add({ErrorCode::InvalidManifest, "shard spec not found in model", ap.shardId, nodeId_});
                result.failedShardIds.push_back(ap.shardId);
                if (policy_.failFast) break;
                continue;
            }

            Status shardStatus = executeShard(*spec, ap, opt);
            if (shardStatus) {
                result.completedShardIds.push_back(ap.shardId);
            } else {
                result.failedShardIds.push_back(ap.shardId);
                result.status = shardStatus; // propagate error
                if (policy_.failFast) break;
            }
        }

        if (result.completedShardIds.empty() && !result.failedShardIds.empty()) {
            result.status.code = ErrorCode::PartialExecute;
        } else if (!result.failedShardIds.empty()) {
            result.status.code = ErrorCode::PartialExecute;
        } else {
            result.status = Status::ok();
        }

        // Rollback on failure if transactional
        if (!result.failedShardIds.empty() && policy_.transactionalNode) {
            rollbackLocal(result.failedShardIds);
        }

        return result;
    }

private:
    ::vvm::network::MultiNodePoolManager& node_;
    ::vvm::network::ModelHub* hub_;
    std::string nodeId_;
    PlacementPolicy policy_;
    ShardTable shardTable_;

    Status executeShard(const ShardSpec& spec, const ShardPlacement& placement, const ExecuteOptions& opt) {
        // Idempotency: check if already ready with same hash
        ShardState* existing = shardTable_.get(spec.shardId);
        if (existing) {
            std::lock_guard<std::mutex> lock(existing->mutex);
            if (existing->state == ShardState::State::Ready && existing->contentHash == spec.contentHash) {
                return Status::ok();
            }
            if (existing->state == ShardState::State::Ready && existing->contentHash != spec.contentHash) {
                return Status::fail(ErrorCode::InvalidManifest, "shard " + spec.shardId + " already loaded with different content hash");
            }
        }

        // Ensure entry exists
        ShardState& state = shardTable_.emplace(spec.shardId, spec.contentHash, placement.tier, spec.bytes);
        state.bytes = spec.bytes;

        // Fetch if needed
        if (opt.fetchIfMissing && hub_) {
            // ModelHub::fetch(shard) -> local cache path
            // For now, assume files are already in cache or fetch is handled by hub
        }

        // Allocate based on tier
        if (placement.tier == MemTier::DeviceLocal || placement.tier == MemTier::HostOffload) {
            // Use UnifiedMemoryPool via MultiNodePoolManager
            // This is a simplified version; actual implementation needs pool access
            // For now, return success
        } else if (placement.tier == MemTier::DiskCache) {
            // Just verify cache file exists
        }

        return Status::ok();
    }

    void rollbackLocal(const std::vector<std::string>& shardIds) {
        for (const auto& id : shardIds) {
            if (auto* state = shardTable_.get(id)) {
                // Free allocation if any
                if (state->allocation) {
                    // Would call pool_->deallocate(std::move(*state->allocation));
                }
                shardTable_.erase(id);
            }
        }
    }
};

PlacementExecutor::PlacementExecutor(::vvm::network::MultiNodePoolManager& node, ::vvm::network::ModelHub* hub)
    : impl_(std::make_unique<Impl>(node, hub)) {}

PlacementExecutor::~PlacementExecutor() = default;

ExecuteResult PlacementExecutor::executeLocal(const ModelManifest& model,
                                              const PlacementPlan& plan,
                                              const ExecuteOptions& opt) {
    return impl_->executeLocal(model, plan, opt);
}

ExecuteResult PlacementExecutor::executeCluster(
    ::vvm::network::MultiNodePoolManager& local,
    ::vvm::network::ModelHub* hub,
    const ModelManifest& model,
    const PlacementPlan& plan,
    const ExecuteOptions& opt) {
    // Coordinator: send execute RPC to each node
    // For v0, just execute locally
    PlacementExecutor executor(local, hub);
    return executor.executeLocal(model, plan, opt);
}

} // namespace vvm::placement