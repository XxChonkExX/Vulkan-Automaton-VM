#include "vulkan_vm/placement.hpp"
#include "vulkan_vm/utils.hpp"
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <set>

namespace vvm::placement {

namespace {

struct NodeBudget {
    std::string nodeId;
    VkDeviceSize vramFree = 0;
    VkDeviceSize hostFree = 0;
    VkDeviceSize diskFree = 0;
    VkDeviceSize originalVramFree = 0;
    uint32_t gpuCount = 1;
    bool trusted = true;
    VkDeviceSize reservedActivation = 0;
    bool activationReserved = false;
};

struct ShardWork {
    ShardSpec spec;
    size_t originalIndex = 0;
};

enum class TryResult {
    Success,
    InsufficientCapacity,
    UnsatisfiableConstraint,
    ShardTooLarge
};

struct TryInfo {
    TryResult result = TryResult::InsufficientCapacity;
    VkDeviceSize bestFree = 0;
    MemTier bestTier = MemTier::DeviceLocal;
    std::string nodeId;
    size_t nodeIdx = SIZE_MAX;
};

static PlacementPlan finalizePlan(PlacementPlan& plan) {
    std::set<std::string> nodesUsedSet;
    VkDeviceSize totalDevice = 0, totalHost = 0, totalDisk = 0;

    for (const auto& a : plan.assignments) {
        if (!a.nodeId.empty()) nodesUsedSet.insert(a.nodeId);
    }
    plan.nodesUsed.assign(nodesUsedSet.begin(), nodesUsedSet.end());
    return plan;
}

} // namespace

PlacementPlan ShardPlacer::plan(const ModelManifest& model,
                                const ClusterCapacity& cluster,
                                const PlacementPolicy& policy) {
    PlacementPlan plan;

    // --- Validation ---
    if (model.shards.empty()) {
        plan.status = Status::fail(ErrorCode::EmptyShardList, "manifest has no shards");
        return plan;
    }
    for (size_t i = 0; i < model.shards.size(); ++i) {
        const auto& shard = model.shards[i];
        if (shard.shardId.empty()) {
            plan.status = Status::fail(ErrorCode::InvalidManifest, "shard at index " + std::to_string(i) + " has empty shardId")
                .add({ErrorCode::InvalidManifest, "empty shardId", "", "", 0, 0});
            return plan;
        }
        if (shard.kind == ShardKind::Weights && shard.bytes == 0) {
            plan.status = Status::fail(ErrorCode::InvalidManifest, "weight shard '" + shard.shardId + "' has zero size")
                .add({ErrorCode::InvalidManifest, "bytes must be > 0", shard.shardId, "", 0, 0});
            return plan;
        }
    }

    // Check for duplicate shardIds
    {
        std::unordered_set<std::string> seen;
        for (const auto& s : model.shards) {
            if (!seen.insert(s.shardId).second) {
                plan.status = Status::fail(ErrorCode::InvalidManifest, "duplicate shardId: " + s.shardId)
                    .add({ErrorCode::InvalidManifest, "duplicate shardId", s.shardId, "", 0, 0});
                return plan;
            }
        }
    }

    // Filter nodes
    std::vector<NodeBudget> nodes;
    nodes.reserve(cluster.nodes.size());
    for (const auto& nc : cluster.nodes) {
        if (policy.requireTrusted && !nc.trusted) continue;
        if (nc.vramFree == 0 && nc.hostOffloadFree == 0 && nc.diskCacheFree == 0) continue;

        NodeBudget nb;
        nb.nodeId = nc.nodeId;
        nb.vramFree = static_cast<VkDeviceSize>(nc.vramFree * policy.maxVramFill);
        nb.hostFree = static_cast<VkDeviceSize>(nc.hostOffloadFree * policy.maxHostFill);
        nb.diskFree = nc.diskCacheFree;
        nb.originalVramFree = nc.vramFree;
        nb.gpuCount = nc.gpuCount;
        nb.trusted = nc.trusted;
        nb.reservedActivation = cluster.reservedActivationBytes;
        nb.activationReserved = false;
        nodes.push_back(std::move(nb));
    }

    if (nodes.empty()) {
        plan.status = Status::fail(ErrorCode::ZeroCapacityCluster, "no usable nodes after filters");
        return plan;
    }

    // Prepare shards: sort by size descending (largest first)
    std::vector<ShardWork> work;
    work.reserve(model.shards.size());
    for (size_t i = 0; i < model.shards.size(); ++i) {
        work.push_back({model.shards[i], i});
    }

    // Sort: largest first, then by layerBegin for contiguous preference
    std::sort(work.begin(), work.end(), [](const ShardWork& a, const ShardWork& b) {
        if (a.spec.bytes != b.spec.bytes) return a.spec.bytes > b.spec.bytes;
        return a.spec.layerBegin < b.spec.layerBegin;
    });

    // Track last node used for contiguous layers
    std::string lastNodeForLayer;

    // Packing
    std::vector<ShardPlacement> assignments;
    std::vector<ShardPlacement> partialAssignments;

    for (const auto& sw : work) {
        const ShardSpec& shard = sw.spec;
        VkDeviceSize need = shard.bytes;

        // Try tiers in order
        std::vector<MemTier> tiers;
        tiers.push_back(MemTier::DeviceLocal);
        if (policy.allowHostOffload && !shard.mustBeDeviceLocal) tiers.push_back(MemTier::HostOffload);
        if (policy.allowDiskCache) tiers.push_back(MemTier::DiskCache);

        // Determine preferred node (contiguous layers)
        std::string preferredNode;
        if (policy.preferContiguousLayers && shard.layerBegin >= 0) {
            // Find previous layer's placement
            for (const auto& ap : assignments) {
                const auto& prevShard = model.shards[ap.shardId == sw.spec.shardId ? 0 : 0];
                // Simplified: use the node of the immediately previous layer if it exists
                // This is a heuristic; a full implementation would track layer ranges
            }
            // For v0, we'll just use the last node if it has room
            if (!lastNodeForLayer.empty()) {
                preferredNode = lastNodeForLayer;
            }
        }

        // Determine node order
        std::vector<size_t> nodeOrder;
        nodeOrder.reserve(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) nodeOrder.push_back(i);

        if (policy.packMode == PlacementPolicy::PackMode::PackDense) {
            // Denser first: sort by available capacity ascending (fill fullest first)
            std::sort(nodeOrder.begin(), nodeOrder.end(),
                [&](size_t a, size_t b) { return nodes[a].vramFree < nodes[b].vramFree; });
        } else {
            // Spread even: sort by available capacity descending
            std::sort(nodeOrder.begin(), nodeOrder.end(),
                [&](size_t a, size_t b) { return nodes[a].vramFree > nodes[b].vramFree; });
        }

        // Try each tier
        TryInfo bestTry;
        bool placed = false;

        for (MemTier tier : tiers) {
            for (size_t idx : nodeOrder) {
                NodeBudget& node = nodes[idx];

                VkDeviceSize available = 0;
                switch (tier) {
                    case MemTier::DeviceLocal: available = node.vramFree; break;
                    case MemTier::HostOffload: available = node.hostFree; break;
                    case MemTier::DiskCache: available = node.diskFree; break;
                }

                if (available < need) {
                    if (available > bestTry.bestFree) {
                        bestTry.bestFree = available;
                        bestTry.bestTier = tier;
                        bestTry.nodeId = node.nodeId;
                    }
                    continue;
                }

                // Check activation reservation
                if (!node.activationReserved && node.reservedActivation > 0) {
                    if (node.vramFree >= node.reservedActivation) {
                        node.vramFree -= node.reservedActivation;
                        node.activationReserved = true;
                    }
                }

                // Allocate
                switch (tier) {
                    case MemTier::DeviceLocal: node.vramFree -= need; break;
                    case MemTier::HostOffload: node.hostFree -= need; break;
                    case MemTier::DiskCache: node.diskFree -= need; break;
                }

                // Record placement
                ShardPlacement sp;
                sp.shardId = shard.shardId;
                sp.nodeId = node.nodeId;
                sp.tier = tier;
                sp.localDeviceIndex = 0; // v0: single GPU per node

                assignments.push_back(sp);
                partialAssignments.push_back(sp);
                lastNodeForLayer = node.nodeId;
                placed = true;
                break;
            }
            if (placed) break;
        }

        if (!placed) {
            // Build error detail
            ErrorDetail detail;
            detail.code = (shard.mustBeDeviceLocal) ? ErrorCode::UnsatisfiableConstraint : ErrorCode::InsufficientCapacity;
            detail.message = "no node can hold shard at any allowed tier";
            detail.shardId = shard.shardId;
            detail.nodeId = bestTry.nodeId;
            detail.bytesNeeded = need;
            detail.bytesAvailable = bestTry.bestFree;
            detail.triedTier = bestTry.bestTier;

            if (shard.mustBeDeviceLocal) {
                detail.code = ErrorCode::UnsatisfiableConstraint;
                detail.message = "mustBeDeviceLocal but no device-local region large enough";
            }

            // Check if shard is larger than any possible node even with all tiers
            VkDeviceSize maxTotal = 0;
            for (const auto& n : nodes) {
                VkDeviceSize total = n.vramFree;
                if (policy.allowHostOffload) total += n.hostFree;
                if (policy.allowDiskCache) total += n.diskFree;
                maxTotal = std::max(maxTotal, total);
            }
            if (need > maxTotal) {
                detail.code = ErrorCode::ShardTooLarge;
                detail.message = "shard larger than any single node capacity (all tiers combined)";
            }

            if (policy.bestEffort) {
                // Return partial plan with failure status
                plan.status = Status::fail(detail.code, detail.message).add(std::move(detail));
                plan.assignments = std::move(assignments);
                plan.partialAssignments = std::move(partialAssignments);
                return finalizePlan(plan);
            } else {
                plan.status = Status::fail(detail.code, detail.message).add(std::move(detail));
                plan.partialAssignments = std::move(partialAssignments);
                return plan;
            }
        }
    }

    // Success
    plan.status = Status::ok();
    plan.assignments = std::move(assignments);
    return finalizePlan(plan);
}

static PlacementPlan finalizePlan(PlacementPlan& plan) {
    std::set<std::string> nodesUsedSet;
    VkDeviceSize totalDevice = 0, totalHost = 0, totalDisk = 0;

    for (const auto& a : plan.assignments) {
        if (!a.nodeId.empty()) nodesUsedSet.insert(a.nodeId);
        // We'd need to sum bytes from the spec; for now just count tiers
        if (a.tier == MemTier::DeviceLocal) {
            // Would sum from shard spec
        }
    }
    plan.nodesUsed.assign(nodesUsedSet.begin(), nodesUsedSet.end());
    return plan;
}

PlacementPlan ShardPlacer::replan(const ModelManifest& model,
                                  const ClusterCapacity& cluster,
                                  const PlacementPlan& previous,
                                  const PlacementPolicy& policy) {
    // For v0: just re-plan from scratch. A full implementation would try to
    // preserve existing placements where capacity still fits.
    (void)previous;
    return plan(model, cluster, policy);
}

} // namespace vvm::placement