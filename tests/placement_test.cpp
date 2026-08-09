#include "vulkan_vm/placement.hpp"
#include <iostream>
#include <cassert>

using namespace vvm::placement;

int main() {
    std::cout << "=== Shard Placement Tests ===\n\n";

    // Test 1: Simple two-node cluster, shards fit
    {
        ModelManifest model;
        model.modelId = "test/model";
        model.version = "v1";
        model.shards.emplace_back(ShardSpec{"blk.0-3", "hash1", ShardKind::Weights, 2ull * 1024 * 1024 * 1024, 0, 3});
        model.shards.emplace_back(ShardSpec{"blk.4-7", "hash2", ShardKind::Weights, 2ull * 1024 * 1024 * 1024, 4, 7});
        model.shards.emplace_back(ShardSpec{"blk.8-11", "hash3", ShardKind::Weights, 2ull * 1024 * 1024 * 1024, 8, 11});

        ClusterCapacity cluster;
        cluster.nodes.emplace_back(NodeCapacity{"node-a", 8ull * 1024 * 1024 * 1024, 8ull * 1024 * 1024 * 1024, 4ull * 1024 * 1024 * 1024, 0, 1, 1000, true});
        cluster.nodes.emplace_back(NodeCapacity{"node-b", 8ull * 1024 * 1024 * 1024, 8ull * 1024 * 1024 * 1024, 4ull * 1024 * 1024 * 1024, 0, 1, 1000, true});
        cluster.reservedActivationBytes = 512ull * 1024 * 1024;

        PlacementPolicy policy;
        policy.allowHostOffload = true;

        PlacementPlan plan = ShardPlacer::plan(model, cluster, policy);
        assert(plan.status);
        assert(plan.assignments.size() == 3);
        std::cout << "Test 1 (simple fit): PASS\n";
    }

    // Test 2: Shards too large for VRAM, must offload
    {
        ModelManifest model;
        model.modelId = "test/large";
        model.version = "v1";
        model.shards.emplace_back(ShardSpec{"huge-shard", "hash1", ShardKind::Weights, 12ull * 1024 * 1024 * 1024, -1, -1});

        ClusterCapacity cluster;
        cluster.nodes.emplace_back(NodeCapacity{"node-a", 8ull * 1024 * 1024 * 1024, 8ull * 1024 * 1024 * 1024, 16ull * 1024 * 1024 * 1024, 0, 1, 1000, true});
        cluster.reservedActivationBytes = 512ull * 1024 * 1024;

        PlacementPolicy policy;
        policy.allowHostOffload = true;

        PlacementPlan plan = ShardPlacer::plan(model, cluster, policy);
        assert(plan.status);
        assert(plan.assignments[0].tier == MemTier::HostOffload);
        std::cout << "Test 2 (host offload): PASS\n";
    }

    // Test 3: mustBeDeviceLocal but no VRAM -> UnsatisfiableConstraint
    {
        ModelManifest model;
        model.modelId = "test/constraint";
        model.version = "v1";
        model.shards.emplace_back(ShardSpec{"must-be-gpu", "hash1", ShardKind::Weights, 4ull * 1024 * 1024 * 1024, -1, -1, false, false});

        ClusterCapacity cluster;
        cluster.nodes.emplace_back(NodeCapacity{"node-a", 2ull * 1024 * 1024 * 1024, 8ull * 1024 * 1024 * 1024, 0, 0, 1, 1000, true});

        PlacementPolicy policy;
        policy.allowHostOffload = true;

        PlacementPlan plan = ShardPlacer::plan(model, cluster, policy);
        assert(!plan.status);
        assert(plan.status.code == ErrorCode::UnsatisfiableConstraint);
        std::cout << "Test 3 (mustBeDeviceLocal constraint): PASS\n";
    }

    // Test 4: Empty cluster
    {
        ModelManifest model;
        model.modelId = "test/empty";
        model.shards.emplace_back(ShardSpec{"s1", "h1", ShardKind::Weights, 1024});

        ClusterCapacity cluster; // empty nodes

        PlacementPlan plan = ShardPlacer::plan(model, cluster, {});
        assert(!plan.status);
        assert(plan.status.code == ErrorCode::ZeroCapacityCluster);
        std::cout << "Test 4 (empty cluster): PASS\n";
    }

// Test 5: Contiguous layers preference
    {
        ModelManifest model;
        model.shards.emplace_back(ShardSpec{"blk.0-1", "h1", ShardKind::Weights, 1ull * 1024 * 1024 * 1024, 0, 1});
        model.shards.emplace_back(ShardSpec{"blk.2-3", "h2", ShardKind::Weights, 1ull * 1024 * 1024 * 1024, 2, 3});
        model.shards.emplace_back(ShardSpec{"blk.4-5", "h3", ShardKind::Weights, 1ull * 1024 * 1024 * 1024, 4, 5});

        ClusterCapacity cluster;
        cluster.nodes.emplace_back(NodeCapacity{"node-a", 4ull * 1024 * 1024 * 1024, 0, 0, 0, 1, 1000, true});
        cluster.nodes.emplace_back(NodeCapacity{"node-b", 4ull * 1024 * 1024 * 1024, 0, 0, 0, 1, 1000, true});

        PlacementPolicy policy;
        policy.preferContiguousLayers = true;
        policy.packMode = PlacementPolicy::PackMode::PackDense;

        PlacementPlan plan = ShardPlacer::plan(model, cluster, policy);
        assert(plan.status);
        // With PackDense + contiguous, all should go to node-a
        int onA = 0;
        for (const auto& a : plan.assignments) if (a.nodeId == "node-a") ++onA;
        assert(onA == 3);
        std::cout << "Test 5 (contiguous layers): PASS\n";
    }

    // Test 6: ShardTooLarge
    {
        ModelManifest model;
        model.shards.emplace_back(ShardSpec{"huge", "h1", ShardKind::Weights, 100ull * 1024 * 1024 * 1024});

        ClusterCapacity cluster;
        cluster.nodes.emplace_back(NodeCapacity{"node-a", 8ull * 1024 * 1024 * 1024, 0, 0, 0, 1, 1000, true});

        PlacementPlan plan = ShardPlacer::plan(model, cluster, {});
        assert(!plan.status);
        assert(plan.status.code == ErrorCode::ShardTooLarge);
        std::cout << "Test 6 (ShardTooLarge): PASS\n";
    }

    // Test 7: Validation - duplicate shardId
    {
        ModelManifest model;
        model.shards.emplace_back(ShardSpec{"s1", "h1", ShardKind::Weights, 1024});
        model.shards.emplace_back(ShardSpec{"s1", "h2", ShardKind::Weights, 1024});

        ClusterCapacity cluster;
        cluster.nodes.emplace_back(NodeCapacity{"node-a", 8ull << 30, 0, 0, 0, 1, 1000, true});

        PlacementPlan plan = ShardPlacer::plan(model, cluster, {});
        assert(!plan.status);
        assert(plan.status.code == ErrorCode::InvalidManifest);
        std::cout << "Test 7 (duplicate shardId): PASS\n";
    }

    // Test 8: Validation - empty shardId
    {
        ModelManifest model;
        model.shards.emplace_back(ShardSpec{"", "h1", ShardKind::Weights, 1024});

        ClusterCapacity cluster;
        cluster.nodes.emplace_back(NodeCapacity{"node-a", 8ull << 30, 0, 0, 0, 1, 1000, true});

        PlacementPlan plan = ShardPlacer::plan(model, cluster, {});
        assert(!plan.status);
        assert(plan.status.code == ErrorCode::InvalidManifest);
        std::cout << "Test 8 (empty shardId): PASS\n";
    }

    // Test 9: Activation reserve
    {
        ModelManifest model;
        model.shards.emplace_back(ShardSpec{"s1", "h1", ShardKind::Weights, 2ull * 1024 * 1024 * 1024, -1, -1});

        ClusterCapacity cluster;
        cluster.nodes.emplace_back(NodeCapacity{"node-a", 3ull * 1024 * 1024 * 1024, 0, 0, 0, 1, 1000, true});
        cluster.reservedActivationBytes = 1ull * 1024 * 1024 * 1024; // 1GB reserved

        PlacementPolicy policy;
        PlacementPlan plan = ShardPlacer::plan(model, cluster, policy);
        // 3GB free, 2GB shard, 1GB reserved -> fits exactly after reserve
        assert(plan.status);
        std::cout << "Test 9 (activation reserve): PASS\n";
    }

    // Test 10: bestEffort mode
    {
        ModelManifest model;
        model.shards.emplace_back(ShardSpec{"huge1", "h1", ShardKind::Weights, 8ull * 1024 * 1024 * 1024});
        model.shards.emplace_back(ShardSpec{"huge2", "h2", ShardKind::Weights, 8ull * 1024 * 1024 * 1024});

        ClusterCapacity cluster;
        cluster.nodes.emplace_back(NodeCapacity{"node-a", 8ull * 1024 * 1024 * 1024, 0, 0, 0, 1, 1000, true});

        PlacementPolicy policy;
        policy.bestEffort = true;

        PlacementPlan plan = ShardPlacer::plan(model, cluster, policy);
        assert(!plan.status);
        assert(plan.status.code == ErrorCode::InsufficientCapacity);
        assert(plan.assignments.size() == 1); // one placed
        std::cout << "Test 10 (bestEffort mode): PASS\n";
    }

    std::cout << "\n=== All tests passed ===\n";
    return 0;
}