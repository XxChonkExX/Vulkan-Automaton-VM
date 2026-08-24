#pragma once

// Shard placement planning - pure logic, no I/O, no network. Part of the
// CORE layer: usable standalone with just the memory pool.
#include "vulkan_vm/core.hpp"
#include <vector>
#include <string>
#include <optional>
#include <cstdint>

// Forward declarations - in global vvm::network namespace
namespace vvm::network {
    class ModelHub;
    class MultiNodePoolManager;
}

namespace vvm::placement {

// ============================================================================
// Capacity (what each node can honestly offer)
// ============================================================================

struct NodeCapacity {
    std::string nodeId;              // same as MultiNodePoolManager node id
    VkDeviceSize vramFree = 0;       // usable DEVICE_LOCAL under maxHeapFraction
    VkDeviceSize vramTotal = 0;
    VkDeviceSize hostOffloadFree = 0; // host shadow / system RAM budget for spill
    VkDeviceSize diskCacheFree = 0;   // optional: for "weights on disk, load on demand"
    uint32_t gpuCount = 1;
    float networkBwMBps = 0;          // optional hint; NOT required for placement
    bool trusted = true;              // LAN vs random peer (policy only)
};

struct ClusterCapacity {
    std::vector<NodeCapacity> nodes;
    VkDeviceSize reservedActivationBytes = 64ull << 20; // per-node slack for KV/acts
};

// ============================================================================
// Model description (from ModelHub manifest)
// ============================================================================

enum class ShardKind {
    Weights,       // static parameters
    Tokenizer,     // small; prefer "coordinator" node
    Config,        // tiny
    Extra          // adapters, etc.
};

struct ShardSpec {
    std::string shardId;             // e.g. "layers.0-3.weight" or content hash
    std::string contentHash;         // SHA-256 of chunk set (ModelHub)
    ShardKind kind = ShardKind::Weights;
    VkDeviceSize bytes = 0;          // uncompressed size needed in memory when "hot"
    int32_t layerBegin = -1;         // optional topology for pipeline parallel
    int32_t layerEnd = -1;           // inclusive
    bool mustBeDeviceLocal = false;  // rare; default false -> may offload
    bool mustStayTogether = false;   // if true, all bytes on one node (no split of this shard)
};

struct ModelManifest {
    std::string modelId;             // "chonk/llama-3b-q4"
    std::string version;             // "v1"
    std::vector<ShardSpec> shards;
    VkDeviceSize estimatedActivationBytes = 0; // soft; planner adds slack
};

// ============================================================================
// Placement result
// ============================================================================

enum class MemTier {
    DeviceLocal,   // in UnifiedMemoryPool GPU block
    HostOffload,   // in host shadow; reload on use
    DiskCache      // only on disk until demanded (cold)
};

struct ShardPlacement {
    std::string shardId;
    std::string nodeId;
    MemTier tier = MemTier::DeviceLocal;
    // Optional: which local GPU index if node has multi-GPU
    uint32_t localDeviceIndex = 0;
};

// ============================================================================
// Error handling
// ============================================================================

enum class ErrorCode {
    Ok = 0,

    // --- Input / validation ---
    InvalidManifest,
    InvalidCluster,
    InvalidPolicy,
    EmptyShardList,
    ZeroCapacityCluster,

    // --- Planning ---
    InsufficientCapacity,
    UnsatisfiableConstraint,
    ShardTooLarge, // single shard > any node even with offload
    ActivationReserveFailed, // reservedActivationBytes can't be held

    // --- Execution / I/O ---
    NodeUnreachable,
    NodeRejectedPlan, // peer returned error
    FetchFailed,
    ChecksumMismatch,
    CacheFull,
    AllocationFailed,
    OffloadFailed,
    PartialExecute, // some shards OK, some not (see details)

    // --- Runtime / lifecycle ---
    PlanStale, // capacity changed since plan
    NodeLeft,
    AlreadyExecuted,
    Cancelled,

    InternalError
};

struct ErrorDetail {
    ErrorCode code = ErrorCode::Ok;
    std::string message; // human-readable, one line preferred
    std::string shardId; // if applicable
    std::string nodeId;  // if applicable
    VkDeviceSize bytesNeeded = 0;
    VkDeviceSize bytesAvailable = 0;
    MemTier triedTier = MemTier::DeviceLocal;
    int osErrno = 0; // optional
    VkResult vkResult = VK_SUCCESS; // optional
};

struct Status {
    ErrorCode code = ErrorCode::Ok;
    std::string message; // summary
    std::vector<ErrorDetail> details;

    explicit operator bool() const { return code == ErrorCode::Ok; }

    static Status ok() { return {}; }
    static Status fail(ErrorCode c, std::string msg) {
        Status s;
        s.code = c;
        s.message = std::move(msg);
        return s;
    }
    Status& add(ErrorDetail d) {
        details.push_back(std::move(d));
        return *this;
    }
};

// ============================================================================
// Placement plan
// ============================================================================

struct PlacementPlan {
    Status status; // replaces PlacementStatus + message
    std::vector<ShardPlacement> assignments;
    VkDeviceSize totalDeviceBytes = 0;
    VkDeviceSize totalHostBytes = 0;
    VkDeviceSize totalDiskBytes = 0;
    std::vector<std::string> nodesUsed;

    // Debug: best-effort partial packing before failure (optional)
    std::vector<ShardPlacement> partialAssignments;
};

// ============================================================================
// Policy
// ============================================================================

struct PlacementPolicy {
    // Prefer packing onto fewer nodes (easier orchestration) vs spreading
    enum class PackMode { PackDense, SpreadEven } packMode = PackMode::PackDense;

    // Allow HostOffload when VRAM full
    bool allowHostOffload = true;
    // Allow DiskCache for cold weights (load on first use)
    bool allowDiskCache = false;

    // Never place more than this fraction of a node's reported free VRAM
    float maxVramFill = 0.90f;
    float maxHostFill = 0.80f;

    // Pipeline-friendly: try to keep consecutive layers on same node
    bool preferContiguousLayers = true;

    // Reject untrusted nodes (public pool mode)
    bool requireTrusted = true;

    // If true, return partial plan with non-Ok status instead of failing
    bool bestEffort = false;

    // Execution policies
    bool failFast = true;             // stop execute on first peer failure
    bool revalidateCapacity = true;   // PlanStale check before execute
    bool transactionalNode = true;    // rollback local slice on error
    int fetchRetries = 2;
    uint64_t fetchTimeoutMs = 120000;
};

// ============================================================================
// Core API
// ============================================================================

class ShardPlacer {
public:
    // Pure function: no I/O, no Vulkan — easy to unit test
    static PlacementPlan plan(const ModelManifest& model,
                              const ClusterCapacity& cluster,
                              const PlacementPolicy& policy = {});

    // Optional: re-plan after a node leaves or capacity changes
    static PlacementPlan replan(const ModelManifest& model,
                                const ClusterCapacity& cluster,
                                const PlacementPlan& previous,
                                const PlacementPolicy& policy = {});
};

// ============================================================================
// Execution
// ============================================================================

struct ExecuteOptions {
    bool fetchIfMissing = true;      // ModelHub::fetch chunks
    bool pinImmediately = true;      // allocate in pool / offload now
    std::string cacheRoot;           // default ~/.cache/vvm/models
};

struct ExecuteResult {
    Status status;
    std::vector<std::string> completedShardIds;
    std::vector<std::string> failedShardIds;
};

class PlacementExecutor {
public:
    // node-local: only executes assignments for this nodeId
    explicit PlacementExecutor(::vvm::network::MultiNodePoolManager& node,
                               ::vvm::network::ModelHub* hub = nullptr);

    ~PlacementExecutor();

    ExecuteResult executeLocal(const ModelManifest& model,
                               const PlacementPlan& plan,
                               const ExecuteOptions& opt = {});

    // Coordinator: RPC "execute your slice of the plan" to each node
    // (thin wrapper over your control-plane messages)
    static ExecuteResult executeCluster(
        ::vvm::network::MultiNodePoolManager& local,
        ::vvm::network::ModelHub* hub,
        const ModelManifest& model,
        const PlacementPlan& plan,
        const ExecuteOptions& opt = {});

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vvm::placement