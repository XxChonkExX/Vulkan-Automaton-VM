#pragma once

#include "vulkan_vm/network/network_types.hpp"
#include "vulkan_vm/network/rdma_transport.hpp"
#include <memory>
#include <vector>
#include <optional>
#include <string>
#include <functional>
#include <future>
#include <mutex>
#include <unordered_map>

namespace vvm {
namespace network {

// ============================================================================
// Forward declaration for generated gRPC
// ============================================================================

namespace grpc {
class Channel;
class ClientContext;
class Status;
}  // namespace grpc

// ============================================================================
// ClusterClient - gRPC client for control plane
// ============================================================================

class ClusterClient {
public:
    // Factory
    static std::unique_ptr<ClusterClient> create(const NetworkConfig& config);
    
    virtual ~ClusterClient() = default;
    
    // Non-copyable, movable
    ClusterClient(const ClusterClient&) = delete;
    ClusterClient& operator=(const ClusterClient&) = delete;
    ClusterClient(ClusterClient&&) noexcept = default;
    ClusterClient& operator=(ClusterClient&&) noexcept = default;
    
    // ========================================================================
    // Connection
    // ========================================================================
    
    virtual bool connect(const std::string& target) = 0;
    virtual void disconnect() = 0;
    virtual bool isConnected() const = 0;
    
    // ========================================================================
    // Allocation RPCs
    // ========================================================================
    
    virtual std::optional<RemoteAllocationDesc> allocateRemote(
        const NodeId& target,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags flags,
        bool enableRdma,
        uint64_t timeoutNs = UINT64_MAX) = 0;
    
    virtual std::future<std::optional<RemoteAllocationDesc>> allocateRemoteAsync(
        const NodeId& target,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags flags,
        bool enableRdma) = 0;
    
    // ========================================================================
    // Export/Import RPCs
    // ========================================================================
    
    virtual std::optional<RemoteAllocationDesc> exportRemote(
        const NodeId& target,
        uint64_t localAllocId,
        bool enableRdma,
        bool forceHostShadow,
        uint64_t timeoutNs = UINT64_MAX) = 0;
    
    virtual std::optional<Allocation> importRemote(
        const NodeId& target,
        const RemoteAllocationDesc& desc,
        VkBufferUsageFlags usage,
        uint64_t timeoutNs = UINT64_MAX) = 0;
    
    virtual std::future<std::optional<Allocation>> importRemoteAsync(
        const NodeId& target,
        const RemoteAllocationDesc& desc,
        VkBufferUsageFlags usage) = 0;
    
    // ========================================================================
    // Migration RPCs
    // ========================================================================
    
    virtual std::optional<NetworkMigrationOperation> migrate(
        const NodeId& target,
        const RemoteAllocationDesc& source,
        uint64_t destinationAllocId,
        bool useRdma,
        uint64_t timeoutNs = UINT64_MAX) = 0;
    
    // ========================================================================
    // Cluster management
    // ========================================================================
    
    virtual std::optional<std::vector<NodeInfo>> registerNode(
        const NodeInfo& info,
        uint64_t timeoutNs = UINT64_MAX) = 0;
    
    virtual std::optional<std::vector<NodeInfo>> heartbeat(
        const NodeId& node,
        uint64_t timeoutNs = UINT64_MAX) = 0;
    
    virtual std::optional<std::vector<NodeInfo>> getClusterView(
        uint64_t timeoutNs = UINT64_MAX) = 0;
    
    // ========================================================================
    // Streaming (for long-running migrations)
    // ========================================================================
    
    virtual bool startMigrationStream(
        const NodeId& target,
        std::function<void(const NetworkMigrationOperation&)> progressCallback) = 0;
    
    virtual void stopMigrationStream() = 0;
};

} // namespace network
} // namespace vvm