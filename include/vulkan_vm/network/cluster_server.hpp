#pragma once

#include "vulkan_vm/network/network_types.hpp"
#include "vulkan_vm/network/rdma_transport.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include <memory>
#include <vector>
#include <optional>
#include <string>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <thread>
#include <atomic>

namespace vvm {
namespace network {

// Forward declare
class MultiNodePoolManager;

// Authorization result for gRPC callbacks
enum class AuthorizationResult {
    Success = 0,
    Unauthorized = 1,
    InvalidRequest = 2,
    InternalError = 3
};

// ============================================================================
// Forward declaration for generated gRPC
// ============================================================================

namespace grpc {
class Server;
class ServerBuilder;
class ServerContext;
class Status;
class ServerCompletionQueue;
}  // namespace grpc

// ============================================================================
// ClusterServer - gRPC server for control plane
// ============================================================================

class ClusterServer {
public:
    // Factory
    static std::unique_ptr<ClusterServer> create(
        const NetworkConfig& config,
        MultiNodePoolManager* poolManager);
    
    virtual ~ClusterServer() = default;
    
    // Non-copyable, movable
    ClusterServer(const ClusterServer&) = delete;
    ClusterServer& operator=(const ClusterServer&) = delete;
    ClusterServer(ClusterServer&&) noexcept = default;
    ClusterServer& operator=(ClusterServer&&) noexcept = default;
    
    // ========================================================================
    // Lifecycle
    // ========================================================================
    
    virtual bool start() = 0;
    virtual void stop() = 0;
    virtual bool isRunning() const = 0;
    
    // ========================================================================
    // Callback registration (for handling RPCs)
    // ========================================================================
    
    using AllocateHandler = std::function<
        std::optional<RemoteAllocationDesc>(
            const NodeId& requester,
            VkDeviceSize size,
            VkBufferUsageFlags usage,
            VkMemoryPropertyFlags flags,
            bool enableRdma)>;
    
    using ExportHandler = std::function<
        std::optional<RemoteAllocationDesc>(
            const NodeId& requester,
            uint64_t localAllocId,
            bool enableRdma,
            bool forceHostShadow)>;
    
    using ImportHandler = std::function<
        std::optional<Allocation>(
            const NodeId& requester,
            const RemoteAllocationDesc& desc,
            VkBufferUsageFlags usage)>;
    
    using MigrateHandler = std::function<
        std::optional<NetworkMigrationOperation>(
            const NodeId& requester,
            const RemoteAllocationDesc& source,
            uint64_t destinationAllocId,
            bool useRdma)>;
    
using RegisterHandler = std::function<
        std::optional<std::vector<NodeInfo>>(
            const NodeInfo& info)>;

    using AuthCallback = std::function<AuthorizationResult(uint32_t messageType)>;
    
    virtual void setAllocateHandler(AllocateHandler handler) = 0;
    virtual void setExportHandler(ExportHandler handler) = 0;
    virtual void setImportHandler(ImportHandler handler) = 0;
    virtual void setMigrateHandler(MigrateHandler handler) = 0;
    virtual void setRegisterHandler(RegisterHandler handler) = 0;
    virtual void setAuthCallback(AuthCallback callback) = 0;
    
    // ========================================================================
    // Cluster state
    // ========================================================================
    
    virtual void updateClusterView(const std::vector<NodeInfo>& view) = 0;
    virtual std::vector<NodeInfo> getClusterView() const = 0;
    
    // ========================================================================
    // Stats
    // ========================================================================
    
    struct ServerStats {
        uint64_t rpcCallsReceived = 0;
        uint64_t rpcCallsSucceeded = 0;
        uint64_t rpcCallsFailed = 0;
        uint64_t bytesReceived = 0;
        uint64_t bytesSent = 0;
    };
    
    virtual ServerStats getStats() const = 0;
};

} // namespace network
} // namespace vvm