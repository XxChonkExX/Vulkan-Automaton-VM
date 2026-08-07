#pragma once

#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/network/network_types.hpp"
#include "vulkan_vm/network/tcp_transport.hpp"
#include "vulkan_vm/network/rdma_transport.hpp"
#include <memory>
#include <vector>
#include <optional>
#include <future>
#include <string>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <atomic>
#include <thread>

namespace vvm {
namespace network {

// ============================================================================
// Forward declarations
// ============================================================================

class ClusterClient;
class ClusterServer;
class RdmaTransport;

// ============================================================================
// MultiNodePoolManager
// ============================================================================

class MultiNodePoolManager {
public:
    // Factory
    static std::optional<MultiNodePoolManager> create(
        const std::vector<vvm::DeviceConfig>& localDevices,
        const vvm::PoolConfig& poolConfig,
        const NetworkConfig& networkConfig);
    
    // Non-copyable, movable
    MultiNodePoolManager(const MultiNodePoolManager&) = delete;
    MultiNodePoolManager& operator=(const MultiNodePoolManager&) = delete;
    MultiNodePoolManager(MultiNodePoolManager&&) noexcept;
    MultiNodePoolManager& operator=(MultiNodePoolManager&&) noexcept;
    ~MultiNodePoolManager();
    
    // ========================================================================
    // Local pool access
    // ========================================================================
    
    UnifiedMemoryPool& getLocalPool(uint32_t deviceIndex = 0) {
        return localPools_[deviceIndex];
    }
    
    const UnifiedMemoryPool& getLocalPool(uint32_t deviceIndex = 0) const {
        return localPools_[deviceIndex];
    }
    
    size_t getLocalPoolCount() const { return localPools_.size(); }
    
    // ========================================================================
    // Local allocation (same as UnifiedMemoryPool but with network awareness)
    // ========================================================================
    
    // Allocate locally and optionally advertise to cluster
    std::optional<Allocation> allocateLocal(
        VkDeviceSize size,
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VkMemoryPropertyFlags flags = 0,
        bool advertise = false);  // register in cluster directory
    
    // Allocate tensor-optimized (bindless-ready)
    std::optional<Allocation> allocateTensor(
        VkDeviceSize size,
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        bool advertise = false);
    
    // Deallocate local allocation
    void deallocateLocal(Allocation&& alloc);

    // Deallocate an allocation referenced by a remote descriptor (local or remote owner)
    bool deallocateRemote(const RemoteAllocationDesc& desc);
    
    // ========================================================================
    // Remote allocation (request remote node to allocate)
    // ========================================================================
    
    // Async: request remote node to allocate and return descriptor
    std::future<RemoteAllocationDesc> allocateRemoteAsync(
        const NodeId& target,
        VkDeviceSize size,
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        bool enableRdma = true);
    
    // Sync version
    std::optional<RemoteAllocationDesc> allocateRemote(
        const NodeId& target,
        VkDeviceSize size,
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        bool enableRdma = true,
        uint64_t timeoutNs = UINT64_MAX);
    
    // ========================================================================
    // Export local allocation for remote access
    // ========================================================================
    
    // Export local allocation for remote RDMA or host-staged access
    std::optional<RemoteAllocationDesc> exportForRemote(
        const Allocation& alloc,
        bool enableRdma = true,
        bool forceHostShadow = false);
    
    // ========================================================================
    // Import remote allocation
    // ========================================================================
    
    // Import remote allocation (RDMA or host-staged fallback)
    std::optional<Allocation> importRemote(
        const RemoteAllocationDesc& desc,
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    
    // Async import
    std::future<std::optional<Allocation>> importRemoteAsync(
        const RemoteAllocationDesc& desc,
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    
    // ========================================================================
    // Migration (data movement)
    // ========================================================================
    
    // Pull data from remote into local allocation
    std::optional<NetworkMigrationOperation> migrateFromRemote(
        const RemoteAllocationDesc& source,
        Allocation& destination,
        bool useRdma = true,
        uint64_t timeoutNs = UINT64_MAX);
    
    // Push local data to remote allocation
    std::optional<NetworkMigrationOperation> migrateToRemote(
        Allocation& source,
        const RemoteAllocationDesc& destination,
        bool useRdma = true,
        uint64_t timeoutNs = UINT64_MAX);
    
    // Wait for migration completion
    void waitMigration(NetworkMigrationOperation& op);
    bool pollMigration(NetworkMigrationOperation& op);
    
    // Async migration with callback
    using MigrationCallback = std::function<void(const NetworkMigrationOperation&)>;
    void migrateFromRemoteAsync(
        const RemoteAllocationDesc& source,
        Allocation& destination,
        MigrationCallback callback,
        bool useRdma = true);
    
    // ========================================================================
    // Cluster management
    // ========================================================================
    
    // Get current cluster view
    std::vector<NodeInfo> getClusterView() const;
    
    // Find node by GPU vendor/device
    std::optional<NodeId> findNodeByGpu(uint32_t vendorId, uint32_t deviceId) const;
    
    // Find node with specific capability
    std::optional<NodeId> findNodeWithCapability(bool requireRdma, bool requireGpuDirect) const;
    
    // Look up a registered (remote-visible) allocation by its local id.
    std::optional<Allocation> getRegisteredAllocation(uint64_t localAllocId) const;

    // ========================================================================
    // Remote tensor announcement (GPU-to-GPU VRAM share over TCP)
    // ========================================================================

    // Advertise a tensor (by name) to a specific peer. The peer receives a
    // RemoteAllocationDesc it can use to pull the VRAM with migrateFromRemote.
    bool announceRemoteTensor(
        const NodeId& target,
        const std::string& name,
        const RemoteAllocationDesc& desc);

    // Wait until a peer announces a tensor with the given name; returns its
    // descriptor. Blocks up to timeoutNs. The tensor then stays in the peer's
    // VRAM and can be pulled with migrateFromRemote().
    std::optional<RemoteAllocationDesc> waitRemoteTensor(
        const NodeId& source,
        const std::string& name,
        uint64_t timeoutNs = UINT64_MAX);
    
    // Manual cluster operations
    bool registerWithCluster();
    void leaveCluster();
    
    // ========================================================================
    // Statistics and monitoring
    // ========================================================================
    
    struct NetworkStats {
        uint64_t bytesSentRdma = 0;
        uint64_t bytesReceivedRdma = 0;
        uint64_t bytesSentHostStaged = 0;
        uint64_t bytesReceivedHostStaged = 0;
        uint32_t activeMigrations = 0;
        uint32_t completedMigrations = 0;
        uint32_t failedMigrations = 0;
        double avgMigrationLatencyMs = 0.0;
    };
    
    NetworkStats getNetworkStats() const;
    void resetNetworkStats();
    
    // ========================================================================
    // Lifecycle
    // ========================================================================
    
    // Start/stop network services
    bool start();
    void stop();
    bool isRunning() const;
    
    // Get local node ID
    const NodeId& getLocalNodeId() const { return localNodeId_; }
    
    // Get network config
    const NetworkConfig& getNetworkConfig() const { return networkConfig_; }
    
    // True when a usable RDMA transport is active (verbs built in + live NIC).
    bool rdmaAvailable() const {
        return rdmaTransport_ != nullptr && rdmaTransport_->isReady();
    }
    
private:
    friend class ClusterClient;
    friend class ClusterServer;
    
    // Private constructor (use create())
    MultiNodePoolManager(
        const std::vector<DeviceConfig>& localDevices,
        const PoolConfig& poolConfig,
        const NetworkConfig& networkConfig);
    
    bool initialize();
    void cleanup();
    
    // Network components
    std::unique_ptr<ClusterClient> clusterClient_;   // optional gRPC client (when built with gRPC)
    std::unique_ptr<ClusterServer> clusterServer_;   // optional gRPC server (when built with gRPC)
    std::unique_ptr<RdmaTransport> rdmaTransport_;   // optional verbs transport (when built with libibverbs)

#if defined(VVM_NETWORK_HAS_VERBS)
    // One in-flight RDMA "host shadow" export: a host staging buffer holding a
    // copy of the exported data, kept RDMA-addressable by a registered MR.
    struct RdmaExport {
        RdmaMemoryRegion region;   // verbs MR over hostShadow.hostPtr
        Allocation hostShadow;     // host copy of the exported data
        VkDeviceSize size = 0;
    };
    // Key = pendingKey(localNodeId_, allocId).
    std::unordered_map<std::string, RdmaExport> rdmaShadowExports_;
    mutable std::mutex rdmaShadowMutex_;

    // Persistent RDMA connections: peer.toString() -> connection.
    std::unordered_map<std::string, RdmaConnection> rdmaConnections_;
    mutable std::mutex rdmaConnectionsMutex_;
#endif

    // TCP host-staged control + data plane (always available)
    std::unique_ptr<TcpTransport> tcpTransport_;
    uint16_t tcpPort_ = 0;
    std::string localHost_ = "127.0.0.1";
    std::vector<std::pair<std::string, uint16_t>> seedEndpoints_;

    // Copy engine: device <-> host staging (for host-staged migration)
    VkCommandPool copyCmdPool_ = VK_NULL_HANDLE;
    VkQueue transferQueue_ = VK_NULL_HANDLE;
    uint32_t transferQueueFamily_ = UINT32_MAX;

    // Allocation registry: localAllocId -> Allocation (for remote access)
    std::unordered_map<uint64_t, Allocation> remoteAllocs_;
    mutable std::mutex allocsMutex_;
    uint64_t nextAllocId_ = 1;

    // Persistent peer connections: "host:port" -> ConnId
    std::unordered_map<std::string, TcpTransport::ConnId> peerConns_;
    mutable std::mutex connsMutex_;

    // Heartbeat
    std::thread heartbeatThread_;
    std::atomic<bool> stopHeartbeat_{false};
    mutable std::mutex heartbeatMutex_;
    mutable std::condition_variable heartbeatCV_;

    // Pending remote tensor announcements: key = "sourceNode|name" -> descriptor
    std::unordered_map<std::string, RemoteAllocationDesc> pendingRemoteTensors_;
    mutable std::mutex remoteTensorsMutex_;
    std::condition_variable remoteTensorsCV_;
    
    // Local pools (one per GPU)
    std::vector<UnifiedMemoryPool> localPools_;
    std::vector<DeviceConfig> localDeviceConfigs_;
    PoolConfig poolConfig_;
    NetworkConfig networkConfig_;
    
    // Local node identity
    NodeId localNodeId_;
    
    // Active migrations
    struct ActiveMigration {
        NetworkMigrationOperation op;
        std::shared_ptr<MigrationCallback> callback;
    };
    std::unordered_map<uint64_t, ActiveMigration> activeMigrations_;
    mutable std::mutex migrationsMutex_;
    uint64_t nextMigrationId_ = 1;

    // Cluster state
    std::vector<NodeInfo> clusterView_;
    mutable std::mutex clusterViewMutex_;
    bool running_ = false;
    
    // Stats
    mutable std::mutex statsMutex_;
    NetworkStats networkStats_;
    
    // Internal helpers
    std::optional<Allocation> createLocalAllocationForImport(
        const RemoteAllocationDesc& desc,
        VkBufferUsageFlags usage);
    
    bool registerMemoryForRdma(const Allocation& alloc, uint64_t& outRdmaAddr, uint32_t& outRkey);
    void unregisterMemoryForRdma(const Allocation& alloc);

    // Release the host-shadow export (MR + staging) for a local allocation id.
    void releaseRdmaExport(uint64_t localAllocId);
    // Return (and cache) an established RDMA connection to the peer node.
    std::optional<RdmaConnection> ensureRdmaConnection(const NodeId& peer);
    // Tear down all host-shadow exports (stop()/cleanup()).
    void releaseAllRdmaExports();
    
    // Host-staged fallback
    std::optional<NetworkMigrationOperation> migrateHostStaged(
        const RemoteAllocationDesc& source,
        Allocation& destination,
        bool toHost,  // true = pull from remote, false = push to remote
        uint64_t timeoutNs);

    // ========================================================================
    // TCP host-staged helpers
    // ========================================================================

    bool initCopyEngine();
    std::optional<Allocation> createStaging(VkDeviceSize size);
    bool copyDeviceToHost(const Allocation& src, VkDeviceSize srcOffset,
                          const Allocation& staging, VkDeviceSize size);
    bool copyHostToDevice(const Allocation& staging, const Allocation& dst,
                          VkDeviceSize dstOffset, VkDeviceSize size);
    bool runCopy(VkBuffer srcBuffer, VkBuffer dstBuffer,
                 VkDeviceSize srcOffset, VkDeviceSize dstOffset, VkDeviceSize size);

    uint64_t registerAllocation(Allocation&& alloc);
    std::optional<Allocation> findAllocation(uint64_t localAllocId);
    bool unregisterAllocation(uint64_t localAllocId);
    std::optional<uint64_t> findAllocIdByBuffer(VkBuffer buffer);

    TcpTransport::ConnId getPeerConnection(const std::string& host, uint16_t port);
    void onTcpRequest(TcpMessage& request, TcpMessage& response);
    std::optional<std::vector<NodeInfo>> handleRegisterRequest(const NodeInfo& info);
    std::optional<RemoteAllocationDesc> handleAllocateRequest(const NodeId& requester, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags, bool enableRdma);
    std::optional<RemoteAllocationDesc> handleExportRequest(const NodeId& owner, uint64_t localAllocId, bool enableRdma, bool forceHostShadow);
    std::optional<Allocation> handleImportRequest(const NodeId& owner, const RemoteAllocationDesc& desc, VkBufferUsageFlags usage);
    std::optional<NetworkMigrationOperation> handleMigrateRequest(const RemoteAllocationDesc& source, uint64_t destinationAllocId, bool useRdma, uint64_t timeoutNs);
    void heartbeatLoop();
    void mergeClusterView(const std::vector<NodeInfo>& view);
    void parseListenAddress(const std::string& listenAddress, std::string& outHost, uint16_t& outPort);
    static bool parseEndpoint(const std::string& endpoint, std::string& outHost, uint16_t& outPort);
};

} // namespace network
} // namespace vvm