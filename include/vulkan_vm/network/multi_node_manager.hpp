#pragma once

#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/network/network_types.hpp"
#include "vulkan_vm/network/tcp_transport.hpp"
#include "vulkan_vm/network/cluster_client.hpp"
#include "vulkan_vm/network/cluster_server.hpp"
#include "vulkan_vm/network/rdma_transport.hpp"
#include <memory>
#include <vector>
#include <optional>
#include <future>
#include <string>
#include <functional>
#include <mutex>
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
#if defined(VVM_HAS_UCX)
class UcxTransport;
#endif

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
    
    // Check if RDMA is available
    bool rdmaAvailable() const;
    
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
    // UCX Transport Integration (only available when UCX is built)
    // ========================================================================
    
    #if defined(VVM_HAS_UCX)
    // Set UCX transport for GPU-aware RMA
    void setUcxTransport(vvm::network::UcxTransport* ucxTransport);
    
    // UCX-enabled export: registers GPU memory with UCX and returns RMA keys
    std::optional<bool> exportForRemoteUcx(
        const RemoteAllocationDesc& desc,
        const Allocation& alloc,
        uint32_t deviceIndex);
    
    // UCX-enabled migration: pulls data via UCX RMA (GPU-aware)
    std::optional<NetworkMigrationOperation> migrateFromRemoteUcx(
        const RemoteAllocationDesc& source,
        Allocation& destination,
        bool useRdma = true,
        uint64_t timeoutNs = UINT64_MAX);
    
    // Wait for UCX migration completion
    void waitUcxMigration(NetworkMigrationOperation& op);
    #endif
    
    // ========================================================================
    // Cluster name-based operations (used by TensorTransport)
    // ========================================================================
    
    // Announce a tensor name is available on this node for remote pull
    bool announceRemoteTensor(const NodeId& targetNode, const std::string& tensorName, const RemoteAllocationDesc& desc);
    
    // Wait for a tensor announcement from a remote node
    std::optional<RemoteAllocationDesc> waitRemoteTensor(const NodeId& sourceNode, const std::string& tensorName, uint64_t timeoutNs);
    
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
    
    // Access internal components (for TensorTransport integration)
    class TcpTransport* getTcpTransport() const { return tcpTransport_.get(); }
    TcpTransport::ConnId getPeerConnection(const std::string& host, uint16_t port);
    std::mutex& clusterViewMutex() { return clusterViewMutex_; }
    std::vector<NodeInfo>& clusterView() { return clusterView_; }
    
    // Get network config
    const NetworkConfig& getNetworkConfig() const { return networkConfig_; }
    
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

    // TCP host-staged control + data plane (always available)
    std::unique_ptr<TcpTransport> tcpTransport_;
    uint16_t tcpPort_ = 0;
    std::string localHost_ = "127.0.0.1";
    std::vector<std::pair<std::string, uint16_t>> seedEndpoints_;

    // Copy engine: device <-> host staging (for host-staged migration)
    VkCommandPool copyCmdPool_ = VK_NULL_HANDLE;
    VkQueue transferQueue_ = VK_NULL_HANDLE;
    uint32_t transferQueueFamily_ = UINT32_MAX;

    // Pre-allocated command buffer/fence pool for async copies (avoids per-copy alloc)
    struct CopyContext {
        VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool inUse = false;
    };
    std::vector<CopyContext> copyContexts_;
    std::mutex copyContextsMutex_;
    uint32_t maxCopyContexts_ = 8;  // configurable via NetworkConfig

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
    
    // Announced remote tensors (name -> descriptor)
    std::unordered_map<std::string, RemoteAllocationDesc> announcedTensors_;
    
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

private:
    friend class ClusterClient;
    friend class ClusterServer;
    
    void onTcpRequest(TcpMessage& request, TcpMessage& response);
    std::optional<std::vector<NodeInfo>> handleRegisterRequest(const NodeInfo& info);
    std::optional<RemoteAllocationDesc> handleAllocateRequest(const NodeId& requester, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags, bool enableRdma);
    std::optional<RemoteAllocationDesc> handleExportRequest(const NodeId& owner, uint64_t localAllocId, bool enableRdma, bool forceHostShadow);
    std::optional<Allocation> handleImportRequest(const NodeId& owner, const RemoteAllocationDesc& desc, VkBufferUsageFlags usage);
    std::optional<NetworkMigrationOperation> handleMigrateRequest(const RemoteAllocationDesc& source, uint64_t destinationAllocId, bool useRdma, uint64_t timeoutNs);
    void handleTensorAnnounce(const std::vector<uint8_t>& body);
    void heartbeatLoop();
    void mergeClusterView(const std::vector<NodeInfo>& view);
    void parseListenAddress(const std::string& listenAddress, std::string& outHost, uint16_t& outPort);
    static bool parseEndpoint(const std::string& endpoint, std::string& outHost, uint16_t& outPort);
};

// Whether same-process zero-copy import is allowed for a (srcVendor, dstVendor)
// PCI vendor pair. Cross-vendor pairs are refused on Linux by default
// (VVM_ALLOW_CROSSVENDOR_ZC=1 overrides) - see zcAllowedForPair() in
// multi_node_manager.cpp for rationale.
bool sameProcessZcAllowed(uint32_t srcVendorId, uint32_t dstVendorId);

} // namespace network
} // namespace vvm