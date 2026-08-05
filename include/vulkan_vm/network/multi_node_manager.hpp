#pragma once

#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/network/network_config.hpp"
#include <memory>
#include <vector>
#include <optional>
#include <future>
#include <string>
#include <functional>
#include <mutex>
#include <unordered_map>

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
        const std::vector<DeviceConfig>& localDevices,
        const PoolConfig& poolConfig,
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
    std::unique_ptr<ClusterClient> clusterClient_;
    std::unique_ptr<ClusterServer> clusterServer_;
    std::unique_ptr<RdmaTransport> rdmaTransport_;
    
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
    
    // Migration callback type
    using MigrationCallback = std::function<void(const NetworkMigrationOperation&)>;
    
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
    
    // Host-staged fallback
    std::optional<NetworkMigrationOperation> migrateHostStaged(
        const RemoteAllocationDesc& source,
        Allocation& destination,
        bool toHost,  // true = pull from remote, false = push to remote
        uint64_t timeoutNs);
};

} // namespace network
} // namespace vvm