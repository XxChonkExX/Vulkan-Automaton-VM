#include "vulkan_vm/network/multi_node_manager.hpp"
#include "vulkan_vm/network/cluster_client.hpp"
#include "vulkan_vm/network/cluster_server.hpp"
#include "vulkan_vm/network/rdma_transport.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <chrono>
#include <random>
#include <algorithm>

namespace vvm {
namespace network {

// ============================================================================
// MultiNodePoolManager Implementation
// ============================================================================

std::optional<MultiNodePoolManager> MultiNodePoolManager::create(
    const std::vector<DeviceConfig>& localDevices,
    const PoolConfig& poolConfig,
    const NetworkConfig& networkConfig) {
    
    if (localDevices.empty()) {
        VVM_LOG_ERROR("MultiNodePoolManager: no local devices provided");
        return std::nullopt;
    }
    
    if (!networkConfig.validate()) {
        VVM_LOG_ERROR("MultiNodePoolManager: invalid network config");
        return std::nullopt;
    }
    
    MultiNodePoolManager manager(localDevices, poolConfig, networkConfig);
    if (!manager.initialize()) {
        return std::nullopt;
    }
    
    return manager;
}

MultiNodePoolManager::MultiNodePoolManager(
    const std::vector<DeviceConfig>& localDevices,
    const PoolConfig& poolConfig,
    const NetworkConfig& networkConfig)
    : localDeviceConfigs_(localDevices)
    , poolConfig_(poolConfig)
    , networkConfig_(networkConfig) {
    
    // Generate local node ID
    localNodeId_.host = networkConfig_.advertiseAddress.empty() ? "localhost" : networkConfig_.advertiseAddress;
    localNodeId_.port = 50051;  // extract from listenAddress
    localNodeId_.nodeIndex = 0;
    
    // Generate UUID
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    localNodeId_.uuid = std::to_string(dis(gen));
    
    VVM_LOG_INFO("MultiNodePoolManager created for node: {}", localNodeId_.toString());
}

MultiNodePoolManager::MultiNodePoolManager(MultiNodePoolManager&& other) noexcept
    : clusterClient_(std::move(other.clusterClient_))
    , clusterServer_(std::move(other.clusterServer_))
    , rdmaTransport_(std::move(other.rdmaTransport_))
    , localPools_(std::move(other.localPools_))
    , localDeviceConfigs_(std::move(other.localDeviceConfigs_))
    , poolConfig_(std::move(other.poolConfig_))
    , networkConfig_(std::move(other.networkConfig_))
    , localNodeId_(std::move(other.localNodeId_))
    , activeMigrations_(std::move(other.activeMigrations_))
    , nextMigrationId_(other.nextMigrationId_)
    , clusterView_(std::move(other.clusterView_))
    , running_(other.running_)
    , networkStats_(other.networkStats_) {
    other.running_ = false;
}

MultiNodePoolManager& MultiNodePoolManager::operator=(MultiNodePoolManager&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        clusterClient_ = std::move(other.clusterClient_);
        clusterServer_ = std::move(other.clusterServer_);
        rdmaTransport_ = std::move(other.rdmaTransport_);
        localPools_ = std::move(other.localPools_);
        localDeviceConfigs_ = std::move(other.localDeviceConfigs_);
        poolConfig_ = std::move(other.poolConfig_);
        networkConfig_ = std::move(other.networkConfig_);
        localNodeId_ = std::move(other.localNodeId_);
        activeMigrations_ = std::move(other.activeMigrations_);
        nextMigrationId_ = other.nextMigrationId_;
        clusterView_ = std::move(other.clusterView_);
        running_ = other.running_;
        networkStats_ = other.networkStats_;
        
        other.running_ = false;
    }
    return *this;
}

MultiNodePoolManager::~MultiNodePoolManager() {
    cleanup();
}

bool MultiNodePoolManager::initialize() {
    // Create local pools for each GPU
    localPools_.reserve(localDeviceConfigs_.size());
    
    for (size_t i = 0; i < localDeviceConfigs_.size(); ++i) {
        auto pool = UnifiedMemoryPool::create(localDeviceConfigs_[i], poolConfig_);
        if (!pool) {
            VVM_LOG_ERROR("Failed to create local pool for device {}", i);
            return false;
        }
        localPools_.push_back(std::move(*pool));
        VVM_LOG_INFO("Created local pool {} for device {}", i, localDeviceConfigs_[i].physicalDevice);
    }
    
    // Create network components
    clusterClient_ = ClusterClient::create(networkConfig_);
    clusterServer_ = ClusterServer::create(networkConfig_, this);
    rdmaTransport_ = RdmaTransport::create(networkConfig_, 
                                           localDeviceConfigs_[0].physicalDevice,
                                           localDeviceConfigs_[0].device);
    
    if (!rdmaTransport_ || !rdmaTransport_->initialize()) {
        VVM_LOG_WARN("RDMA transport initialization failed, using host-staged fallback only");
        rdmaTransport_.reset();
    }
    
    // Set up server handlers
    if (clusterServer_) {
        clusterServer_->setAllocateHandler([this](const NodeId& requester,
                                                   VkDeviceSize size,
                                                   VkBufferUsageFlags usage,
                                                   VkMemoryPropertyFlags flags,
                                                   bool enableRdma) {
            return handleAllocateRequest(requester, size, usage, flags, enableRdma);
        });
        
        clusterServer_->setExportHandler([this](const NodeId& requester,
                                                 uint64_t localAllocId,
                                                 bool enableRdma,
                                                 bool forceHostShadow) {
            return handleExportRequest(requester, localAllocId, enableRdma, forceHostShadow);
        });
        
        clusterServer_->setImportHandler([this](const NodeId& requester,
                                                 const RemoteAllocationDesc& desc,
                                                 VkBufferUsageFlags usage) {
            return handleImportRequest(requester, desc, usage);
        });
        
        clusterServer_->setMigrateHandler([this](const NodeId& requester,
                                                  const RemoteAllocationDesc& source,
                                                  uint64_t destinationAllocId,
                                                  bool useRdma) {
            return handleMigrateRequest(requester, source, destinationAllocId, useRdma);
        });
        
        clusterServer_->setRegisterHandler([this](const NodeInfo& info) {
            return handleRegisterRequest(info);
        });
    }
    
    running_ = true;
    VVM_LOG_INFO("MultiNodePoolManager initialized with {} local pools", localPools_.size());
    return true;
}

void MultiNodePoolManager::cleanup() {
    if (running_) {
        stop();
        
        // Wait for active migrations
        {
            std::lock_guard<std::mutex> lock(migrationsMutex_);
            for (auto& [id, migration] : activeMigrations_) {
                waitMigration(migration.op);
            }
            activeMigrations_.clear();
        }
        
        // Shutdown network
        if (clusterServer_) clusterServer_->stop();
        if (clusterClient_) clusterClient_->disconnect();
        if (rdmaTransport_) rdmaTransport_->shutdown();
        
        // Destroy local pools
        localPools_.clear();
        
        running_ = false;
        VVM_LOG_INFO("MultiNodePoolManager cleaned up");
    }
}

bool MultiNodePoolManager::start() {
    if (!running_) return false;
    
    if (clusterServer_ && !clusterServer_->start()) {
        VVM_LOG_ERROR("Failed to start cluster server");
        return false;
    }
    
    if (clusterClient_) {
        std::string target = localNodeId_.host + ":" + std::to_string(localNodeId_.port);
        if (!clusterClient_->connect(target)) {
            VVM_LOG_WARN("Failed to connect to local cluster server");
        }
    }
    
    // Register with cluster
    if (!registerWithCluster()) {
        VVM_LOG_WARN("Failed to register with cluster");
    }
    
    VVM_LOG_INFO("MultiNodePoolManager started");
    return true;
}

void MultiNodePoolManager::stop() {
    if (!running_) return;
    
    leaveCluster();
    
    if (clusterServer_) clusterServer_->stop();
    if (clusterClient_) clusterClient_->disconnect();
    
    running_ = false;
    VVM_LOG_INFO("MultiNodePoolManager stopped");
}

bool MultiNodePoolManager::isRunning() const {
    return running_;
}

// ============================================================================
// Local allocation
// ============================================================================

std::optional<Allocation> MultiNodePoolManager::allocateLocal(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags flags,
    bool advertise) {
    
    // Use first pool by default (could add device selection logic)
    if (localPools_.empty()) return std::nullopt;
    
    auto alloc = localPools_[0].allocate(size, usage, flags);
    if (!alloc) return std::nullopt;
    
    if (advertise && clusterServer_) {
        // Export and register in cluster directory
        auto desc = exportForRemote(*alloc, true, false);
        if (desc) {
            VVM_LOG_DEBUG("Advertised allocation {} in cluster", alloc->deviceAddress);
        }
    }
    
    return alloc;
}

std::optional<Allocation> MultiNodePoolManager::allocateTensor(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    bool advertise) {
    
    return allocateLocal(size, usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 0, advertise);
}

void MultiNodePoolManager::deallocateLocal(Allocation&& alloc) {
    if (localPools_.empty()) return;
    
    // Unregister from RDMA if needed
    if (rdmaTransport_) {
        // TODO: track RDMA registrations per allocation
        // unregisterMemoryForRdma(alloc);
    }
    
    localPools_[0].deallocate(std::move(alloc));
}

// ============================================================================
// Remote allocation
// ============================================================================

std::future<RemoteAllocationDesc> MultiNodePoolManager::allocateRemoteAsync(
    const NodeId& target,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags flags,
    bool enableRdma) {
    
    if (!clusterClient_) {
        return std::async(std::launch::deferred, []() {
            return std::optional<RemoteAllocationDesc>{};
        });
    }
    
    return clusterClient_->allocateRemoteAsync(target, size, usage, flags, enableRdma);
}

std::optional<RemoteAllocationDesc> MultiNodePoolManager::allocateRemote(
    const NodeId& target,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags flags,
    bool enableRdma,
    uint64_t timeoutNs) {
    
    if (!clusterClient_) return std::nullopt;
    return clusterClient_->allocateRemote(target, size, usage, flags, enableRdma, timeoutNs);
}

// ============================================================================
// Export/Import
// ============================================================================

std::optional<RemoteAllocationDesc> MultiNodePoolManager::exportForRemote(
    const Allocation& alloc,
    bool enableRdma,
    bool forceHostShadow) {
    
    // First try local pool export
    auto desc = localPools_[0].exportMemory(alloc, 
        enableRdma ? ExternalHandleType::OpaqueFd : ExternalHandleType::OpaqueFd);
    
    if (!desc) {
        VVM_LOG_ERROR("Failed to export allocation memory");
        return std::nullopt;
    }
    
    RemoteAllocationDesc netDesc;
    netDesc.owner = localNodeId_;
    netDesc.size = alloc.size;
    netDesc.localAllocId = alloc.deviceAddress;  // use device address as ID
    netDesc.usageFlags = alloc.buffer != VK_NULL_HANDLE ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : 0;
    netDesc.memoryTypeIndex = alloc.blockIndex != UINT32_MAX ? alloc.blockIndex : 0;
    netDesc.dedicatedAllocation = alloc.isExternal;
    netDesc.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    // Try RDMA registration
    if (enableRdma && rdmaTransport_ && !forceHostShadow) {
        uint64_t rdmaAddr = 0;
        uint32_t rkey = 0;
        if (registerMemoryForRdma(alloc, rdmaAddr, rkey)) {
            netDesc.hasRdmaAddr = true;
            netDesc.rdmaAddr = rdmaAddr;
            netDesc.rkey = rkey;
            VVM_LOG_DEBUG("Registered allocation for GPU-direct RDMA: addr=0x{:x}, rkey=0x{:x}", rdmaAddr, rkey);
        }
    }
    
    // Fallback: host shadow
    if (!netDesc.hasRdmaAddr) {
        netDesc.hasHostShadow = true;
        netDesc.handleType = ExternalHandleType::OpaqueFd;
        VVM_LOG_DEBUG("Using host-staged fallback for export");
    }
    
    return netDesc;
}

std::optional<Allocation> MultiNodePoolManager::importRemote(
    const RemoteAllocationDesc& desc,
    VkBufferUsageFlags usage) {
    
    // If we have RDMA address, try GPU-direct import
    if (desc.canUseRdma() && rdmaTransport_) {
        // For GPU-direct, we need to register the remote memory locally
        // This is complex and vendor-specific - for now use host-staged
        VVM_LOG_DEBUG("GPU-direct import not fully implemented, using host-staged");
    }
    
    // Host-staged fallback: allocate local, pull data via host
    return createLocalAllocationForImport(desc, usage);
}

std::future<std::optional<Allocation>> MultiNodePoolManager::importRemoteAsync(
    const RemoteAllocationDesc& desc,
    VkBufferUsageFlags usage) {
    
    return std::async(std::launch::async, [this, desc, usage]() {
        return importRemote(desc, usage);
    });
}

// ============================================================================
// Migration
// ============================================================================

std::optional<NetworkMigrationOperation> MultiNodePoolManager::migrateFromRemote(
    const RemoteAllocationDesc& source,
    Allocation& destination,
    bool useRdma,
    uint64_t timeoutNs) {
    
    if (!source.canUseRdma() || !useRdma || !rdmaTransport_) {
        return migrateHostStaged(source, destination, true, timeoutNs);
    }
    
    // GPU-direct RDMA path
    // This is a placeholder - real implementation would use rdmaTransport_->rdmaRead()
    VVM_LOG_WARN("GPU-direct RDMA read not fully implemented, falling back to host-staged");
    return migrateHostStaged(source, destination, true, timeoutNs);
}

std::optional<NetworkMigrationOperation> MultiNodePoolManager::migrateToRemote(
    Allocation& source,
    const RemoteAllocationDesc& destination,
    bool useRdma,
    uint64_t timeoutNs) {
    
    if (!destination.canUseRdma() || !useRdma || !rdmaTransport_) {
        return migrateHostStaged(source, destination, false, timeoutNs);
    }
    
    // GPU-direct RDMA write path
    VVM_LOG_WARN("GPU-direct RDMA write not fully implemented, falling back to host-staged");
    return migrateHostStaged(source, destination, false, timeoutNs);
}

void MultiNodePoolManager::waitMigration(NetworkMigrationOperation& op) {
    if (op.completionFence != VK_NULL_HANDLE) {
        vkWaitForFences(localPools_[0].getDevice(), 1, &op.completionFence, VK_TRUE, UINT64_MAX);
    }
}

bool MultiNodePoolManager::pollMigration(NetworkMigrationOperation& op) {
    if (op.completionFence == VK_NULL_HANDLE) return true;
    return vkGetFenceStatus(localPools_[0].getDevice(), op.completionFence) == VK_SUCCESS;
}

void MultiNodePoolManager::migrateFromRemoteAsync(
    const RemoteAllocationDesc& source,
    Allocation& destination,
    MigrationCallback callback,
    bool useRdma) {
    
    auto op = migrateFromRemote(source, destination, useRdma, UINT64_MAX);
    if (op) {
        std::lock_guard<std::mutex> lock(migrationsMutex_);
        uint64_t id = nextMigrationId_++;
        activeMigrations_[id] = {std::move(*op), std::make_shared<MigrationCallback>(std::move(callback))};
        
        // Poll in background thread
        std::thread([this, id]() {
            auto it = activeMigrations_.find(id);
            if (it != activeMigrations_.end()) {
                while (!pollMigration(it->second.op)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                if (it->second.callback) {
                    (*it->second.callback)(it->second.op);
                }
                std::lock_guard<std::mutex> lock(migrationsMutex_);
                activeMigrations_.erase(id);
            }
        }).detach();
    }
}

// ============================================================================
// Cluster management
// ============================================================================

std::vector<NodeInfo> MultiNodePoolManager::getClusterView() const {
    std::lock_guard<std::mutex> lock(clusterViewMutex_);
    return clusterView_;
}

std::optional<NodeId> MultiNodePoolManager::findNodeByGpu(uint32_t vendorId, uint32_t deviceId) const {
    std::lock_guard<std::mutex> lock(clusterViewMutex_);
    for (const auto& node : clusterView_) {
        for (const auto& gpu : node.gpuDevices) {
            // Parse GPU string for vendor/device - simplified
            if (gpu.find(std::to_string(vendorId)) != std::string::npos) {
                return node.id;
            }
        }
    }
    return std::nullopt;
}

std::optional<NodeId> MultiNodePoolManager::findNodeWithCapability(bool requireRdma, bool requireGpuDirect) const {
    std::lock_guard<std::mutex> lock(clusterViewMutex_);
    for (const auto& node : clusterView_) {
        if ((!requireRdma || node.rdmaCapable) && (!requireGpuDirect || node.gpuDirectCapable)) {
            return node.id;
        }
    }
    return std::nullopt;
}

bool MultiNodePoolManager::registerWithCluster() {
    if (!clusterClient_) return false;
    
    NodeInfo info;
    info.id = localNodeId_;
    info.nicName = networkConfig_.nicName;
    info.rdmaCapable = rdmaTransport_ != nullptr;
    info.gpuDirectCapable = rdmaTransport_ && rdmaTransport_->supportsGpuDirect();
    info.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    // Populate GPU devices from local pools
    for (const auto& pool : localPools_) {
        auto memInfo = pool.getDeviceMemoryInfo();
        info.gpuDevices.push_back("GPU_" + std::to_string(memInfo.memProps.memoryHeapCount));
    }
    
    auto result = clusterClient_->registerNode(info, networkConfig_.rpcTimeout.count() * 1'000'000);
    if (result) {
        std::lock_guard<std::mutex> lock(clusterViewMutex_);
        clusterView_ = *result;
        VVM_LOG_INFO("Registered with cluster, {} nodes visible", clusterView_.size());
        return true;
    }
    
    return false;
}

void MultiNodePoolManager::leaveCluster() {
    // TODO: Implement leave cluster RPC
    VVM_LOG_INFO("Leaving cluster");
}

const NodeId& MultiNodePoolManager::getLocalNodeId() const {
    return localNodeId_;
}

const NetworkConfig& MultiNodePoolManager::getNetworkConfig() const {
    return networkConfig_;
}

// ============================================================================
// Stats
// ============================================================================

MultiNodePoolManager::NetworkStats MultiNodePoolManager::getNetworkStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return networkStats_;
}

void MultiNodePoolManager::resetNetworkStats() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    networkStats_ = NetworkStats{};
}

// ============================================================================
// Internal helpers
// ============================================================================

std::optional<Allocation> MultiNodePoolManager::createLocalAllocationForImport(
    const RemoteAllocationDesc& desc,
    VkBufferUsageFlags usage) {
    
    // Allocate local memory to receive data
    auto alloc = localPools_[0].allocate(desc.size, usage, 0);
    if (!alloc) {
        VVM_LOG_ERROR("Failed to allocate local memory for import");
        return std::nullopt;
    }
    
    // If we have external handle, import it
    if (!desc.externalHandle.empty()) {
        // Import via external handle
        ExternalMemoryInfo extInfo;
        extInfo.type = desc.handleType;
        extInfo.size = desc.size;
        extInfo.memoryTypeIndex = desc.memoryTypeIndex;
        
        if (desc.handleType == ExternalHandleType::OpaqueFd) {
            // Copy fd from handle data
            if (desc.externalHandle.size() >= sizeof(int)) {
                extInfo.fd = *reinterpret_cast<const int*>(desc.externalHandle.data());
            }
        }
        
        auto imported = localPools_[0].importMemory(extInfo, usage);
        if (imported) {
            return imported;
        }
    }
    
    return alloc;
}

bool MultiNodePoolManager::registerMemoryForRdma(const Allocation& alloc, uint64_t& outRdmaAddr, uint32_t& outRkey) {
    if (!rdmaTransport_) return false;
    
    // This would call into the RDMA transport to register the VkDeviceMemory
    // For NVIDIA, this uses VK_NV_external_memory_rdma + ibv_reg_mr
    // Placeholder for now
    VVM_LOG_DEBUG("RDMA registration placeholder for allocation");
    return false;
}

void MultiNodePoolManager::unregisterMemoryForRdma(const Allocation& alloc) {
    // TODO: Unregister from RDMA
}

std::optional<NetworkMigrationOperation> MultiNodePoolManager::migrateHostStaged(
    const RemoteAllocationDesc& source,
    Allocation& destination,
    bool toHost,  // true = pull from remote, false = push to remote
    uint64_t timeoutNs) {
    
    // This uses the existing offload/reload mechanism
    // Pull from remote: remote.offloadToHost -> transfer host buffer -> local.reloadToDevice
    // Push to remote: local.offloadToHost -> transfer host buffer -> remote.reloadToDevice
    
    NetworkMigrationOperation op;
    op.operationId = nextMigrationId_++;
    op.source = source;
    op.destinationAllocId = destination.deviceAddress;
    op.useRdma = false;
    op.timeoutNs = timeoutNs;
    op.completed = false;
    
    VVM_LOG_INFO("Host-staged migration: {} bytes, toHost={}", source.size, toHost);
    
    // For now, return a placeholder operation
    // Real implementation would coordinate with remote node via RPC
    op.completed = true;  // placeholder
    op.bytesTransferred = source.size;
    
    // Update stats
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        networkStats_.bytesSentHostStaged += source.size;
        networkStats_.bytesReceivedHostStaged += source.size;
        networkStats_.completedMigrations++;
    }
    
    return op;
}

// ============================================================================
// RPC Handlers (called by ClusterServer)
// ============================================================================

std::optional<RemoteAllocationDesc> MultiNodePoolManager::handleAllocateRequest(
    const NodeId& requester,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags flags,
    bool enableRdma) {
    
    VVM_LOG_INFO("Allocate request from {}: {} bytes", requester.toString(), size);
    
    auto alloc = allocateLocal(size, usage, flags, false);
    if (!alloc) return std::nullopt;
    
    return exportForRemote(*alloc, enableRdma, false);
}

std::optional<RemoteAllocationDesc> MultiNodePoolManager::handleExportRequest(
    const NodeId& requester,
    uint64_t localAllocId,
    bool enableRdma,
    bool forceHostShadow) {
    
    VVM_LOG_INFO("Export request from {} for alloc {}", requester.toString(), localAllocId);
    
    // Find allocation by device address
    // This is simplified - real implementation would track allocations
    return std::nullopt;
}

std::optional<Allocation> MultiNodePoolManager::handleImportRequest(
    const NodeId& requester,
    const RemoteAllocationDesc& desc,
    VkBufferUsageFlags usage) {
    
    VVM_LOG_INFO("Import request from {}: {} bytes", requester.toString(), desc.size);
    
    return importRemote(desc, usage);
}

std::optional<NetworkMigrationOperation> MultiNodePoolManager::handleMigrateRequest(
    const NodeId& requester,
    const RemoteAllocationDesc& source,
    uint64_t destinationAllocId,
    bool useRdma) {
    
    VVM_LOG_INFO("Migrate request from {}: {} bytes", requester.toString(), source.size);
    
    // Find destination allocation
    // This is simplified
    return std::nullopt;
}

std::optional<std::vector<NodeInfo>> MultiNodePoolManager::handleRegisterRequest(
    const NodeInfo& info) {
    
    VVM_LOG_INFO("Node registration from: {}", info.id.toString());
    
    std::lock_guard<std::mutex> lock(clusterViewMutex_);
    clusterView_.push_back(info);
    
    return clusterView_;
}

} // namespace network
} // namespace vvm