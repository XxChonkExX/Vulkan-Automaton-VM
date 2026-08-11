#include "vulkan_vm/network/multi_node_manager.hpp"
#include "vulkan_vm/network/tcp_transport.hpp"
#include "vulkan_vm/network/network_types.hpp"
#include "vulkan_vm/cross_gpu/external_memory.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#if defined(VVM_PLATFORM_LINUX)
#include <ifaddrs.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#endif

// Helper to auto-detect primary LAN IP (Linux)
#if defined(VVM_PLATFORM_LINUX)
static std::string getPrimaryLanIp() {
    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) != 0) return "127.0.0.1";
    
    std::string bestIp = "127.0.0.1";
    for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
        if (!(ifa->ifa_flags & IFF_UP) || (ifa->ifa_flags & IFF_LOOPBACK)) continue;
        
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr, ip, sizeof(ip));
        
        // Prefer non-link-local addresses
        struct in_addr addr;
        inet_pton(AF_INET, ip, &addr);
        uint32_t addr32 = ntohl(addr.s_addr);
        
        // Skip link-local (169.254.x.x) and loopback
        if ((addr32 & 0xFFFF0000) == 0xA9FE0000) continue;
        if ((addr32 & 0xFF000000) == 0x7F000000) continue;
        
        // Return first valid LAN IP
        freeifaddrs(ifaddr);
        return ip;
    }
    freeifaddrs(ifaddr);
    return bestIp;
}
#elif defined(VVM_PLATFORM_WINDOWS)
static std::string getPrimaryLanIp() {
    // Windows: use GetAdaptersAddresses or fallback
    return "127.0.0.1";  // TODO: implement Windows IP detection
}
#else
static std::string getPrimaryLanIp() {
    return "127.0.0.1";
}
#endif

namespace vvm {
namespace network {

namespace {

constexpr VkDeviceSize kTransferChunk = 4ull * 1024 * 1024;  // 4MB chunks over TCP

// Process-local registry of exported external-memory handles that are still
// alive. OS handles (FDs/HANDLEs) are process-relative and CANNOT be shipped
// over TCP, so a same-process peer import consumes the live handle from here,
// while cross-process/machine peers fall back to host-staged copies.
std::unordered_map<std::string, vvm::ExternalMemoryInfo> g_pendingExports;
std::mutex g_pendingExportsMutex;

std::string pendingKey(const NodeId& owner, uint64_t allocId) {
    return owner.host + ":" + std::to_string(owner.port) + ":" + std::to_string(allocId);
}

// Build a response message with the given flag.
TcpMessage makeResponse(uint32_t type, uint32_t flags) {
    TcpMessage resp;
    resp.type = type;
    resp.flags = flags;
    return resp;
}

}  // namespace

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

    // Parse listen address into host + port
    std::string listenHost;
    uint16_t listenPort = 0;
    parseListenAddress(networkConfig_.listenAddress, listenHost, listenPort);
    tcpPort_ = listenPort;
    // Use advertiseAddress if set, otherwise auto-detect LAN IP
    localHost_ = networkConfig_.advertiseAddress.empty() ? getPrimaryLanIp() : networkConfig_.advertiseAddress;

    // Parse seed nodes into host:port endpoints
    for (const auto& seed : networkConfig_.seedNodes) {
        std::string host;
        uint16_t port = 0;
        if (parseEndpoint(seed, host, port)) {
            seedEndpoints_.emplace_back(host, port);
        } else {
            VVM_LOG_WARN("Ignoring invalid seed node: {}", seed);
        }
    }

    // Build local node identity
    localNodeId_.host = localHost_;
    localNodeId_.port = tcpPort_;
    localNodeId_.nodeIndex = 0;

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
    , tcpTransport_(std::move(other.tcpTransport_))
    , tcpPort_(other.tcpPort_)
    , localHost_(std::move(other.localHost_))
    , seedEndpoints_(std::move(other.seedEndpoints_))
    , copyCmdPool_(other.copyCmdPool_)
    , transferQueue_(other.transferQueue_)
    , transferQueueFamily_(other.transferQueueFamily_)
    , remoteAllocs_(std::move(other.remoteAllocs_))
    , nextAllocId_(other.nextAllocId_)
    , peerConns_(std::move(other.peerConns_))
    , heartbeatThread_(std::move(other.heartbeatThread_))
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
    other.copyCmdPool_ = VK_NULL_HANDLE;
    other.transferQueue_ = VK_NULL_HANDLE;
    other.transferQueueFamily_ = UINT32_MAX;
}

MultiNodePoolManager& MultiNodePoolManager::operator=(MultiNodePoolManager&& other) noexcept {
    if (this != &other) {
        cleanup();

        clusterClient_ = std::move(other.clusterClient_);
        clusterServer_ = std::move(other.clusterServer_);
        rdmaTransport_ = std::move(other.rdmaTransport_);
        tcpTransport_ = std::move(other.tcpTransport_);
        tcpPort_ = other.tcpPort_;
        localHost_ = std::move(other.localHost_);
        seedEndpoints_ = std::move(other.seedEndpoints_);
        copyCmdPool_ = other.copyCmdPool_;
        transferQueue_ = other.transferQueue_;
        transferQueueFamily_ = other.transferQueueFamily_;
        remoteAllocs_ = std::move(other.remoteAllocs_);
        nextAllocId_ = other.nextAllocId_;
        peerConns_ = std::move(other.peerConns_);
        heartbeatThread_ = std::move(other.heartbeatThread_);
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
        other.copyCmdPool_ = VK_NULL_HANDLE;
        other.transferQueue_ = VK_NULL_HANDLE;
        other.transferQueueFamily_ = UINT32_MAX;
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

    if (!initCopyEngine()) {
        VVM_LOG_WARN("Copy engine unavailable; migrations will fail");
    }

#if defined(VVM_NETWORK_HAS_GRPC)
    clusterClient_ = ClusterClient::create(networkConfig_);
    clusterServer_ = ClusterServer::create(networkConfig_, this);
#endif

#if defined(VVM_NETWORK_HAS_VERBS)
    rdmaTransport_ = RdmaTransport::create(networkConfig_,
                                           localDeviceConfigs_[0].physicalDevice,
                                           localDeviceConfigs_[0].device);
    if (!rdmaTransport_ || !rdmaTransport_->initialize()) {
        VVM_LOG_WARN("RDMA transport initialization failed, using host-staged fallback only");
        rdmaTransport_.reset();
    }
#endif

    VVM_LOG_INFO("MultiNodePoolManager initialized with {} local pools", localPools_.size());
    return true;
}

void MultiNodePoolManager::cleanup() {
    if (running_) {
        stop();
    }

    // Wait for active migrations
    {
        std::lock_guard<std::mutex> lock(migrationsMutex_);
        for (auto& [id, migration] : activeMigrations_) {
            waitMigration(migration.op);
        }
        activeMigrations_.clear();
    }

    // Destroy copy engine
    if (copyCmdPool_ != VK_NULL_HANDLE && !localPools_.empty()) {
        vkDestroyCommandPool(localPools_[0].getDevice(), copyCmdPool_, nullptr);
        copyCmdPool_ = VK_NULL_HANDLE;
    }

#if defined(VVM_NETWORK_HAS_GRPC)
    if (clusterServer_) clusterServer_->stop();
    if (clusterClient_) clusterClient_->disconnect();
#endif
#if defined(VVM_NETWORK_HAS_VERBS)
    if (rdmaTransport_) rdmaTransport_->shutdown();
#endif

    // Destroy local pools
    localPools_.clear();

    VVM_LOG_INFO("MultiNodePoolManager cleaned up");
}

bool MultiNodePoolManager::start() {
    if (running_) return true;

    // Start TCP control/data plane server
    tcpTransport_ = std::make_unique<TcpTransport>();
    std::string listenHost;
    uint16_t listenPort = 0;
    parseListenAddress(networkConfig_.listenAddress, listenHost, listenPort);

    auto handler = [this](TcpMessage& request, TcpMessage& response) {
        onTcpRequest(request, response);
    };

    if (!tcpTransport_->start(listenHost, listenPort, std::move(handler))) {
        VVM_LOG_ERROR("Failed to start TCP server on {}:{}", listenHost, listenPort);
        tcpTransport_.reset();
        return false;
    }
    tcpPort_ = tcpTransport_->getBoundPort();
    localNodeId_.port = tcpPort_;

    // Start heartbeat thread
    stopHeartbeat_ = false;
    heartbeatThread_ = std::thread([this]() { heartbeatLoop(); });

    // Register with cluster
    if (!registerWithCluster()) {
        VVM_LOG_WARN("Failed to register with cluster");
    }

    running_ = true;
    VVM_LOG_INFO("MultiNodePoolManager started (node {}, port {})", localNodeId_.host, tcpPort_);
    return true;
}

void MultiNodePoolManager::stop() {
    if (!running_) return;

    leaveCluster();

    stopHeartbeat_ = true;
    if (heartbeatThread_.joinable()) heartbeatThread_.join();

    if (tcpTransport_) {
        tcpTransport_->stop();
        tcpTransport_.reset();
    }

    {
        std::lock_guard<std::mutex> lock(connsMutex_);
        peerConns_.clear();
    }

    // Release any exported handles this node still owns (closes FDs/HANDLEs).
    {
        std::lock_guard<std::mutex> lock(g_pendingExportsMutex);
        for (auto it = g_pendingExports.begin(); it != g_pendingExports.end();) {
            if (it->first.rfind(localNodeId_.host + ":" + std::to_string(localNodeId_.port) + ":", 0) == 0) {
                it = g_pendingExports.erase(it);
            } else {
                ++it;
            }
        }
    }

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

    if (localPools_.empty()) return std::nullopt;

    auto alloc = localPools_[0].allocate(size, usage, flags);
    if (!alloc) return std::nullopt;

    if (advertise) {
        auto desc = exportForRemote(*alloc, true, false);
        if (!desc) {
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

    // Remove from remote-visible registry if present
    if (auto id = findAllocIdByBuffer(alloc.buffer)) {
        unregisterAllocation(*id);
    }

    localPools_[0].deallocate(std::move(alloc));
}

bool MultiNodePoolManager::deallocateRemote(const RemoteAllocationDesc& desc) {
    if (desc.owner != localNodeId_) {
        // Ask the owner to free it
        auto conn = getPeerConnection(desc.owner.host, desc.owner.port);
        if (conn == 0) return false;
        TcpMessage req;
        req.type = MsgDeallocate;
        req.flags = TcpFlagsRequest;
        std::vector<uint8_t> body;
        detail::putU64(body, desc.localAllocId);
        req.body = std::move(body);
        auto resp = tcpTransport_->request(conn, req);
        return resp.has_value() && resp->flags != TcpFlagsError;
    }
    // Local allocation
    if (auto alloc = findAllocation(desc.localAllocId)) {
        unregisterAllocation(desc.localAllocId);
        localPools_[0].deallocate(std::move(*alloc));
        return true;
    }
    return false;
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

    return std::async(std::launch::async, [this, target, size, usage, flags, enableRdma]() {
        auto desc = allocateRemote(target, size, usage, flags, enableRdma, UINT64_MAX);
        return desc ? *desc : RemoteAllocationDesc{};
    });
}

std::optional<RemoteAllocationDesc> MultiNodePoolManager::allocateRemote(
    const NodeId& target,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags flags,
    bool enableRdma,
    uint64_t timeoutNs) {

    (void)timeoutNs;

    auto conn = getPeerConnection(target.host, target.port);
    if (conn == 0) {
        VVM_LOG_ERROR("allocateRemote: no connection to {}", target.toString());
        return std::nullopt;
    }

    std::vector<uint8_t> body;
    detail::putU64(body, size);
    detail::putU32(body, usage);
    detail::putU32(body, flags);
    detail::putU8(body, enableRdma ? 1 : 0);

    TcpMessage req;
    req.type = MsgAllocate;
    req.flags = TcpFlagsRequest;
    req.body = std::move(body);

    auto resp = tcpTransport_->request(conn, req);
    if (!resp || resp->flags == TcpFlagsError) {
        VVM_LOG_ERROR("allocateRemote: request failed for {}", target.toString());
        return std::nullopt;
    }

    const uint8_t* p = resp->body.data();
    const uint8_t* end = p + resp->body.size();
    uint8_t success = 0;
    if (!detail::getU8(p, end, success) || success == 0) {
        VVM_LOG_ERROR("allocateRemote: remote node refused allocation");
        return std::nullopt;
    }

    RemoteAllocationDesc desc;
    if (!deserializeAllocationDesc(p, end, desc)) {
        VVM_LOG_ERROR("allocateRemote: failed to parse response");
        return std::nullopt;
    }

    VVM_LOG_INFO("allocateRemote: {} bytes allocated on {}", size, target.toString());
    return desc;
}

// ============================================================================
// Export/Import
// ============================================================================

std::optional<RemoteAllocationDesc> MultiNodePoolManager::exportForRemote(
    const Allocation& alloc,
    bool enableRdma,
    bool forceHostShadow) {

    if (localPools_.empty()) return std::nullopt;

    RemoteAllocationDesc netDesc;
    netDesc.owner = localNodeId_;
    netDesc.size = alloc.size;
    netDesc.usageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    netDesc.memoryTypeIndex = 0;
    netDesc.dedicatedAllocation = false;
    netDesc.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    netDesc.allocationName = alloc.buffer != VK_NULL_HANDLE ? "local_export" : "";

    // GPU-direct path (only when verbs + GPUDirect are available)
#if defined(VVM_NETWORK_HAS_VERBS)
    if (enableRdma && rdmaTransport_ && !forceHostShadow) {
        uint64_t rdmaAddr = 0;
        uint32_t rkey = 0;
        if (registerMemoryForRdma(alloc, rdmaAddr, rkey)) {
            netDesc.hasRdmaAddr = true;
            netDesc.rdmaAddr = rdmaAddr;
            netDesc.rkey = rkey;
        }
    }
#else
    (void)enableRdma;
    (void)forceHostShadow;
#endif

    // Host-staged fallback
    std::optional<Allocation> promoted;
    std::optional<ExternalMemoryInfo> exportInfo;
    if (!netDesc.hasRdmaAddr) {
        netDesc.hasHostShadow = true;

        // Cross-GPU export requires a dedicated allocation. If the source is
        // sub-allocated from a shared block, promote it to a dedicated
        // exportable copy (copying the data over) so we can hand out a real
        // external handle.
        const Allocation* exportSrc = &alloc;
        if (alloc.blockIndex != UINT32_MAX) {
            promoted = localPools_[0].allocateDedicatedExportable(
                alloc.size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                alloc.memoryFlags);
            if (!promoted) {
                VVM_LOG_ERROR("exportForRemote: failed to promote sub-allocated allocation to dedicated");
                return std::nullopt;
            }
            bool copied = false;
            if (alloc.hostPtr && promoted->hostPtr) {
                std::memcpy(promoted->hostPtr, alloc.hostPtr, static_cast<size_t>(alloc.size));
                copied = true;
            } else {
                copied = runCopy(alloc.buffer, promoted->buffer, 0, 0, alloc.size);
            }
            if (!copied) {
                VVM_LOG_ERROR("exportForRemote: failed to copy data into dedicated allocation");
                return std::nullopt;
            }
            exportSrc = &*promoted;
            netDesc.dedicatedAllocation = true;
        }

        // Export the actual handle from the export source. The ExternalMemoryInfo
        // OWNS the handle; we keep it alive in a process-local registry so a
        // same-process import can consume it (cross-process hosts stage data).
        vvm::ExternalHandleType exportType = vvm::ExternalHandleType::None;
        #ifdef VVM_PLATFORM_LINUX
        {
            // Determine best export handle type based on GPU vendor
            VkPhysicalDevice physDev = localPools_[0].getPhysicalDevice();
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(physDev, &props);
            
            // AMD/Intel on Linux: prefer DMA-BUF for better cross-vendor compatibility
            // NVIDIA: OPAQUE_FD is fine
            if (props.vendorID == 0x1002 || props.vendorID == 0x8086) {
                exportType = vvm::ExternalHandleType::DmaBuf;
            } else {
                exportType = vvm::ExternalHandleType::OpaqueFd;
            }
        }
        exportInfo = localPools_[0].exportMemory(*exportSrc, exportType);
        netDesc.handleType = exportInfo ? exportType : netDesc.handleType;
        #elif defined(VVM_PLATFORM_WINDOWS)
        exportInfo = localPools_[0].exportMemory(*exportSrc, vvm::ExternalHandleType::OpaqueWin32);
        netDesc.handleType = exportInfo ? vvm::ExternalHandleType::OpaqueWin32
                                        : netDesc.handleType;
        #endif
        if (!exportInfo) {
            VVM_LOG_ERROR("exportForRemote: failed to export handle for allocation");
            return std::nullopt;
        }
        // NOTE: the raw handle value must NOT be copied into netDesc.externalHandle.
        // OS handles are process-relative and TCP cannot transfer them. Instead
        // the handle lives in g_pendingExports (below) keyed by this node + allocId,
        // so a same-process peer import can consume it directly.
    }

    // Register the allocation (or its promoted dedicated copy) so peers can
    // address it and remote read-back sees the exported data.
    uint64_t allocId = 0;
    if (promoted) {
        allocId = registerAllocation(std::move(*promoted));
    } else if (auto existing = findAllocIdByBuffer(alloc.buffer)) {
        allocId = *existing;
    } else {
        allocId = registerAllocation(Allocation(alloc));  // copy for registry
    }
    netDesc.localAllocId = allocId;

    // Keep the exported handle alive so a same-process peer can import it
    // zero-copy. Ownership moves to the importing peer (or is closed when the
    // export is deallocated / the manager stops).
    if (exportInfo) {
        std::lock_guard<std::mutex> lock(g_pendingExportsMutex);
        g_pendingExports[pendingKey(localNodeId_, allocId)] = std::move(*exportInfo);
    }

    return netDesc;
}

std::optional<Allocation> MultiNodePoolManager::importRemote(
    const RemoteAllocationDesc& desc,
    VkBufferUsageFlags usage) {

#if defined(VVM_NETWORK_HAS_VERBS)
    if (desc.canUseRdma() && rdmaTransport_) {
        VVM_LOG_DEBUG("GPU-direct import not fully implemented, using host-staged");
    }
#endif

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

#if defined(VVM_NETWORK_HAS_VERBS)
    if (source.canUseRdma() && useRdma && rdmaTransport_) {
        VVM_LOG_WARN("GPU-direct RDMA read not implemented, using host-staged path");
    }
#else
    (void)useRdma;
#endif

    if (source.owner == localNodeId_) {
        VVM_LOG_ERROR("migrateFromRemote: source is a local allocation");
        return std::nullopt;
    }

    auto conn = getPeerConnection(source.owner.host, source.owner.port);
    if (conn == 0) {
        VVM_LOG_ERROR("migrateFromRemote: no connection to {}", source.owner.toString());
        return std::nullopt;
    }

    // Stage received bytes into a host-visible buffer, then copy to device
    auto staging = createStaging(source.size);
    if (!staging) {
        VVM_LOG_ERROR("migrateFromRemote: failed to allocate staging buffer");
        return std::nullopt;
    }

    std::vector<uint8_t> body;
    detail::putU64(body, source.localAllocId);
    detail::putU64(body, 0);  // srcOffset within allocation
    detail::putU64(body, source.size);

    TcpMessage req;
    req.type = MsgMigratePull;
    req.flags = TcpFlagsRequest;
    req.body = std::move(body);

    // Slice mode: stream the pulled data straight into the staging buffer.
    StreamIO streamIO;
    streamIO.readBuffer = staging->hostPtr;
    streamIO.readLen = source.size;

    auto resp = tcpTransport_->request(conn, req, &streamIO);
    if (!resp || resp->flags == TcpFlagsError) {
        VVM_LOG_ERROR("migrateFromRemote: pull request failed for {} bytes", source.size);
        return std::nullopt;
    }

    if (resp->streamLen != source.size && resp->stream.size() != source.size) {
        VVM_LOG_ERROR("migrateFromRemote: expected {} bytes, got {}/{}", source.size,
                      resp->stream.size(), resp->streamLen);
        return std::nullopt;
    }

    if (!copyHostToDevice(*staging, destination, 0, source.size)) {
        VVM_LOG_ERROR("migrateFromRemote: staging -> device copy failed");
        return std::nullopt;
    }

    NetworkMigrationOperation op;
    op.operationId = nextMigrationId_++;
    op.source = source;
    op.destinationAllocId = destination.deviceAddress;
    op.useRdma = false;
    op.timeoutNs = timeoutNs;
    op.bytesTransferred = source.size;
    op.completed = true;

    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        networkStats_.bytesReceivedHostStaged += source.size;
        networkStats_.completedMigrations++;
    }

    VVM_LOG_INFO("migrateFromRemote: pulled {} bytes from {}", source.size, source.owner.toString());
    return op;
}

std::optional<NetworkMigrationOperation> MultiNodePoolManager::migrateToRemote(
    Allocation& source,
    const RemoteAllocationDesc& destination,
    bool useRdma,
    uint64_t timeoutNs) {

#if defined(VVM_NETWORK_HAS_VERBS)
    if (destination.canUseRdma() && useRdma && rdmaTransport_) {
        VVM_LOG_WARN("GPU-direct RDMA write not implemented, using host-staged path");
    }
#else
    (void)useRdma;
#endif

    if (destination.owner == localNodeId_) {
        VVM_LOG_ERROR("migrateToRemote: destination is a local allocation");
        return std::nullopt;
    }

    // Stage device memory into a host-visible buffer
    auto staging = createStaging(source.size);
    if (!staging) {
        VVM_LOG_ERROR("migrateToRemote: failed to allocate staging buffer");
        return std::nullopt;
    }

    if (!copyDeviceToHost(source, 0, *staging, source.size)) {
        VVM_LOG_ERROR("migrateToRemote: device -> staging copy failed");
        return std::nullopt;
    }

    std::vector<uint8_t> body;
    detail::putU64(body, destination.localAllocId);
    detail::putU64(body, destination.size);

    TcpMessage req;
    req.type = MsgMigratePush;
    req.flags = TcpFlagsRequest;
    req.body = std::move(body);

    // Slice mode: stream the staged data out of the staging buffer directly.
    StreamIO streamIO;
    streamIO.writeBuffer = staging->hostPtr;
    streamIO.writeLen = source.size;

    auto conn = getPeerConnection(destination.owner.host, destination.owner.port);
    if (conn == 0) {
        VVM_LOG_ERROR("migrateToRemote: no connection to {}", destination.owner.toString());
        return std::nullopt;
    }

    auto resp = tcpTransport_->request(conn, req, &streamIO);
    if (!resp || resp->flags == TcpFlagsError) {
        VVM_LOG_ERROR("migrateToRemote: push request failed for {} bytes", source.size);
        return std::nullopt;
    }

    NetworkMigrationOperation op;
    op.operationId = nextMigrationId_++;
    op.source = destination;
    op.destinationAllocId = source.deviceAddress;
    op.useRdma = false;
    op.timeoutNs = timeoutNs;
    op.bytesTransferred = source.size;
    op.completed = true;

    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        networkStats_.bytesSentHostStaged += source.size;
        networkStats_.completedMigrations++;
    }

    VVM_LOG_INFO("migrateToRemote: pushed {} bytes to {}", source.size, destination.owner.toString());
    return op;
}

void MultiNodePoolManager::waitMigration(NetworkMigrationOperation& op) {
    if (op.completionFence != VK_NULL_HANDLE) {
        vkWaitForFences(localPools_[0].getDevice(), 1, &op.completionFence, VK_TRUE, UINT64_MAX);
    }
    // Host-staged operations complete synchronously; nothing else to wait for.
}

bool MultiNodePoolManager::pollMigration(NetworkMigrationOperation& op) {
    if (op.completionFence != VK_NULL_HANDLE) {
        return vkGetFenceStatus(localPools_[0].getDevice(), op.completionFence) == VK_SUCCESS;
    }
    return op.completed;
}

void MultiNodePoolManager::migrateFromRemoteAsync(
    const RemoteAllocationDesc& source,
    Allocation& destination,
    MigrationCallback callback,
    bool useRdma) {

    auto op = migrateFromRemote(source, destination, useRdma, UINT64_MAX);
    if (!op) {
        if (callback) {
            NetworkMigrationOperation failed;
            failed.completed = false;
            failed.errorMessage = "migrateFromRemote failed";
            failed.operationId = nextMigrationId_++;
            callback(failed);
        }
        return;
    }

    std::lock_guard<std::mutex> lock(migrationsMutex_);
    uint64_t id = nextMigrationId_++;
    activeMigrations_[id] = {std::move(*op), std::make_shared<MigrationCallback>(std::move(callback))};

    // Host-staged ops are already complete; invoke the callback immediately.
    auto it = activeMigrations_.find(id);
    if (it != activeMigrations_.end()) {
        if (it->second.callback) {
            (*it->second.callback)(it->second.op);
        }
        activeMigrations_.erase(id);
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
            if (gpu.find(std::to_string(vendorId)) != std::string::npos) {
                return node.id;
            }
        }
    }
    (void)deviceId;
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
    if (seedEndpoints_.empty()) {
        // No peers to contact: this node boots the cluster itself.
        VVM_LOG_INFO("registerWithCluster: bootstrap node (no seeds), cluster starts with self");
        NodeInfo self;
        self.id = localNodeId_;
        self.nicName = networkConfig_.nicName;
        self.rdmaCapable = false;
        self.gpuDirectCapable = false;
        self.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        for (const auto& pool : localPools_) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(pool.getPhysicalDevice(), &props);
            self.gpuDevices.push_back(props.deviceName);
        }
        {
            std::lock_guard<std::mutex> lock(clusterViewMutex_);
            clusterView_.clear();
            clusterView_.push_back(self);
        }
        return true;
    }

    NodeInfo info;
    info.id = localNodeId_;
    info.nicName = networkConfig_.nicName;
    info.rdmaCapable = false;
    info.gpuDirectCapable = false;
    info.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    for (const auto& pool : localPools_) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pool.getPhysicalDevice(), &props);
        info.gpuDevices.push_back(props.deviceName);
    }

    auto body = serializeNodeInfo(info);

    bool anySuccess = false;
    for (const auto& [host, port] : seedEndpoints_) {
        auto conn = getPeerConnection(host, port);
        if (conn == 0) {
            VVM_LOG_WARN("registerWithCluster: cannot reach seed {}:{}", host, port);
            continue;
        }

        TcpMessage req;
        req.type = MsgRegisterNode;
        req.flags = TcpFlagsRequest;
        req.body = body;

        auto resp = tcpTransport_->request(conn, req);
        if (!resp || resp->flags == TcpFlagsError) {
            VVM_LOG_WARN("registerWithCluster: registration rejected by {}:{}", host, port);
            continue;
        }

        std::vector<NodeInfo> view;
        if (deserializeNodeList(resp->body, view)) {
            mergeClusterView(view);
            anySuccess = true;
        }
    }

    if (anySuccess) {
        VVM_LOG_INFO("Registered with cluster, {} nodes visible", clusterView_.size());
    }
    return anySuccess;
}

void MultiNodePoolManager::leaveCluster() {
    VVM_LOG_INFO("Leaving cluster");
std::lock_guard<std::mutex> lock(clusterViewMutex_);
    clusterView_.clear();
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
// TCP helpers
// ============================================================================

bool MultiNodePoolManager::initCopyEngine() {
    if (localPools_.empty()) return false;

    const auto& cfg = localPools_[0].getDeviceConfig();
    transferQueue_ = cfg.transferQueue;
    transferQueueFamily_ = cfg.transferQueueFamily;

    if (transferQueue_ == VK_NULL_HANDLE || transferQueueFamily_ == UINT32_MAX) {
        transferQueue_ = cfg.computeQueue;
        transferQueueFamily_ = cfg.computeQueueFamily;
    }
    if (transferQueue_ == VK_NULL_HANDLE || transferQueueFamily_ == UINT32_MAX) {
        transferQueue_ = cfg.graphicsQueue;
        transferQueueFamily_ = cfg.graphicsQueueFamily;
    }
    if (transferQueue_ == VK_NULL_HANDLE || transferQueueFamily_ == UINT32_MAX) {
        VVM_LOG_ERROR("initCopyEngine: no usable queue for copy operations");
        return false;
    }

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = transferQueueFamily_;

    VkResult res = vkCreateCommandPool(localPools_[0].getDevice(), &poolInfo, nullptr, &copyCmdPool_);
    if (res != VK_SUCCESS) {
        VVM_LOG_ERROR("initCopyEngine: failed to create command pool: {}", vkResultToString(res));
        return false;
    }
    return true;
}

std::optional<Allocation> MultiNodePoolManager::createStaging(VkDeviceSize size) {
    if (localPools_.empty()) return std::nullopt;

    auto staging = localPools_[0].allocate(
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (!staging || !staging->hostPtr) {
        VVM_LOG_ERROR("createStaging: failed to allocate host staging buffer of {} bytes", size);
        return std::nullopt;
    }
    return staging;
}

bool MultiNodePoolManager::copyDeviceToHost(const Allocation& src, VkDeviceSize srcOffset,
                                            const Allocation& staging, VkDeviceSize size) {
    return runCopy(src.buffer, staging.buffer, srcOffset, 0, size);
}

bool MultiNodePoolManager::copyHostToDevice(const Allocation& staging, const Allocation& dst,
                                            VkDeviceSize dstOffset, VkDeviceSize size) {
    return runCopy(staging.buffer, dst.buffer, 0, dstOffset, size);
}

bool MultiNodePoolManager::runCopy(VkBuffer srcBuffer, VkBuffer dstBuffer,
                                   VkDeviceSize srcOffset, VkDeviceSize dstOffset,
                                   VkDeviceSize size) {
    if (copyCmdPool_ == VK_NULL_HANDLE || srcBuffer == VK_NULL_HANDLE || dstBuffer == VK_NULL_HANDLE) {
        VVM_LOG_ERROR("runCopy: invalid copy context (pool=%p src=%p dst=%p)",
                      copyCmdPool_, srcBuffer, dstBuffer);
        return false;
    }

    VkDevice device = localPools_[0].getDevice();

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = copyCmdPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != VK_SUCCESS) return false;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(cmdBuffer, &beginInfo) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, copyCmdPool_, 1, &cmdBuffer);
        return false;
    }

    VkBufferCopy region{};
    region.srcOffset = srcOffset;
    region.dstOffset = dstOffset;
    region.size = size;
    vkCmdCopyBuffer(cmdBuffer, srcBuffer, dstBuffer, 1, &region);

    if (vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, copyCmdPool_, 1, &cmdBuffer);
        return false;
    }

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, copyCmdPool_, 1, &cmdBuffer);
        return false;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    VkResult res = vkQueueSubmit(transferQueue_, 1, &submitInfo, fence);
    if (res == VK_SUCCESS) {
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    } else {
        VVM_LOG_ERROR("runCopy: vkQueueSubmit failed: {}", vkResultToString(res));
    }

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, copyCmdPool_, 1, &cmdBuffer);
    return res == VK_SUCCESS;
}

uint64_t MultiNodePoolManager::registerAllocation(Allocation&& alloc) {
    std::lock_guard<std::mutex> lock(allocsMutex_);
    uint64_t id = nextAllocId_++;
    remoteAllocs_[id] = std::move(alloc);
    return id;
}

std::optional<Allocation> MultiNodePoolManager::getRegisteredAllocation(uint64_t localAllocId) const {
    std::lock_guard<std::mutex> lock(allocsMutex_);
    auto it = remoteAllocs_.find(localAllocId);
    if (it == remoteAllocs_.end()) return std::nullopt;
    return it->second;
}

std::optional<Allocation> MultiNodePoolManager::findAllocation(uint64_t localAllocId) {
    std::lock_guard<std::mutex> lock(allocsMutex_);
    auto it = remoteAllocs_.find(localAllocId);
    if (it == remoteAllocs_.end()) return std::nullopt;
    return it->second;
}

bool MultiNodePoolManager::unregisterAllocation(uint64_t localAllocId) {
    {
        std::lock_guard<std::mutex> lock(allocsMutex_);
        if (remoteAllocs_.erase(localAllocId) == 0) return false;
    }
    // Close any pending exported handle for this allocation.
    std::lock_guard<std::mutex> lock(g_pendingExportsMutex);
    g_pendingExports.erase(pendingKey(localNodeId_, localAllocId));
    return true;
}

std::optional<uint64_t> MultiNodePoolManager::findAllocIdByBuffer(VkBuffer buffer) {
    if (buffer == VK_NULL_HANDLE) return std::nullopt;
    std::lock_guard<std::mutex> lock(allocsMutex_);
    for (const auto& [id, alloc] : remoteAllocs_) {
        if (alloc.buffer == buffer) return id;
    }
    return std::nullopt;
}

TcpTransport::ConnId MultiNodePoolManager::getPeerConnection(const std::string& host, uint16_t port) {
    std::string key = host + ":" + std::to_string(port);

    std::lock_guard<std::mutex> lock(connsMutex_);
    auto it = peerConns_.find(key);
    if (it != peerConns_.end() && tcpTransport_->isConnected(it->second)) {
        return it->second;
    }

    if (!tcpTransport_) return 0;

    // Hold the lock during connect to prevent TOCTOU race where two threads
    // simultaneously create connections to the same peer.
    auto conn = tcpTransport_->connect(host, port,
                                       static_cast<int32_t>(networkConfig_.connectTimeout.count()));
    if (!conn || *conn == 0) return 0;

    peerConns_[key] = *conn;
    return *conn;
}

void MultiNodePoolManager::heartbeatLoop() {
    while (!stopHeartbeat_) {
        std::this_thread::sleep_for(networkConfig_.heartbeatInterval);

        if (stopHeartbeat_ || !tcpTransport_) break;

        auto body = serializeNodeId(localNodeId_);
        for (const auto& [host, port] : seedEndpoints_) {
            auto conn = getPeerConnection(host, port);
            if (conn == 0) continue;

            TcpMessage req;
            req.type = MsgHeartbeat;
            req.flags = TcpFlagsRequest;
            req.body = body;

            auto resp = tcpTransport_->request(conn, req);
            if (!resp || resp->flags == TcpFlagsError) continue;

            std::vector<NodeInfo> view;
            if (deserializeNodeList(resp->body, view)) {
                mergeClusterView(view);
            }
        }
    }
}

void MultiNodePoolManager::mergeClusterView(const std::vector<NodeInfo>& view) {
    std::lock_guard<std::mutex> lock(clusterViewMutex_);
    for (const auto& info : view) {
        bool found = false;
        for (auto& existing : clusterView_) {
            if (existing.id == info.id) {
                existing = info;
                found = true;
                break;
            }
        }
        if (!found) {
            clusterView_.push_back(info);
        }
    }
}

void MultiNodePoolManager::parseListenAddress(const std::string& listenAddress,
                                              std::string& outHost, uint16_t& outPort) {
    outHost = "0.0.0.0";
    outPort = 50051;

    size_t colon = listenAddress.rfind(':');
    if (colon == std::string::npos) return;

    outHost = listenAddress.substr(0, colon);
    try {
        outPort = static_cast<uint16_t>(std::stoul(listenAddress.substr(colon + 1)));
    } catch (...) {
        outPort = 50051;
    }
}

bool MultiNodePoolManager::parseEndpoint(const std::string& endpoint,
                                         std::string& outHost, uint16_t& outPort) {
    size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos) return false;
    outHost = endpoint.substr(0, colon);
    try {
        outPort = static_cast<uint16_t>(std::stoul(endpoint.substr(colon + 1)));
    } catch (...) {
        return false;
    }
    return outPort != 0;
}

// ============================================================================
// TCP request dispatch (server side)
// ============================================================================

void MultiNodePoolManager::onTcpRequest(TcpMessage& request, TcpMessage& response) {
    const uint8_t* p = request.body.data();
    const uint8_t* end = p + request.body.size();

    switch (request.type) {
        case MsgRegisterNode: {
            NodeInfo info;
            if (!deserializeNodeInfo(p, end, info)) {
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }
            auto view = handleRegisterRequest(info);
            response = makeResponse(request.type, TcpFlagsResponse);
            if (view) response.body = serializeNodeList(*view);
            break;
        }
        case MsgGetClusterView: {
            std::lock_guard<std::mutex> lock(clusterViewMutex_);
            response = makeResponse(request.type, TcpFlagsResponse);
            response.body = serializeNodeList(clusterView_);
            break;
        }
        case MsgAllocate: {
            uint64_t size = 0;
            uint32_t usage = 0, flags = 0;
            uint8_t enableRdma = 0;
            if (!detail::getU64(p, end, size) ||
                !detail::getU32(p, end, usage) ||
                !detail::getU32(p, end, flags) ||
                !detail::getU8(p, end, enableRdma)) {
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }
            NodeId requester = localNodeId_;
            auto desc = handleAllocateRequest(requester, size, usage, flags, enableRdma != 0);
            response = makeResponse(request.type, TcpFlagsResponse);
            if (desc) {
                detail::putU8(response.body, 1);
                auto bytes = serializeAllocationDesc(*desc);
                response.body.insert(response.body.end(), bytes.begin(), bytes.end());
            } else {
                detail::putU8(response.body, 0);
            }
            break;
        }
        case MsgExport: {
            uint64_t localAllocId = 0;
            uint8_t enableRdma = 0, forceHostShadow = 0;
            if (!detail::getU64(p, end, localAllocId) ||
                !detail::getU8(p, end, enableRdma) ||
                !detail::getU8(p, end, forceHostShadow)) {
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }
            auto desc = handleExportRequest(localNodeId_, localAllocId, enableRdma != 0, forceHostShadow != 0);
            response = makeResponse(request.type, TcpFlagsResponse);
            if (desc) {
                detail::putU8(response.body, 1);
                auto bytes = serializeAllocationDesc(*desc);
                response.body.insert(response.body.end(), bytes.begin(), bytes.end());
            } else {
                detail::putU8(response.body, 0);
            }
            break;
        }
        case MsgImport: {
            RemoteAllocationDesc desc;
            uint32_t usage = 0;
            if (!deserializeAllocationDesc(p, end, desc) || !detail::getU32(p, end, usage)) {
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }
            auto alloc = handleImportRequest(localNodeId_, desc, usage);
            response = makeResponse(request.type, TcpFlagsResponse);
            if (alloc) {
                detail::putU8(response.body, 1);
                detail::putU64(response.body, registerAllocation(std::move(*alloc)));
            } else {
                detail::putU8(response.body, 0);
            }
            break;
        }
        case MsgMigratePull: {
            uint64_t localAllocId = 0, srcOffset = 0, size = 0;
            if (!detail::getU64(p, end, localAllocId) ||
                !detail::getU64(p, end, srcOffset) ||
                !detail::getU64(p, end, size)) {
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }

            auto alloc = findAllocation(localAllocId);
            if (!alloc) {
                VVM_LOG_ERROR("MigratePull: unknown allocation {}", localAllocId);
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }

            auto staging = createStaging(size);
            if (!staging) {
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }

            if (!copyDeviceToHost(*alloc, srcOffset, *staging, size)) {
                VVM_LOG_ERROR("MigratePull: device -> host copy failed");
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }

            response = makeResponse(request.type, TcpFlagsResponse);
            response.streamSource = staging->hostPtr;
            response.streamLen = size;
            response.streamCleanup = [staging = std::move(staging)]() mutable { (void)staging; };
            VVM_LOG_INFO("MigratePull: serving {} bytes for allocation {}", size, localAllocId);
            break;
        }
        case MsgMigratePush: {
            uint64_t localAllocId = 0, size = 0;
            if (!detail::getU64(p, end, localAllocId) || !detail::getU64(p, end, size)) {
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }

            auto alloc = findAllocation(localAllocId);
            if (!alloc) {
                VVM_LOG_ERROR("MigratePush: unknown allocation {}", localAllocId);
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }

            if (!request.streamReceived) {
                // Prepare phase: allocate the sink the incoming stream is staged into.
                auto rawStaging = createStaging(size);
                if (!rawStaging) {
                    response = makeResponse(request.type, TcpFlagsError);
                    return;
                }
                // Shared ownership keeps the staging alive across both handler
                // phases; the raw Allocation* is stable because shared_ptr
                // never moves the pointee.
                auto staging = std::make_shared<Allocation>(std::move(*rawStaging));
                request.streamContext = staging.get();
                request.streamSink = staging->hostPtr;
                request.streamSinkCleanup = [staging]() mutable { (void)staging; };
                response = makeResponse(request.type, TcpFlagsResponse);
                response.streamSink = staging->hostPtr;
                response.streamSinkCleanup = [staging]() mutable { (void)staging; };
                return;
            }

            auto* staging = static_cast<Allocation*>(request.streamContext);
            if (staging == nullptr || staging->hostPtr == nullptr) {
                VVM_LOG_ERROR("MigratePush: missing staging buffer on finalize");
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }

            if (!copyHostToDevice(*staging, *alloc, 0, size)) {
                VVM_LOG_ERROR("MigratePush: host -> device copy failed");
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }

            response = makeResponse(request.type, TcpFlagsResponse);
            detail::putU8(response.body, 1);
            VVM_LOG_INFO("MigratePush: received {} bytes for allocation {}", size, localAllocId);
            break;
        }
        case MsgHeartbeat: {
            NodeId node;
            if (!deserializeNodeId(p, end, node)) {
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }
            NodeInfo info;
            info.id = node;
            info.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            mergeClusterView({info});
            std::lock_guard<std::mutex> lock(clusterViewMutex_);
            response = makeResponse(request.type, TcpFlagsResponse);
            response.body = serializeNodeList(clusterView_);
            break;
        }
        case MsgLeaveCluster: {
            NodeId node;
            if (deserializeNodeId(p, end, node)) {
                std::lock_guard<std::mutex> lock(clusterViewMutex_);
                clusterView_.erase(std::remove_if(clusterView_.begin(), clusterView_.end(),
                                                  [&](const NodeInfo& n) { return n.id == node; }),
                                   clusterView_.end());
            }
            response = makeResponse(request.type, TcpFlagsResponse);
            break;
        }
        case MsgDeallocate: {
            uint64_t localAllocId = 0;
            if (!detail::getU64(p, end, localAllocId)) {
                response = makeResponse(request.type, TcpFlagsError);
                return;
            }
            if (auto alloc = findAllocation(localAllocId)) {
                unregisterAllocation(localAllocId);
                localPools_[0].deallocate(std::move(*alloc));
                VVM_LOG_INFO("Deallocate: freed remote allocation {}", localAllocId);
            } else {
                VVM_LOG_WARN("Deallocate: unknown allocation {}", localAllocId);
            }
            response = makeResponse(request.type, TcpFlagsResponse);
            detail::putU8(response.body, 1);
            break;
        }
        default:
            VVM_LOG_WARN("Unknown TCP message type {}", request.type);
            response = makeResponse(request.type, TcpFlagsError);
            break;
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

std::optional<Allocation> MultiNodePoolManager::createLocalAllocationForImport(
    const RemoteAllocationDesc& desc,
    VkBufferUsageFlags usage) {

    // Same-process zero-copy path: if THIS process still owns the exported
    // handle (registry keyed by owner node + allocId), import it directly.
    // Consuming moves the handle into the driver (or closes it on failure).
    {
        std::lock_guard<std::mutex> lock(g_pendingExportsMutex);
        auto it = g_pendingExports.find(pendingKey(desc.owner, desc.localAllocId));
        if (it != g_pendingExports.end()) {
            ExternalMemoryInfo ext = std::move(it->second);
            g_pendingExports.erase(it);
            ext.type = static_cast<vvm::ExternalHandleType>(desc.handleType);
            ext.size = desc.size;
            auto imported = localPools_[0].importMemory(std::move(ext), usage);
            if (imported) {
                VVM_LOG_INFO("importRemote: zero-copy handle import for allocId=%llu succeeded",
                             desc.localAllocId);
                return imported;
            }
            VVM_LOG_WARN("importRemote: same-process handle import failed; "
                         "falling back to host-staged copy");
        }
    }

    auto alloc = localPools_[0].allocate(desc.size, usage, 0);
    if (!alloc) {
        VVM_LOG_ERROR("Failed to allocate local memory for import");
        return std::nullopt;
    }

    // SECURITY: OS handles (FDs/HANDLEs) are process-relative and CANNOT be
    // transferred over TCP. The externalHandle field in RemoteAllocationDesc
    // is only valid for same-process imports (handled above via g_pendingExports).
    // Cross-machine peers must use host-staged copies. Reject any network-received
    // handle values to prevent FD/HANDLE injection attacks.
    if (!desc.externalHandle.empty()) {
        VVM_LOG_WARN("importRemote: rejecting %zu-byte external handle received over network "
                     "(OS handles are process-relative and cannot be transferred via TCP; "
                     "use host-staged migration instead)",
                     desc.externalHandle.size());
    }

    return alloc;
}

#if defined(VVM_NETWORK_HAS_VERBS)
bool MultiNodePoolManager::registerMemoryForRdma(const Allocation& alloc, uint64_t& outRdmaAddr, uint32_t& outRkey) {
    if (!rdmaTransport_) return false;

    auto region = rdmaTransport_->registerGpuMemory(alloc.memory, alloc.offset, alloc.size, alloc.buffer);
    if (!region) return false;

    outRdmaAddr = region->rdmaAddr;
    outRkey = region->rkey;
    return true;
}

void MultiNodePoolManager::unregisterMemoryForRdma(const Allocation& alloc) {
    (void)alloc;
}
#else
bool MultiNodePoolManager::registerMemoryForRdma(const Allocation& alloc, uint64_t& outRdmaAddr, uint32_t& outRkey) {
    (void)alloc;
    (void)outRdmaAddr;
    (void)outRkey;
    return false;
}

void MultiNodePoolManager::unregisterMemoryForRdma(const Allocation& alloc) {
    (void)alloc;
}
#endif

std::optional<NetworkMigrationOperation> MultiNodePoolManager::migrateHostStaged(
    const RemoteAllocationDesc& source,
    Allocation& destination,
    bool toHost,
    uint64_t timeoutNs) {

    if (toHost) {
        return migrateFromRemote(source, destination, false, timeoutNs);
    }
    VVM_LOG_ERROR("migrateHostStaged: push path requires a RemoteAllocationDesc destination");
    return std::nullopt;
}

// ============================================================================
// RPC Handlers
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

    auto desc = exportForRemote(*alloc, enableRdma, false);
    if (!desc) {
        localPools_[0].deallocate(std::move(*alloc));
        return std::nullopt;
    }
    return desc;
}

std::optional<RemoteAllocationDesc> MultiNodePoolManager::handleExportRequest(
    const NodeId& requester,
    uint64_t localAllocId,
    bool enableRdma,
    bool forceHostShadow) {

    VVM_LOG_INFO("Export request from {} for alloc {}", requester.toString(), localAllocId);

    auto alloc = findAllocation(localAllocId);
    if (!alloc) return std::nullopt;

    return exportForRemote(*alloc, enableRdma, forceHostShadow);
}

std::optional<Allocation> MultiNodePoolManager::handleImportRequest(
    const NodeId& requester,
    const RemoteAllocationDesc& desc,
    VkBufferUsageFlags usage) {

    VVM_LOG_INFO("Import request from {}: {} bytes", requester.toString(), desc.size);

    return importRemote(desc, usage);
}

std::optional<NetworkMigrationOperation> MultiNodePoolManager::handleMigrateRequest(
    const RemoteAllocationDesc& source,
    uint64_t destinationAllocId,
    bool useRdma,
    uint64_t timeoutNs) {

    (void)source;
    (void)destinationAllocId;
    (void)useRdma;
    (void)timeoutNs;
    VVM_LOG_WARN("handleMigrateRequest: use MigratePush/MigratePull over TCP instead");
    return std::nullopt;
}

std::optional<std::vector<NodeInfo>> MultiNodePoolManager::handleRegisterRequest(
    const NodeInfo& info) {

    VVM_LOG_INFO("Node registration from: {}", info.id.toString());

    std::lock_guard<std::mutex> lock(clusterViewMutex_);
    bool found = false;
    for (auto& existing : clusterView_) {
        if (existing.id == info.id) {
            existing = info;
            found = true;
            break;
        }
    }
    if (!found) {
        clusterView_.push_back(info);
    }

    return clusterView_;
}

}  // namespace network
}  // namespace vvm