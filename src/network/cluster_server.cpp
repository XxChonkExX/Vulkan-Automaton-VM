#include "vulkan_vm/network/cluster_server.hpp"
#include "vulkan_vm/network/tcp_transport.hpp"
#include "vulkan_vm/network/network_types.hpp"
#include "vulkan_vm/network/wire_format.hpp"
#include "vulkan_vm/logging.hpp"

#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <functional>
#include <thread>
#include <algorithm>

namespace vvm {
namespace network {

static AuthorizationResult authorize(const PeerIdentity& peer, Capability cap) {
    if (!peer.hasCapability(cap)) {
        return AuthorizationResult::InsufficientCapabilities;
    }
    return AuthorizationResult::Allow;
}

static PeerIdentity getPeerIdentityFromTLS(const TcpTransport::ConnId& /*connId*/) {
    // In a real implementation, this would extract the peer's certificate fingerprint
    // from the TLS connection. For now, return a default identity with all capabilities.
    PeerIdentity identity;
    identity.capabilities = {
        Capability::RegisterNode,
        Capability::ReadClusterView,
        Capability::AllocateMemory,
        Capability::MigrateMemory,
        Capability::RDMAAccess,
        Capability::PublishModel,
        Capability::FetchModel,
        Capability::AdministerCluster
    };
    return identity;
}

static AuthorizationResult checkAuth(const TcpTransport::ConnId& connId, Capability cap) {
    PeerIdentity peer = getPeerIdentityFromTLS(connId);
    return authorize(peer, cap);
}

// ============================================================================
// TCP-based ClusterServer Implementation
// ============================================================================

class ClusterServerImpl : public ClusterServer {
public:
    ClusterServerImpl(const NetworkConfig& config, MultiNodePoolManager* poolManager)
        : config_(config), poolManager_(poolManager) {}

    ~ClusterServerImpl() override {
        stop();
    }

    bool start() override {
        if (running_.load()) return true;

        transport_ = TcpTransport::create();
        if (!transport_) {
            VVM_LOG_ERROR("Failed to create TCP transport for cluster server");
            return false;
        }

        uint16_t port = 51010; // default control port
        size_t colonPos = config_.listenAddress.rfind(':');
        if (colonPos != std::string::npos) {
            try {
                port = static_cast<uint16_t>(std::stoul(config_.listenAddress.substr(colonPos + 1)));
            } catch (...) {
                VVM_LOG_WARN("Invalid port in listen address, using default 51010");
            }
        }

        if (!transport_->start(config_.listenAddress.substr(0, config_.listenAddress.rfind(':')),
                               port,
                               [this](TcpMessage& req, TcpMessage& resp) {
            handleRequest(req, resp);
        })) {
            VVM_LOG_ERROR("Failed to start TCP transport for cluster server");
            return false;
        }

        running_.store(true);
        VVM_LOG_INFO("Cluster server started on port {}", port);
        return true;
    }

    void stop() override {
        if (!running_.load()) return;

        running_.store(false);
        
        if (transport_) {
            transport_->stop();
            transport_.reset();
        }

        VVM_LOG_INFO("Cluster server stopped");
    }

    bool isRunning() const override {
        return running_.load();
    }

    void setAllocateHandler(AllocateHandler handler) override {
        std::lock_guard<std::mutex> lock(handlerMutex_);
        allocateHandler_ = std::move(handler);
    }

    void setExportHandler(ExportHandler handler) override {
        std::lock_guard<std::mutex> lock(handlerMutex_);
        exportHandler_ = std::move(handler);
    }

    void setImportHandler(ImportHandler handler) override {
        std::lock_guard<std::mutex> lock(handlerMutex_);
        importHandler_ = std::move(handler);
    }

    void setMigrateHandler(MigrateHandler handler) override {
        std::lock_guard<std::mutex> lock(handlerMutex_);
        migrateHandler_ = std::move(handler);
    }

    void setRegisterHandler(RegisterHandler handler) override {
        std::lock_guard<std::mutex> lock(handlerMutex_);
        registerHandler_ = std::move(handler);
    }

    void setAuthCallback(AuthCallback callback) override {
        std::lock_guard<std::mutex> lock(handlerMutex_);
        authCallback_ = std::move(callback);
    }

    void updateClusterView(const std::vector<NodeInfo>& view) override {
        std::lock_guard<std::mutex> lock(viewMutex_);
        clusterView_ = view;
    }

    std::vector<NodeInfo> getClusterView() const override {
        std::lock_guard<std::mutex> lock(viewMutex_);
        return clusterView_;
    }

    ServerStats getStats() const override {
        std::lock_guard<std::mutex> lock(statsMutex_);
        return stats_;
    }

private:
    void handleRequest(TcpMessage& req, TcpMessage& resp) {
        if (!running_.load()) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(0)};
            return;
        }

        // Check authorization if auth callback is set
        if (authCallback_) {
            AuthorizationResult authResult = authCallback_(req.type);
            if (authResult != AuthorizationResult::Allow) {
                resp.type = MsgError;
                resp.body = {static_cast<uint8_t>(authResult == AuthorizationResult::Unauthenticated ? 10 : 11)};
                return;
            }
        }

        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.rpcCallsReceived++;

        try {
            switch (req.type) {
                case MsgAllocate:
                    handleAllocate(req, resp);
                    break;
                case MsgExport:
                    handleExport(req, resp);
                    break;
                case MsgImport:
                    handleImport(req, resp);
                    break;
                case MsgMigrate:
                    handleMigrate(req, resp);
                    break;
                case MsgRegisterNode:
                    handleRegisterNode(req, resp);
                    break;
                case MsgHeartbeat:
                    handleHeartbeat(req, resp);
                    break;
                case MsgGetClusterView:
                    handleGetClusterView(req, resp);
                    break;
                case MsgLeaveCluster:
                    handleLeaveCluster(req, resp);
                    break;
                default:
                    resp.type = MsgError;
                    resp.body = {static_cast<uint8_t>(1)}; // unknown message type
                    break;
            }
        } catch (const std::exception& e) {
            VVM_LOG_ERROR("Exception handling request: {}", e.what());
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(2)}; // internal error
        }

        std::lock_guard<std::mutex> lock(statsMutex_);
        if (resp.type == MsgError) {
            stats_.rpcCallsFailed++;
        } else {
            stats_.rpcCallsSucceeded++;
        }
    }

    void handleAllocate(const TcpMessage& req, TcpMessage& resp) {
        if (!allocateHandler_) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(3)};
            return;
        }

        const uint8_t* p = req.body.data();
        const uint8_t* end = p + req.body.size();
        std::string host;
        uint32_t port = 0, nodeIndex = 0;
        std::string uuid;
        uint64_t size = 0, usageVal = 0, flagsVal = 0;
        uint8_t enableRdma = 0;
        if (!wire::getStr(p, end, host) || !wire::getU32(p, end, port) ||
            !wire::getU32(p, end, nodeIndex) || !wire::getStr(p, end, uuid) ||
            !wire::getU64(p, end, size) || !wire::getU64(p, end, usageVal) ||
            !wire::getU64(p, end, flagsVal) || !wire::getU8(p, end, enableRdma)) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)}; // malformed request
            return;
        }

        NodeId requester{host, port, nodeIndex, uuid};
        auto result = allocateHandler_(requester, static_cast<VkDeviceSize>(size),
                                       static_cast<VkBufferUsageFlags>(usageVal),
                                       static_cast<VkMemoryPropertyFlags>(flagsVal),
                                       enableRdma != 0);
        if (!result) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(5)}; // allocation failed
            return;
        }

        resp.type = MsgAllocate;
        resp.body = serializeAllocationDesc(*result);
    }

    void handleExport(const TcpMessage& req, TcpMessage& resp) {
        if (!exportHandler_) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(3)};
            return;
        }

        const uint8_t* p = req.body.data();
        const uint8_t* end = p + req.body.size();
        std::string host;
        uint32_t port = 0, nodeIndex = 0;
        std::string uuid;
        uint64_t localAllocId = 0;
        uint8_t enableRdma = 0, forceHostShadow = 0;
        if (!wire::getStr(p, end, host) || !wire::getU32(p, end, port) ||
            !wire::getU32(p, end, nodeIndex) || !wire::getStr(p, end, uuid) ||
            !wire::getU64(p, end, localAllocId) ||
            !wire::getU8(p, end, enableRdma) || !wire::getU8(p, end, forceHostShadow)) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)};
            return;
        }

        NodeId requester{host, port, nodeIndex, uuid};
        auto result = exportHandler_(requester, localAllocId, enableRdma != 0, forceHostShadow != 0);
        if (!result) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(5)};
            return;
        }

        resp.type = MsgExport;
        resp.body = serializeAllocationDesc(*result);
    }

    void handleImport(const TcpMessage& req, TcpMessage& resp) {
        if (!importHandler_) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(3)};
            return;
        }

        const uint8_t* p = req.body.data();
        const uint8_t* end = p + req.body.size();
        std::string host;
        uint32_t port = 0, nodeIndex = 0;
        std::string uuid;
        uint64_t size = 0, usageVal = 0, descUsageVal = 0;
        uint8_t hasRdma = 0, hasUcx = 0, hasHostShadow = 0;
        if (!wire::getStr(p, end, host) || !wire::getU32(p, end, port) ||
            !wire::getU32(p, end, nodeIndex) || !wire::getStr(p, end, uuid) ||
            !wire::getU64(p, end, size) || !wire::getU64(p, end, usageVal) ||
            !wire::getU64(p, end, descUsageVal) || !wire::getU8(p, end, hasRdma)) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)};
            return;
        }

        RemoteAllocationDesc desc;
        desc.owner = NodeId{host, port, nodeIndex, uuid};
        desc.size = size;
        desc.usageFlags = static_cast<VkBufferUsageFlags>(descUsageVal);
        desc.hasRdmaAddr = (hasRdma != 0);
        if (desc.hasRdmaAddr) {
            if (!wire::getU64(p, end, desc.rdmaAddr) || !wire::getU32(p, end, desc.rkey)) {
                resp.type = MsgError;
                resp.body = {static_cast<uint8_t>(4)};
                return;
            }
        }
        if (!wire::getU8(p, end, hasUcx)) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)};
            return;
        }
        desc.hasUcxAddr = (hasUcx != 0);
        if (desc.hasUcxAddr) {
            if (!wire::getBytes(p, end, desc.ucxWorkerAddr) ||
                !wire::getBytes(p, end, desc.ucxPackedRkey) ||
                !wire::getU64(p, end, desc.ucxRemoteAddr) ||
                !wire::getU32(p, end, desc.ucxDeviceIndex)) {
                resp.type = MsgError;
                resp.body = {static_cast<uint8_t>(4)};
                return;
            }
        }
        uint8_t hostShadow = 0;
        if (!wire::getU8(p, end, hostShadow)) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)};
            return;
        }
        desc.hasHostShadow = (hostShadow != 0);
        if (!wire::getBytes(p, end, desc.externalHandle) || !wire::getStr(p, end, desc.allocationName)) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)};
            return;
        }

        NodeId requester{host, port, nodeIndex, uuid};
        auto result = importHandler_(requester, desc, static_cast<VkBufferUsageFlags>(usageVal));
        if (!result) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(5)};
            return;
        }

        resp.type = MsgImport;
        resp.body = serializeAllocation(*result);
    }

    void handleMigrate(const TcpMessage& req, TcpMessage& resp) {
        if (!migrateHandler_) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(3)};
            return;
        }

        const uint8_t* p = req.body.data();
        const uint8_t* end = p + req.body.size();
        std::string host;
        uint32_t port = 0, nodeIndex = 0;
        std::string uuid;
        uint64_t size = 0, localAllocId = 0, dstAllocId = 0;
        uint8_t hasRdma = 0, useRdma = 0;
        if (!wire::getStr(p, end, host) || !wire::getU32(p, end, port) ||
            !wire::getU32(p, end, nodeIndex) || !wire::getStr(p, end, uuid) ||
            !wire::getU64(p, end, size) || !wire::getU64(p, end, localAllocId) ||
            !wire::getU8(p, end, hasRdma) || !wire::getU64(p, end, dstAllocId) ||
            !wire::getU8(p, end, useRdma)) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)};
            return;
        }

        RemoteAllocationDesc source;
        source.owner = NodeId{host, port, nodeIndex, uuid};
        source.size = size;
        source.localAllocId = localAllocId;
        source.hasRdmaAddr = (hasRdma != 0);
        if (source.hasRdmaAddr) {
            if (!wire::getU64(p, end, source.rdmaAddr) || !wire::getU32(p, end, source.rkey)) {
                resp.type = MsgError;
                resp.body = {static_cast<uint8_t>(4)};
                return;
            }
        }

        NodeId requester{host, port, nodeIndex, uuid};
        auto result = migrateHandler_(requester, source, dstAllocId, useRdma != 0);
        if (!result) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(5)};
            return;
        }

        resp.type = MsgMigrate;
        resp.body = serializeMigrationOp(*result);
    }

    static constexpr size_t kMaxHostnameBytes = 255;
    static constexpr size_t kMaxUuidBytes = 128;
    static constexpr size_t kMaxGpuNameBytes = 256;
    static constexpr size_t kMaxNicNameBytes = 256;
    static constexpr uint32_t kMaxGpuDevices = 64;

    void handleRegisterNode(const TcpMessage& req, TcpMessage& resp) {
        if (!registerHandler_) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(3)};
            return;
        }

        const uint8_t* p = req.body.data();
        const uint8_t* end = p + req.body.size();
        NodeInfo info;

        if (!wire::getStrLimited(p, end, info.id.host, kMaxHostnameBytes) ||
            !wire::getU32(p, end, info.id.port) ||
            !wire::getU32(p, end, info.id.nodeIndex) ||
            !wire::getStrLimited(p, end, info.id.uuid, kMaxUuidBytes)) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)};
            return;
        }
        uint32_t devCount = 0;
        if (!wire::getU32(p, end, devCount) || devCount > kMaxGpuDevices) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)};
            return;
        }
        info.gpuDevices.reserve(devCount);
        for (uint32_t d = 0; d < devCount; ++d) {
            std::string gpu;
            if (!wire::getStrLimited(p, end, gpu, kMaxGpuNameBytes)) {
                resp.type = MsgError;
                resp.body = {static_cast<uint8_t>(4)};
                return;
            }
            info.gpuDevices.emplace_back(std::move(gpu));
        }
        uint8_t v = 0;
        if (!wire::getStrLimited(p, end, info.nicName, kMaxNicNameBytes) || !wire::getU8(p, end, v)) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)};
            return;
        }
        info.rdmaCapable = (v != 0);
        if (!wire::getU8(p, end, v)) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)};
            return;
        }
        info.gpuDirectCapable = (v != 0);
        if (!wire::getU64(p, end, info.timestamp) || !wire::getU64(p, end, info.caps.totalVram) ||
            !wire::getU64(p, end, info.caps.availableVram)) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)};
            return;
        }
        if (!wire::getU8(p, end, v)) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(4)};
            return;
        }
        info.caps.supportsRdmaAddr = (v != 0);
        if (!wire::getU8(p, end, v)) { resp.type = MsgError; resp.body = {4}; return; }
        info.caps.supportsDmaBuf = (v != 0);
        if (!wire::getU8(p, end, v)) { resp.type = MsgError; resp.body = {4}; return; }
        info.caps.supportsOpaqueFd = (v != 0);
        if (!wire::getU8(p, end, v)) { resp.type = MsgError; resp.body = {4}; return; }
        info.caps.supportsOpaqueWin32 = (v != 0);
        if (!wire::getU8(p, end, v)) { resp.type = MsgError; resp.body = {4}; return; }
        info.caps.supportsD3D12Heap = (v != 0);

        auto result = registerHandler_(info);
        if (!result) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(5)};
            return;
        }

        resp.type = MsgRegisterNode;
        resp.body = serializeNodeList(*result);
    }

    void handleHeartbeat(const TcpMessage& req, TcpMessage& resp) {
        // Heartbeat just returns current cluster view
        std::lock_guard<std::mutex> lock(viewMutex_);
        resp.type = MsgHeartbeat;
        resp.body = serializeNodeList(clusterView_);
    }

    void handleGetClusterView(const TcpMessage& req, TcpMessage& resp) {
        std::lock_guard<std::mutex> lock(viewMutex_);
        resp.type = MsgGetClusterView;
        resp.body = serializeNodeList(clusterView_);
    }

    void handleLeaveCluster(const TcpMessage& req, TcpMessage& resp) {
        // Node is leaving, remove from cluster view
        // For now, just acknowledge
        resp.type = MsgLeaveCluster;
        resp.body = {static_cast<uint8_t>(0)}; // success
    }

    std::vector<uint8_t> serializeNodeList(const std::vector<NodeInfo>& nodes) {
        std::vector<uint8_t> data;
        wire::putU32(data, static_cast<uint32_t>(nodes.size()));
        for (const auto& info : nodes) {
            wire::putStr(data, info.id.host);
            wire::putU32(data, info.id.port);
            wire::putU32(data, info.id.nodeIndex);
            wire::putStr(data, info.id.uuid);
            wire::putU32(data, static_cast<uint32_t>(info.gpuDevices.size()));
            for (const auto& dev : info.gpuDevices) {
                wire::putStr(data, dev);
            }
            wire::putStr(data, info.nicName);
            wire::putU8(data, info.rdmaCapable ? 1 : 0);
            wire::putU8(data, info.gpuDirectCapable ? 1 : 0);
            wire::putU64(data, info.timestamp);
            wire::putU64(data, info.caps.totalVram);
            wire::putU64(data, info.caps.availableVram);
            wire::putU8(data, info.caps.supportsRdmaAddr ? 1 : 0);
            wire::putU8(data, info.caps.supportsDmaBuf ? 1 : 0);
            wire::putU8(data, info.caps.supportsOpaqueFd ? 1 : 0);
            wire::putU8(data, info.caps.supportsOpaqueWin32 ? 1 : 0);
            wire::putU8(data, info.caps.supportsD3D12Heap ? 1 : 0);
        }
        return data;
    }

    static std::vector<uint8_t> serializeAllocationDesc(const RemoteAllocationDesc& desc) {
        std::vector<uint8_t> data;
        wire::putStr(data, desc.owner.host);
        wire::putU32(data, desc.owner.port);
        wire::putU32(data, desc.owner.nodeIndex);
        wire::putStr(data, desc.owner.uuid);
        wire::putU64(data, desc.size);
        wire::putU64(data, static_cast<uint64_t>(desc.usageFlags));
        wire::putU8(data, desc.hasRdmaAddr ? 1 : 0);
        if (desc.hasRdmaAddr) {
            wire::putU64(data, desc.rdmaAddr);
            wire::putU32(data, desc.rkey);
        }
        return data;
    }

    static std::vector<uint8_t> serializeAllocation(const Allocation& alloc) {
        std::vector<uint8_t> data;
        wire::putU8(data, 1); // success
        wire::putU64(data, alloc.size);
        wire::putU64(data, reinterpret_cast<uintptr_t>(alloc.buffer));
        wire::putU64(data, reinterpret_cast<uintptr_t>(alloc.memory));
        wire::putU64(data, reinterpret_cast<uintptr_t>(alloc.hostPtr));
        wire::putU64(data, alloc.deviceAddress);
        wire::putU64(data, alloc.offset);
        wire::putU64(data, alloc.blockIndex);
        wire::putU64(data, static_cast<uint64_t>(alloc.usage));
        wire::putU8(data, alloc.mapped ? 1 : 0);
        wire::putStr(data, alloc.name);
        return data;
    }

    static std::vector<uint8_t> serializeMigrationOp(const NetworkMigrationOperation& op) {
        std::vector<uint8_t> data;
        wire::putU64(data, op.operationId);
        wire::putU8(data, op.useRdma ? 1 : 0);
        wire::putU64(data, op.destinationAllocId);
        wire::putU8(data, op.completed ? 1 : 0);
        wire::putStr(data, op.errorMessage);
        return data;
    }

    NetworkConfig config_;
    MultiNodePoolManager* poolManager_ = nullptr;
    std::unique_ptr<TcpTransport> transport_;
    std::atomic<bool> running_{false};

    // Handlers
    std::mutex handlerMutex_;
    AllocateHandler allocateHandler_;
    ExportHandler exportHandler_;
    ImportHandler importHandler_;
    MigrateHandler migrateHandler_;
    RegisterHandler registerHandler_;

    // Authorization callback (optional)
    using AuthCallback = std::function<AuthorizationResult(uint32_t messageType)>;
    AuthCallback authCallback_;

    // Cluster view
    mutable std::mutex viewMutex_;
    std::vector<NodeInfo> clusterView_;

    // Stats
    mutable std::mutex statsMutex_;
    ServerStats stats_;
};

std::unique_ptr<ClusterServer> ClusterServer::create(
    const NetworkConfig& config,
    MultiNodePoolManager* poolManager) {
    return std::make_unique<ClusterServerImpl>(config, poolManager);
}

} // namespace network
} // namespace vvm
