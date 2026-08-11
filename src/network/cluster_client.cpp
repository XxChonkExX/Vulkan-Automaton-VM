#include "vulkan_vm/network/cluster_client.hpp"
#include "vulkan_vm/network/tcp_transport.hpp"
#include "vulkan_vm/network/network_types.hpp"
#include "vulkan_vm/network/wire_format.hpp"
#include "vulkan_vm/utils.hpp"

#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <future>
#include <mutex>
#include <chrono>
#include <thread>
#include <functional>
#include <unordered_map>

namespace vvm {
namespace network {

// ============================================================================
// TCP-based ClusterClient Implementation
// ============================================================================

class ClusterClientImpl : public ClusterClient {
public:
    ClusterClientImpl(const NetworkConfig& config) : config_(config) {}

    ~ClusterClientImpl() override {
        disconnect();
    }

    bool connect(const std::string& target) override {
        if (transport_ && transport_->isRunning()) return true;

        // Parse target address
        std::string host;
        uint16_t port = 51010; // default control port
        size_t colonPos = target.rfind(':');
        if (colonPos != std::string::npos) {
            host = target.substr(0, colonPos);
            try {
                port = static_cast<uint16_t>(std::stoul(target.substr(colonPos + 1)));
            } catch (...) {
                VVM_LOG_WARN("Invalid port in target {}, using default {}", target, port);
            }
        } else {
            host = target;
        }

        // Create TCP transport for control plane
        transport_ = TcpTransport::create();
        if (!transport_) {
            VVM_LOG_ERROR("Failed to create TCP transport for cluster client");
            return false;
        }

        if (!transport_->start("0.0.0.0", 0, [this](TcpMessage& req, TcpMessage& resp) {
            handleRequest(req, resp);
        })) {
            VVM_LOG_ERROR("Failed to start TCP transport for cluster client");
            return false;
        }

        // Connect to target
        auto conn = transport_->connect(host, port);
        if (!conn) {
            VVM_LOG_ERROR("Failed to connect to cluster at {}:{}", host, port);
            return false;
        }

        controlConnId_ = *conn;
        connected_ = true;

        // Register this node
        registerNode();

        VVM_LOG_INFO("Cluster client connected to {}", target);
        return true;
    }

    void disconnect() override {
        if (!connected_) return;

        // Send leave cluster message
        sendLeaveCluster();

        if (transport_) {
            if (controlConnId_) {
                transport_->disconnect(*controlConnId_);
            }
            transport_->stop();
            transport_.reset();
        }

        connected_ = false;
        controlConnId_ = std::nullopt;
        VVM_LOG_INFO("Cluster client disconnected");
    }

    bool isConnected() const override {
        return connected_;
    }

    // ========================================================================
    // Allocation RPCs
    // ========================================================================

    std::optional<RemoteAllocationDesc> allocateRemote(
        const NodeId& target,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags flags,
        bool enableRdma,
        uint64_t timeoutNs) override {

        if (!connected_) return std::nullopt;

        TcpMessage req;
        req.type = MsgAllocate;
        req.body = serializeAllocateRequest(target, size, usage, flags, enableRdma);

        auto resp = sendRequest(MsgAllocate, req, timeoutNs);
        if (!resp || resp->type != MsgAllocate) return std::nullopt;

        return deserializeAllocateResponse(resp->body);
    }

    std::future<std::optional<RemoteAllocationDesc>> allocateRemoteAsync(
        const NodeId& target,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags flags,
        bool enableRdma) override {

        return std::async(std::launch::async, [this, target, size, usage, flags, enableRdma]() {
            return allocateRemote(target, size, usage, flags, enableRdma, UINT64_MAX);
        });
    }

    // ========================================================================
    // Export/Import RPCs
    // ========================================================================

    std::optional<RemoteAllocationDesc> exportRemote(
        const NodeId& target,
        uint64_t localAllocId,
        bool enableRdma,
        bool forceHostShadow,
        uint64_t timeoutNs) override {

        if (!connected_) return std::nullopt;

        TcpMessage req;
        req.type = MsgExport;
        req.body = serializeExportRequest(localAllocId, enableRdma, forceHostShadow);

        auto resp = sendRequest(MsgExport, req, timeoutNs);
        if (!resp || resp->type != MsgExport) return std::nullopt;

        return deserializeExportResponse(resp->body);
    }

    std::optional<Allocation> importRemote(
        const NodeId& target,
        const RemoteAllocationDesc& desc,
        VkBufferUsageFlags usage,
        uint64_t timeoutNs) override {

        if (!connected_) return std::nullopt;

        TcpMessage req;
        req.type = MsgImport;
        req.body = serializeImportRequest(desc, usage);

        auto resp = sendRequest(MsgImport, req, timeoutNs);
        if (!resp || resp->type != MsgImport) return std::nullopt;

        return deserializeImportResponse(resp->body);
    }

    std::future<std::optional<Allocation>> importRemoteAsync(
        const NodeId& target,
        const RemoteAllocationDesc& desc,
        VkBufferUsageFlags usage) override {

        return std::async(std::launch::async, [this, target, desc, usage]() {
            return importRemote(target, desc, usage, UINT64_MAX);
        });
    }

    // ========================================================================
    // Migration RPCs
    // ========================================================================

    std::optional<NetworkMigrationOperation> migrate(
        const NodeId& target,
        const RemoteAllocationDesc& source,
        uint64_t destinationAllocId,
        bool useRdma,
        uint64_t timeoutNs) override {

        if (!connected_) return std::nullopt;

        TcpMessage req;
        req.type = MsgMigrate;
        req.body = serializeMigrateRequest(source, destinationAllocId, useRdma);

        auto resp = sendRequest(MsgMigrate, req, timeoutNs);
        if (!resp || resp->type != MsgMigrate) return std::nullopt;

        return deserializeMigrateResponse(resp->body);
    }

    // ========================================================================
    // Cluster management
    // ========================================================================

    std::optional<std::vector<NodeInfo>> registerNode(
        const NodeInfo& info,
        uint64_t timeoutNs) override {

        if (!connected_) return std::nullopt;

        TcpMessage req;
        req.type = MsgRegisterNode;
        req.body = serializeNodeInfo(info);

        auto resp = sendRequest(MsgRegisterNode, req, timeoutNs);
        if (!resp || resp->type != MsgRegisterNode) return std::nullopt;

        return deserializeNodeList(resp->body);
    }

    std::optional<std::vector<NodeInfo>> heartbeat(
        const NodeId& node,
        uint64_t timeoutNs) override {

        if (!connected_) return std::nullopt;

        TcpMessage req;
        req.type = MsgHeartbeat;
        req.body = serializeHeartbeat(node);

        auto resp = sendRequest(MsgHeartbeat, req, timeoutNs);
        if (!resp || resp->type != MsgHeartbeat) return std::nullopt;

        return deserializeNodeList(resp->body);
    }

    std::optional<std::vector<NodeInfo>> getClusterView(
        uint64_t timeoutNs) override {

        if (!connected_) return std::nullopt;

        TcpMessage req;
        req.type = MsgGetClusterView;

        auto resp = sendRequest(MsgGetClusterView, req, timeoutNs);
        if (!resp || resp->type != MsgGetClusterView) return std::nullopt;

        return deserializeNodeList(resp->body);
    }

    // ========================================================================
    // Streaming
    // ========================================================================

    bool startMigrationStream(
        const NodeId& target,
        std::function<void(const NetworkMigrationOperation&)> progressCallback) override {

        if (!connected_) return false;
        streamCallback_ = std::move(progressCallback);
        // In a full implementation, this would set up a streaming connection
        VVM_LOG_WARN("Migration streaming not yet fully implemented");
        return true;
    }

    void stopMigrationStream() override {
        streamCallback_ = nullptr;
    }

private:
    NetworkConfig config_;
    std::unique_ptr<TcpTransport> transport_;
    std::optional<TcpTransport::ConnId> controlConnId_;
    std::atomic<bool> connected_{false};
    std::function<void(const NetworkMigrationOperation&)> streamCallback_;
    std::mutex callbackMutex_;

    // Helper methods
    void registerNode() {
        NodeInfo info;
        info.id = NodeId{config_.advertiseAddress.empty() ? "127.0.0.1" : config_.advertiseAddress,
                         static_cast<uint16_t>(config_.listenAddress.find(':') != std::string::npos ?
                                             std::stoul(config_.listenAddress.substr(config_.listenAddress.rfind(':') + 1)) : 51010),
                         0, ""};
        info.gpuDevices = enumerateGpuDevices();
        info.nicName = config_.nicName;
        info.rdmaCapable = config_.enableRdma;
        info.gpuDirectCapable = config_.enableGpuDirect;
        info.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        registerNode(info);
    }

    std::vector<std::string> enumerateGpuDevices() {
        std::vector<std::string> devices;
        VkInstance instance = nullptr;
        VkInstanceCreateInfo instInfo{};
        instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS || !instance) {
            devices.push_back("GPU0");
            return devices;
        }
        uint32_t count = 0;
        if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) {
            vkDestroyInstance(instance, nullptr);
            devices.push_back("GPU0");
            return devices;
        }
        std::vector<VkPhysicalDevice> phys(count);
        vkEnumeratePhysicalDevices(instance, &count, phys.data());
        for (uint32_t i = 0; i < count; ++i) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(phys[i], &props);
            devices.push_back(std::string("GPU") + std::to_string(i) + ":" + props.deviceName);
        }
        vkDestroyInstance(instance, nullptr);
        return devices;
    }

    void sendLeaveCluster() {
        if (!connected_ || !controlConnId_) return;
        TcpMessage req;
        req.type = MsgLeaveCluster;
        sendRequest(MsgLeaveCluster, req, 5000000000); // 5 second timeout
    }

    std::optional<TcpMessage> sendRequest(uint32_t type, const TcpMessage& req, uint64_t timeoutNs) {
        if (!connected_ || !controlConnId_) return std::nullopt;

        if (!transport_) return std::nullopt;

        auto resp = transport_->request(*controlConnId_, type, req.body,
                                        static_cast<uint32_t>(timeoutNs / 1000000));
        if (!resp) return std::nullopt;

        TcpMessage out;
        out.type = type;
        out.body = std::move(*resp);
        return out;
    }

    // Serialization helpers (wire-format big-endian, bounds-checked)
    std::vector<uint8_t> serializeAllocateRequest(const NodeId& target, VkDeviceSize size,
                                                  VkBufferUsageFlags usage, VkMemoryPropertyFlags flags,
                                                  bool enableRdma) {
        std::vector<uint8_t> data;
        wire::putStr(data, target.host);
        wire::putU32(data, target.port);
        wire::putU32(data, target.nodeIndex);
        wire::putStr(data, target.uuid);
        wire::putU64(data, static_cast<uint64_t>(size));
        wire::putU64(data, static_cast<uint64_t>(usage));
        wire::putU64(data, static_cast<uint64_t>(flags));
        wire::putU8(data, enableRdma ? 1 : 0);
        return data;
    }

    std::optional<RemoteAllocationDesc> deserializeAllocateResponse(const std::vector<uint8_t>& data) {
        RemoteAllocationDesc desc;
        const uint8_t* p = data.data();
        const uint8_t* end = p + data.size();
        if (!wire::getStr(p, end, desc.owner.host)) return std::nullopt;
        uint32_t port = 0;
        if (!wire::getU32(p, end, port)) return std::nullopt;
        desc.owner.port = port;
        if (!wire::getU32(p, end, desc.owner.nodeIndex)) return std::nullopt;
        if (!wire::getStr(p, end, desc.owner.uuid)) return std::nullopt;
        uint64_t size = 0;
        if (!wire::getU64(p, end, size)) return std::nullopt;
        desc.size = size;
        uint64_t flagsVal = 0;
        if (!wire::getU64(p, end, flagsVal)) return std::nullopt;
        desc.usageFlags = static_cast<VkBufferUsageFlags>(flagsVal);
        uint8_t rdma = 0;
        if (!wire::getU8(p, end, rdma)) return std::nullopt;
        desc.hasRdmaAddr = (rdma != 0);
        if (desc.hasRdmaAddr) {
            if (!wire::getU64(p, end, desc.rdmaAddr)) return std::nullopt;
            if (!wire::getU32(p, end, desc.rkey)) return std::nullopt;
        }
        return desc;
    }

    std::vector<uint8_t> serializeExportRequest(uint64_t localAllocId, bool enableRdma, bool forceHostShadow) {
        std::vector<uint8_t> data;
        wire::putU64(data, localAllocId);
        wire::putU8(data, enableRdma ? 1 : 0);
        wire::putU8(data, forceHostShadow ? 1 : 0);
        return data;
    }

    std::optional<RemoteAllocationDesc> deserializeExportResponse(const std::vector<uint8_t>& data) {
        return deserializeAllocateResponse(data);
    }

    std::optional<Allocation> deserializeImportResponse(const std::vector<uint8_t>& data) {
        const uint8_t* p = data.data();
        const uint8_t* end = p + data.size();
        uint8_t ok = 0;
        if (!wire::getU8(p, end, ok)) return std::nullopt;
        if (ok == 0) return std::nullopt;
        Allocation alloc;
        uint64_t size = 0;
        if (!wire::getU64(p, end, size)) return std::nullopt;
        alloc.size = size;
        uint64_t buffer64 = 0;
        if (!wire::getU64(p, end, buffer64)) return std::nullopt;
        alloc.buffer = reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(buffer64));
        uint64_t memory64 = 0;
        if (!wire::getU64(p, end, memory64)) return std::nullopt;
        alloc.memory = reinterpret_cast<VkDeviceMemory>(static_cast<uintptr_t>(memory64));
        uint64_t hostPtr64 = 0;
        if (!wire::getU64(p, end, hostPtr64)) return std::nullopt;
        alloc.hostPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(hostPtr64));
        uint64_t devAddr64 = 0;
        if (!wire::getU64(p, end, devAddr64)) return std::nullopt;
        alloc.deviceAddress = devAddr64;
        uint64_t offset = 0;
        if (!wire::getU64(p, end, offset)) return std::nullopt;
        alloc.offset = offset;
        uint64_t blockIdx = 0;
        if (!wire::getU64(p, end, blockIdx)) return std::nullopt;
        alloc.blockIndex = static_cast<uint32_t>(blockIdx);
        uint64_t usageVal = 0;
        if (!wire::getU64(p, end, usageVal)) return std::nullopt;
        alloc.usage = static_cast<VkBufferUsageFlags>(usageVal);
        uint8_t mapped = 0;
        if (!wire::getU8(p, end, mapped)) return std::nullopt;
        alloc.mapped = (mapped != 0);
        if (!wire::getStr(p, end, alloc.name)) return std::nullopt;
        return alloc;
    }

    std::vector<uint8_t> serializeImportRequest(const RemoteAllocationDesc& desc, VkBufferUsageFlags usage) {
        std::vector<uint8_t> data;
        wire::putStr(data, desc.owner.host);
        wire::putU32(data, desc.owner.port);
        wire::putU32(data, desc.owner.nodeIndex);
        wire::putStr(data, desc.owner.uuid);
        wire::putU64(data, desc.size);
        wire::putU64(data, static_cast<uint64_t>(usage));
        wire::putU64(data, static_cast<uint64_t>(desc.usageFlags));
        wire::putU8(data, desc.hasRdmaAddr ? 1 : 0);
        if (desc.hasRdmaAddr) {
            wire::putU64(data, desc.rdmaAddr);
            wire::putU32(data, desc.rkey);
        }
        wire::putU8(data, desc.hasUcxAddr ? 1 : 0);
        if (desc.hasUcxAddr) {
            wire::putBytes(data, desc.ucxWorkerAddr);
            wire::putBytes(data, desc.ucxPackedRkey);
            wire::putU64(data, desc.ucxRemoteAddr);
            wire::putU32(data, desc.ucxDeviceIndex);
        }
        wire::putU8(data, desc.hasHostShadow ? 1 : 0);
        wire::putBytes(data, desc.externalHandle);
        wire::putStr(data, desc.allocationName);
        return data;
    }

    std::vector<uint8_t> serializeMigrateRequest(const RemoteAllocationDesc& source,
                                                 uint64_t destinationAllocId, bool useRdma) {
        std::vector<uint8_t> data;
        wire::putStr(data, source.owner.host);
        wire::putU32(data, source.owner.port);
        wire::putU32(data, source.owner.nodeIndex);
        wire::putStr(data, source.owner.uuid);
        wire::putU64(data, source.size);
        wire::putU64(data, source.localAllocId);
        wire::putU8(data, source.hasRdmaAddr ? 1 : 0);
        if (source.hasRdmaAddr) {
            wire::putU64(data, source.rdmaAddr);
            wire::putU32(data, source.rkey);
        }
        wire::putU64(data, destinationAllocId);
        wire::putU8(data, useRdma ? 1 : 0);
        return data;
    }

    std::optional<NetworkMigrationOperation> deserializeMigrateResponse(const std::vector<uint8_t>& data) {
        NetworkMigrationOperation op;
        const uint8_t* p = data.data();
        const uint8_t* end = p + data.size();
        uint64_t opId = 0;
        if (!wire::getU64(p, end, opId)) return std::nullopt;
        op.operationId = opId;
        uint8_t useRdma = 0;
        if (!wire::getU8(p, end, useRdma)) return std::nullopt;
        op.useRdma = (useRdma != 0);
        uint64_t dstId = 0;
        if (!wire::getU64(p, end, dstId)) return std::nullopt;
        op.destinationAllocId = dstId;
        uint8_t completed = 0;
        if (!wire::getU8(p, end, completed)) return std::nullopt;
        op.completed = (completed != 0);
        if (!wire::getStr(p, end, op.errorMessage)) return std::nullopt;
        return op;
    }

    std::vector<uint8_t> serializeNodeInfo(const NodeInfo& info) {
        std::vector<uint8_t> data;
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
        return data;
    }

    std::optional<std::vector<NodeInfo>> deserializeNodeList(const std::vector<uint8_t>& data) {
        std::vector<NodeInfo> list;
        const uint8_t* p = data.data();
        const uint8_t* end = p + data.size();
        uint32_t count = 0;
        if (!wire::getU32(p, end, count)) return std::nullopt;
        for (uint32_t i = 0; i < count; ++i) {
            NodeInfo info;
            if (!wire::getStr(p, end, info.id.host)) return std::nullopt;
            uint32_t port = 0;
            if (!wire::getU32(p, end, port)) return std::nullopt;
            info.id.port = port;
            if (!wire::getU32(p, end, info.id.nodeIndex)) return std::nullopt;
            if (!wire::getStr(p, end, info.id.uuid)) return std::nullopt;
            uint32_t devCount = 0;
            if (!wire::getU32(p, end, devCount)) return std::nullopt;
            info.gpuDevices.resize(devCount);
            for (uint32_t d = 0; d < devCount; ++d) {
                if (!wire::getStr(p, end, info.gpuDevices[d])) return std::nullopt;
            }
            if (!wire::getStr(p, end, info.nicName)) return std::nullopt;
            uint8_t v = 0;
            if (!wire::getU8(p, end, v)) return std::nullopt;
            info.rdmaCapable = (v != 0);
            if (!wire::getU8(p, end, v)) return std::nullopt;
            info.gpuDirectCapable = (v != 0);
            if (!wire::getU64(p, end, info.timestamp)) return std::nullopt;
            if (!wire::getU64(p, end, info.caps.totalVram)) return std::nullopt;
            if (!wire::getU64(p, end, info.caps.availableVram)) return std::nullopt;
            if (!wire::getU8(p, end, v)) return std::nullopt;
            info.caps.supportsRdmaAddr = (v != 0);
            if (!wire::getU8(p, end, v)) return std::nullopt;
            info.caps.supportsDmaBuf = (v != 0);
            if (!wire::getU8(p, end, v)) return std::nullopt;
            info.caps.supportsOpaqueFd = (v != 0);
            if (!wire::getU8(p, end, v)) return std::nullopt;
            info.caps.supportsOpaqueWin32 = (v != 0);
            if (!wire::getU8(p, end, v)) return std::nullopt;
            info.caps.supportsD3D12Heap = (v != 0);
            list.push_back(std::move(info));
        }
        return list;
    }

    std::vector<uint8_t> serializeHeartbeat(const NodeId& node) {
        std::vector<uint8_t> data;
        wire::putStr(data, node.host);
        wire::putU32(data, node.port);
        wire::putU32(data, node.nodeIndex);
        wire::putStr(data, node.uuid);
        return data;
    }
};

std::unique_ptr<ClusterClient> ClusterClient::create(const NetworkConfig& config) {
    return std::make_unique<ClusterClientImpl>(config);
}

} // namespace network
} // namespace vvm