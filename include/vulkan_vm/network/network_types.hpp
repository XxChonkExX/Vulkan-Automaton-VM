#pragma once

#include <vulkan/vulkan.h>

#include "vulkan_vm/utils.hpp"

#include <string>
#include <vector>
#include <optional>
#include <chrono>
#include <cstdint>

namespace vvm {
namespace network {

struct NetworkConfig {
    // gRPC server
    std::string listenAddress = "0.0.0.0:50051";
    std::string advertiseAddress = "";  // auto-detect if empty
    
    // NIC selection for RDMA
    std::string nicName = "";           // e.g., "mlx5_0", "mlx5_1", "" = auto
    uint32_t nicPort = 1;               // physical port on NIC
    
    // RDMA settings
    bool enableRdma = true;
    bool enableGpuDirect = true;        // NVIDIA GPUDirect / AMD ROCm RDMA
    bool enableRdmaWrite = true;        // use RDMA_WRITE (vs RDMA_READ)
    uint32_t rdmaMtu = 4096;            // MTU for RDMA (1024/2048/4096)
    
    // Fallback
    bool enableHostStagedFallback = true;
    uint32_t hostStagedThreads = 4;     // threads for host-staged copies
    
    // Timeouts
    std::chrono::milliseconds connectTimeout{5000};
    std::chrono::milliseconds rpcTimeout{30000};
    std::chrono::milliseconds migrationTimeout{60000};
    std::chrono::milliseconds heartbeatInterval{5000};
    std::chrono::milliseconds connectionIdleTimeout{300000};  // 5 min default idle timeout
    
    // Security (optional)
    bool useTls = false;
    std::string tlsCertPath = "";
    std::string tlsKeyPath = "";
    std::string tlsCaPath = "";
    
    // Cluster
    std::vector<std::string> seedNodes;  // initial cluster contacts
    std::string clusterName = "vulkan-automaton";
    
    // Validation
    bool validate() const {
        if (listenAddress.empty()) return false;
        if (rdmaMtu != 1024 && rdmaMtu != 2048 && rdmaMtu != 4096) return false;
        return true;
    }
};

// ============================================================================
// Node identification
// ============================================================================

struct NodeId {
    std::string host;
    uint32_t port = 0;
    uint32_t nodeIndex = 0;
    std::string uuid;  // generated on startup
    
    bool operator==(const NodeId& other) const {
        return host == other.host && port == other.port && nodeIndex == other.nodeIndex;
    }
    
    bool operator!=(const NodeId& other) const { return !(*this == other); }
    
    std::string toString() const {
        return host + ":" + std::to_string(port) + "#" + std::to_string(nodeIndex);
    }
    
    static NodeId fromString(const std::string& str) {
        NodeId id;
        // parse "host:port#index"
        size_t colon = str.find(':');
        size_t hash = str.find('#');
        if (colon != std::string::npos && hash != std::string::npos && hash > colon) {
            id.host = str.substr(0, colon);
            id.port = static_cast<uint32_t>(std::stoul(str.substr(colon + 1, hash - colon - 1)));
            id.nodeIndex = static_cast<uint32_t>(std::stoul(str.substr(hash + 1)));
        }
        return id;
    }
};

// ============================================================================
// Remote allocation descriptor
// ============================================================================

struct RemoteAllocationDesc {
    NodeId owner;
    uint64_t size = 0;
    uint64_t localAllocId = 0;  // internal handle on owner node
    
    // Fast path: GPU-direct RDMA
    bool hasRdmaAddr = false;
    uint64_t rdmaAddr = 0;      // VkRemoteAddressNV (8 bytes)
    uint32_t rkey = 0;          // RDMA remote key
    
    // Fallback: host-staged
    bool hasHostShadow = false;
    
    // Buffer usage for remote import
    VkBufferUsageFlags usageFlags = 0;
    
    // Memory type info for remote import
    uint32_t memoryTypeIndex = UINT32_MAX;
    bool dedicatedAllocation = false;
    
    // External handle type
    vvm::ExternalHandleType handleType = vvm::ExternalHandleType::OpaqueFd;
    
    // Handle data (for staged import when RDMA not available)
    std::vector<uint8_t> externalHandle;  // serialized fd/handle
    
    // Metadata
    uint64_t timestamp = 0;
    std::string allocationName;  // optional debug name
    
    bool isValid() const {
        return size > 0 && (hasRdmaAddr || hasHostShadow || !externalHandle.empty());
    }
    
    bool canUseRdma() const { return hasRdmaAddr && rkey != 0; }
    bool canUseHostStaged() const { return hasHostShadow || !externalHandle.empty(); }
};

// ============================================================================
// Node information for cluster directory
// ============================================================================

struct NodeInfo {
    NodeId id;
    std::vector<std::string> gpuDevices;  // device names
    std::string nicName;                  // e.g., "mlx5_0"
    bool rdmaCapable = false;
    bool gpuDirectCapable = false;
    uint64_t timestamp = 0;  // last heartbeat
    
    // Capabilities
    struct {
        uint64_t totalVram = 0;
        uint64_t availableVram = 0;
        bool supportsRdmaAddr = false;
        bool supportsDmaBuf = false;
        bool supportsOpaqueFd = false;
        bool supportsOpaqueWin32 = false;
        bool supportsD3D12Heap = false;
    } caps;
};

// ============================================================================
// Migration operation (extends local MigrationOperation)
// ============================================================================

struct NetworkMigrationOperation {
    uint64_t operationId = 0;
    RemoteAllocationDesc source;
    uint64_t destinationAllocId = 0;  // local alloc to receive data
    bool useRdma = true;
    uint64_t timeoutNs = UINT64_MAX;
    
    // Progress tracking
    uint64_t bytesTransferred = 0;
    bool completed = false;
    std::string errorMessage;
    
    // Completion
    VkFence completionFence = VK_NULL_HANDLE;
    VkSemaphore signalSemaphore = VK_NULL_HANDLE;
};

// ============================================================================
// Serialization helpers
// ============================================================================

std::vector<uint8_t> serializeNodeId(const NodeId& id);
bool deserializeNodeId(const uint8_t*& p, const uint8_t* end, NodeId& out);

std::vector<uint8_t> serializeNodeInfo(const NodeInfo& info);
bool deserializeNodeInfo(const uint8_t*& p, const uint8_t* end, NodeInfo& out);

std::vector<uint8_t> serializeNodeList(const std::vector<NodeInfo>& list);
bool deserializeNodeList(const std::vector<uint8_t>& data, std::vector<NodeInfo>& out);

std::vector<uint8_t> serializeAllocationDesc(const RemoteAllocationDesc& desc);
bool deserializeAllocationDesc(const uint8_t*& p, const uint8_t* end, RemoteAllocationDesc& out);

} // namespace network
} // namespace vvm