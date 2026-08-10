#pragma once

#include "vulkan_vm/network/network_types.hpp"
#include "vulkan_vm/network/gpu_direct_registration.hpp"
#include <memory>
#include <vector>
#include <optional>
#include <string>
#include <cstdint>
#include <functional>

namespace vvm {
namespace network {

// RDMA listener port convention: each node derives its RDMA listener port from
// its TCP control-plane port (+ this offset). This keeps RDMA and TCP control
// on different port numbers while letting several nodes share one host in
// loopback tests without port collisions.
constexpr uint32_t kRdmaPortOffset = 1;

// ============================================================================
// RDMA Transport Interface
// ============================================================================

struct RdmaMemoryRegion {
    void* addr = nullptr;           // local virtual address
    uint64_t length = 0;            // size in bytes
    uint32_t lkey = 0;              // local key
    uint32_t rkey = 0;              // remote key
    uint64_t rdmaAddr = 0;          // GPU-direct remote address (VkRemoteAddressNV)
    
    // Ownership
    bool ownsMemory = false;        // if true, destructor frees memory
    VkDeviceMemory vkMemory = VK_NULL_HANDLE;  // associated Vulkan memory
    VkBuffer vkBuffer = VK_NULL_HANDLE;        // associated Vulkan buffer
};

struct RdmaConnection {
    std::string remoteHost;
    uint32_t remotePort = 0;
    uint32_t remoteNodeIndex = 0;
    
    // RDMA queue pair info
    uint32_t qpNum = 0;
    uint16_t lid = 0;               // for InfiniBand
    uint32_t qkey = 0;
    uint32_t psn = 0;
    
    // State
    bool connected = false;
    bool gpuDirect = false;
    
    // Internal opaque pointer to rdma_cm_id*, set by transport implementation.
    // Callers must not touch this; it is private to the transport.
    void* internalId_ = nullptr;
    
    // Last activity
    uint64_t lastActivityNs = 0;
};

class RdmaTransport {
public:
    // Factory
    static std::unique_ptr<RdmaTransport> create(
        const NetworkConfig& config,
        VkPhysicalDevice physicalDevice,
        VkDevice device);
    
    virtual ~RdmaTransport() = default;
    
    // Non-copyable, movable
    RdmaTransport(const RdmaTransport&) = delete;
    RdmaTransport& operator=(const RdmaTransport&) = delete;
    RdmaTransport() = default;
    RdmaTransport(RdmaTransport&&) noexcept = default;
    RdmaTransport& operator=(RdmaTransport&&) noexcept = default;
    
    // ========================================================================
    // Lifecycle
    // ========================================================================
    
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual bool isReady() const = 0;
    
    // ========================================================================
    // Memory registration (GPU-direct)
    // ========================================================================
    
    // Register VkDeviceMemory for RDMA access (GPU-direct)
    // Returns remote address and rkey for peer to use
    virtual std::optional<RdmaMemoryRegion> registerGpuMemory(
        VkDeviceMemory memory,
        VkDeviceSize offset,
        VkDeviceSize size,
        VkBuffer buffer = VK_NULL_HANDLE) = 0;
    
    // Register host memory for RDMA (staged fallback)
    virtual std::optional<RdmaMemoryRegion> registerHostMemory(
        void* ptr,
        size_t size) = 0;
    
    // Unregister memory region
    virtual void unregisterMemory(const RdmaMemoryRegion& region) = 0;
    
    // Persistent host memory pinning (GDRCopy-style)
    // Pin memory for repeated RDMA use - avoids repeated registration overhead
    virtual bool pinPersistentHostMemory(void* ptr, size_t size) {
        (void)ptr; (void)size; return false; // Default: not supported
    }
    
    // Release persistently pinned host memory
    virtual void releasePersistentHostMemory(void* ptr) {
        (void)ptr; // Default: not supported
    }
    
    // ========================================================================
    // Connection management
    // ========================================================================
    
    // Connect to remote node (initiates RDMA CM connection)
    virtual std::optional<RdmaConnection> connect(
        const std::string& host,
        uint32_t port,
        uint32_t nodeIndex = 0) = 0;
    
    // Disconnect from remote node
    virtual void disconnect(const RdmaConnection& conn) = 0;
    
    // Get active connections
    virtual std::vector<RdmaConnection> getConnections() const = 0;
    
    // ========================================================================
    // Data transfer
    // ========================================================================
    
    // RDMA_WRITE: write local memory to remote
    virtual bool rdmaWrite(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        uint64_t timeoutNs = UINT64_MAX) = 0;
    
    // RDMA_READ: read remote memory into local
    virtual bool rdmaRead(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        uint64_t timeoutNs = UINT64_MAX) = 0;
    
    // Async operations with completion callback
    using CompletionCallback = std::function<void(bool success, const std::string& error)>;
    
    virtual bool rdmaWriteAsync(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        CompletionCallback callback,
        uint64_t timeoutNs = UINT64_MAX) = 0;
    
    virtual bool rdmaReadAsync(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        CompletionCallback callback,
        uint64_t timeoutNs = UINT64_MAX) = 0;
    
    // ========================================================================
    // Synchronization
    // ========================================================================
    
    // Wait for all pending operations
    virtual void flush() = 0;
    
    // Poll for completions (call periodically if not using callbacks)
    virtual size_t pollCompletions() = 0;
    
    // ========================================================================
    // Capabilities
    // ========================================================================
    
    virtual bool supportsGpuDirect() const = 0;
    virtual bool supportsRdmaWrite() const = 0;
    virtual bool supportsRdmaRead() const = 0;
    virtual std::string getBackendName() const = 0;  // "verbs", "ucx", etc.
    
    // Get NIC info
    virtual std::string getLocalNicName() const = 0;
    virtual uint32_t getLocalPort() const = 0;
    virtual std::string getDeviceGuid() const = 0;
};

} // namespace network
} // namespace vvm