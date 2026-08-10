#pragma once

// UCX (Unified Communication X) transport backend for VulkanVM
// 
// Based on UCX (https://github.com/openucx/ucx) - Unified Communication X
// UCX is licensed under BSD-3-Clause
// 
// This implementation uses UCX UCP (User Communication Protocol) for
// high-performance GPU-aware communication over InfiniBand, RoCE, TCP,
// and shared memory.
// 
// Credit: UCX team (Pavel Shamis, Yossi Itigin, et al.) and contributors.

#ifndef VVM_UCX_TRANSPORT_HPP
#define VVM_UCX_TRANSPORT_HPP

#include "vulkan_vm/tensor_transport.hpp"

#if defined(VVM_HAS_UCX)
#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <ucs/type/status.h>
#endif

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <unordered_map>

namespace vvm {
namespace tensor {

#if defined(VVM_HAS_UCX)

// ============================================================================
// UCX Transport Configuration
// ============================================================================

struct UcxTransportConfig {
    // UCX worker thread count
    uint32_t workerThreadCount = 1;
    
    // Enable RNDV (rendezvous) protocol for large messages
    bool enableRndv = true;
    
    // RNDV threshold (bytes)
    size_t rndvThreshold = 8192;
    
    // Enable GPU memory registration (requires CUDA/ROCm/Ze support in UCX)
    bool enableGpuMemory = true;
    
    // UCX TLS (transport layer selection) - empty = auto
    std::string tls = "";  // e.g., "rc,ud,sm,tcp"
    
    // Network device filter
    std::string netDevices = "";  // e.g., "mlx5_0:1,mlx5_1:1"
    
    // Enable CUDA IPC for intra-node GPU copies
    bool enableCudaIpc = true;
};

// ============================================================================
// UCX Endpoint Wrapper
// ============================================================================

struct UcxEndpoint {
    ucp_ep_h ep = nullptr;
    std::string remoteAddress;
    uint32_t remoteNodeId = 0;
    bool connected = false;
};

// ============================================================================
// UCX Memory Handle (for GPU memory registration)
// ============================================================================

struct UcxMemoryHandle {
    ucp_mem_h memh = nullptr;
    void* ptr = nullptr;
    size_t size = 0;
    bool isGpuMemory = false;
    uint32_t deviceIndex = 0;
};

// ============================================================================
// UCX Transport Class
// ============================================================================

class UcxTransport {
public:
    UcxTransport() = default;
    ~UcxTransport();
    
    // Non-copyable, movable
    UcxTransport(const UcxTransport&) = delete;
    UcxTransport& operator=(const UcxTransport&) = delete;
    UcxTransport(UcxTransport&&) noexcept = default;
    UcxTransport& operator=(UcxTransport&&) noexcept = default;
    
    // Initialize UCX context and worker
    bool initialize(const UcxTransportConfig& config);
    
    // Shutdown UCX
    void shutdown();
    
    bool isInitialized() const { return context_ != nullptr; }
    
    // Register GPU memory with UCX (for zero-copy RDMA)
    // Returns a handle that can be used for ucp_put/ucp_get
    std::optional<UcxMemoryHandle> registerGpuMemory(
        void* ptr, size_t size, uint32_t deviceIndex);
    
    // Register host memory
    std::optional<UcxMemoryHandle> registerHostMemory(
        void* ptr, size_t size);
    
    // Deregister memory
    void deregisterMemory(const UcxMemoryHandle& handle);
    
    // Create endpoint to remote address
    std::optional<UcxEndpoint> createEndpoint(const std::string& remoteAddress);
    
    // Close endpoint
    void closeEndpoint(UcxEndpoint& endpoint);
    
    // Async put (RDMA write) - GPU memory to remote GPU memory
    bool putAsync(const UcxEndpoint& endpoint,
                  const UcxMemoryHandle& localMem,
                  const UcxMemoryHandle& remoteMem,
                  size_t size,
                  std::function<void(bool)> callback);
    
    // Async get (RDMA read) - remote GPU memory to local GPU memory
    bool getAsync(const UcxEndpoint& endpoint,
                  const UcxMemoryHandle& localMem,
                  const UcxMemoryHandle& remoteMem,
                  size_t size,
                  std::function<void(bool)> callback);
    
    // Tagged send/recv for control messages
    bool tagSendAsync(const UcxEndpoint& endpoint,
                      const void* buffer, size_t size,
                      uint64_t tag,
                      std::function<void(bool)> callback);
    
    bool tagRecvAsync(const UcxEndpoint& endpoint,
                      void* buffer, size_t size,
                      uint64_t tag,
                      std::function<void(bool)> callback);
    
    // Progress UCX worker (call periodically or from dedicated thread)
    void progress();
    
    // Get UCX context for advanced usage
    ucp_context_h getContext() const { return context_; }
    ucp_worker_h getWorker() const { return worker_; }
    
private:
    ucp_context_h context_ = nullptr;
    ucp_worker_h worker_ = nullptr;
    UcxTransportConfig config_;
    bool initialized_ = false;
    
    // Memory handle tracking
    std::mutex memHandlesMutex_;
    std::unordered_map<uintptr_t, UcxMemoryHandle> memHandles_;
    
    // Endpoint tracking
    std::mutex endpointsMutex_;
    std::unordered_map<std::string, UcxEndpoint> endpoints_;
    
    // Callback context for async operations
    struct RequestContext {
        std::function<void(bool)> callback;
    };
    
    static void ucpSendCallback(void* request, ucs_status_t status, void* userData);
    static void ucpRecvCallback(void* request, ucs_status_t status, 
                                ucp_tag_recv_info_t* info, void* userData);
    static void ucpRmaCallback(void* request, ucs_status_t status, void* userData);
};

#else // VVM_HAS_UCX

// Stub when UCX not available
class UcxTransport {
public:
    UcxTransport() = default;
    ~UcxTransport() = default;
    
    bool initialize(const UcxTransportConfig&) { return false; }
    void shutdown() {}
    bool isInitialized() const { return false; }
    
    struct UcxMemoryHandle {};
    struct UcxEndpoint {};
    struct UcxTransportConfig {};
    
    std::optional<UcxMemoryHandle> registerGpuMemory(void*, size_t, uint32_t) { return std::nullopt; }
    std::optional<UcxMemoryHandle> registerHostMemory(void*, size_t) { return std::nullopt; }
    void deregisterMemory(const UcxMemoryHandle&) {}
    std::optional<UcxEndpoint> createEndpoint(const std::string&) { return std::nullopt; }
    void closeEndpoint(UcxEndpoint&) {}
    bool putAsync(const UcxEndpoint&, const UcxMemoryHandle&, const UcxMemoryHandle&, size_t, std::function<void(bool)>) { return false; }
    bool getAsync(const UcxEndpoint&, const UcxMemoryHandle&, const UcxMemoryHandle&, size_t, std::function<void(bool)>) { return false; }
    bool tagSendAsync(const UcxEndpoint&, const void*, size_t, uint64_t, std::function<void(bool)>) { return false; }
    bool tagRecvAsync(const UcxEndpoint&, void*, size_t, uint64_t, std::function<void(bool)>) { return false; }
    void progress() {}
    
    void* getContext() const { return nullptr; }
    void* getWorker() const { return nullptr; }
};

#endif // VVM_HAS_UCX

} // namespace tensor
} // namespace vvm

#endif // VVM_UCX_TRANSPORT_HPP