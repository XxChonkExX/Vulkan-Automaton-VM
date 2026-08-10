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
// UCX Worker Address
// ============================================================================
// Opaque UCX worker address (result of ucp_worker_get_address). Serialize with
// wire::putBytes / getBytes (see wire_format.hpp) over the TCP control plane.
// UCX worker addresses are opaque blobs, not host:port strings; IB/RoCE/TCP
// selection is decided later by UCX TLS, not by the address string.

struct UcxWorkerAddress {
    std::vector<uint8_t> bytes;

    bool empty() const { return bytes.empty(); }
    size_t size() const { return bytes.size(); }
    const void* data() const { return bytes.data(); }
};

// ============================================================================
// UCX Endpoint Wrapper
// ============================================================================
// peerKey is the caller's stable id for caching the endpoint (e.g. "host:port"
// or a cluster node id). It is NOT a UCX address; the UCX address is exchanged
// separately and consumed by connectToAddress().

struct UcxEndpoint {
#if defined(VVM_HAS_UCX)
    ucp_ep_h ep = nullptr;
#endif
    std::string peerKey;
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

    // RMA support: packed remote key and remote virtual address.
    // Filled by packRkey() on the owner, consumed by unpackRkey() on the peer.
    std::vector<uint8_t> packedRkey;  // result of ucp_rkey_pack
    uint64_t remoteAddr = 0;          // remote virtual address (ptr on peer)
    bool rkeyValid = false;           // whether packedRkey/remoteAddr are set
};

// RMA key exchange result: contains packed rkey + remote address for a memory region.
struct UcxRmaKey {
    std::vector<uint8_t> packedRkey;
    uint64_t remoteAddr = 0;
    size_t size = 0;
    uint32_t deviceIndex = 0;  // for GPU memory context
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
    
    // ========================================================================
    // Endpoint / connection surface
    // ------------------------------------------------------------------------
    // UCX endpoints are created from an opaque worker address blob obtained
    // via getLocalAddress() (ucp_worker_get_address). The bootstrap exchange
    // is the caller's responsibility; exchangeAndConnect() provides a
    // convenience wrapper that swaps blobs over any byte pipe send/recv pair
    // (typically the existing TCP control plane via wire_format helpers).
    // ========================================================================

    // Local worker address for bootstrap. Call after initialize(). Caller
    // sends `bytes` to peers over the TCP control plane. Returns nullopt
    // if UCX is not initialized or the address cannot be queried.
    std::optional<UcxWorkerAddress> getLocalAddress();

    // Create an endpoint from a peer's worker-address blob (already exchanged
    // via the control plane). peerKey is your stable id for caching
    // ("host:port" or node id) -- it is NOT passed to UCX. If an endpoint
    // already exists for peerKey and is connected, it is returned directly.
    std::optional<UcxEndpoint> connectToAddress(
        const UcxWorkerAddress& peerAddress,
        const std::string& peerKey,
        uint32_t remoteNodeId = 0);

    // Convenience: mutual exchange over an already-connected byte pipe.
    // sendFn transfers exactly one length-prefixed blob; recvFn reads one.
    // Use wire::putBytes / getBytes on the TCP session.
    // activeSide true: send local addr first, then receive peer.
    // activeSide false: receive peer first, then send local.
    // Exactly one side per peer pair must be active so the two blobs do not
    // deadlock (active sends first).
    std::optional<UcxEndpoint> exchangeAndConnect(
        const std::string& peerKey,
        uint32_t remoteNodeId,
        bool activeSide,
        const std::function<bool(const std::vector<uint8_t>&)>& sendFn,
        const std::function<bool(std::vector<uint8_t>&)>& recvFn);

    // Close an endpoint and drop it from the cache. Non-blocking close +
    // worker progress; safe to call from anywhere that can drive progress().
    void closeEndpoint(UcxEndpoint& endpoint);

    // Look up a cached connected endpoint by peerKey. Does not create one.
    std::optional<UcxEndpoint> getEndpoint(const std::string& peerKey) const;

    // ========================================================================
    // RMA (Remote Memory Access) support
    // ------------------------------------------------------------------------
    // UCX RMA (ucp_put/ucp_get) requires a remote virtual address + rkey,
    // NOT a local ucp_mem_h. The owner must pack the rkey, exchange it with
    // the remote virtual address over the control plane, and the peer unpacks it.
    // ========================================================================

    // Pack an RMA key for a registered memory handle. Returns the packed rkey
    // blob and the remote virtual address (ptr). Call on the OWNER of the memory.
    // The returned UcxRmaKey is sent to the peer over the control plane.
    std::optional<UcxRmaKey> packRmaKey(const UcxMemoryHandle& handle);

    // Unpack an RMA key received from a peer. Returns a handle with the
    // unpacked rkey and remote address, ready for ucp_put/ucp_get.
    // Call on the PEER that will initiate the RMA operation.
    std::optional<UcxMemoryHandle> unpackRmaKey(
        const UcxEndpoint& endpoint,
        const UcxRmaKey& rmaKey);

    // Convenience: exchange RMA keys for a memory handle over an existing
    // connected byte pipe (e.g., TCP control plane). ownerSide=true means
    // we own the memory and send our key; ownerSide=false means we receive
    // the peer's key.
    std::optional<UcxMemoryHandle> exchangeRmaKey(
        const UcxEndpoint& endpoint,
        const UcxMemoryHandle& localHandle,
        bool ownerSide,
        const std::function<bool(const UcxRmaKey&)>& sendFn,
        const std::function<bool(UcxRmaKey&)>& recvFn);

    // Async put (RDMA write) - local GPU memory to remote GPU memory
    // Requires remoteMem to have valid rkeyValid=true (from unpackRmaKey).
    bool putAsync(const UcxEndpoint& endpoint,
                  const UcxMemoryHandle& localMem,
                  const UcxMemoryHandle& remoteMem,
                  size_t size,
                  std::function<void(bool)> callback);

    // Async get (RDMA read) - remote GPU memory to local GPU memory
    // Requires remoteMem to have valid rkeyValid=true (from unpackRmaKey).
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
    // MUST be called regularly to advance async operations (tag, RMA, AM).
    // The caller is responsible for driving progress: either from a dedicated
    // worker thread (polling with ucp_worker_progress) or by calling progress()
    // from the application's event loop. Failure to call progress() will stall
    // all async operations indefinitely.
    // Thread-safe: can be called concurrently with async operations.
    void progress();

    // Error model: UCX failures return false from async methods and log errors.
    // Callbacks are always invoked exactly once (on success or failure).
    // For detailed error status, enable UCX_LOG_LEVEL=info/debug in environment.
    // The epErrorCallback fires asynchronously on endpoint errors; callers
    // should check endpoint.connected or use getEndpoint() to detect failures.
    
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
    
    // Cached local worker address (refreshed in getLocalAddress())
    UcxWorkerAddress localAddress_;
    bool localAddressValid_ = false;

    // Hard cap on exchanged worker-address blobs (defends control-plane
    // allocs against malformed/oversized peer addresses). UCX worker
    // addresses are normally a few hundred bytes; 16 KiB is generous.
    static constexpr size_t kMaxUcxAddrBytes = 16 * 1024;
    
    // Callback context for async operations
    struct RequestContext {
        std::function<void(bool)> callback;
    };
    
    static void ucpSendCallback(void* request, ucs_status_t status, void* userData);
    static void ucpRecvCallback(void* request, ucs_status_t status, 
                                ucp_tag_recv_info_t* info, void* userData);
    static void ucpRmaCallback(void* request, ucs_status_t status, void* userData);

    // UCX endpoint error handler (peer reset / connection failure).
    static void epErrorCallback(void* arg, ucp_ep_h ep, ucs_status_t status);
};

#else // VVM_HAS_UCX

// Stub when UCX not available
struct UcxWorkerAddress {
    std::vector<uint8_t> bytes;
    bool empty() const { return bytes.empty(); }
    size_t size() const { return bytes.size(); }
    const void* data() const { return bytes.data(); }
};

struct UcxRmaKey {
    std::vector<uint8_t> packedRkey;
    uint64_t remoteAddr = 0;
    size_t size = 0;
    uint32_t deviceIndex = 0;
};

class UcxTransport {
public:
    UcxTransport() = default;
    ~UcxTransport() = default;
    
    bool initialize(const UcxTransportConfig&) { return false; }
    void shutdown() {}
    bool isInitialized() const { return false; }
    
    struct UcxMemoryHandle {
        std::vector<uint8_t> packedRkey;
        uint64_t remoteAddr = 0;
        bool rkeyValid = false;
    };
    struct UcxEndpoint {
        std::string peerKey;
        uint32_t remoteNodeId = 0;
        bool connected = false;
    };
    struct UcxTransportConfig {};
    
    std::optional<UcxMemoryHandle> registerGpuMemory(void*, size_t, uint32_t) { return std::nullopt; }
    std::optional<UcxMemoryHandle> registerHostMemory(void*, size_t) { return std::nullopt; }
    void deregisterMemory(const UcxMemoryHandle&) {}

    // Stub connection surface: every call returns empty / nullopt so callers
    // that gate on isInitialized() never reach these.
    std::optional<UcxWorkerAddress> getLocalAddress() { return std::nullopt; }
    std::optional<UcxEndpoint> connectToAddress(const UcxWorkerAddress&,
                                                 const std::string&,
                                                 uint32_t) { return std::nullopt; }
    std::optional<UcxEndpoint> exchangeAndConnect(const std::string&,
                                                   uint32_t,
                                                   bool,
                                                   const std::function<bool(const std::vector<uint8_t>&)>&,
                                                   const std::function<bool(std::vector<uint8_t>&)>&) { return std::nullopt; }
    void closeEndpoint(UcxEndpoint&) {}
    std::optional<UcxEndpoint> getEndpoint(const std::string&) const { return std::nullopt; }

    // Stub RMA surface
    std::optional<UcxRmaKey> packRmaKey(const UcxMemoryHandle&) { return std::nullopt; }
    std::optional<UcxMemoryHandle> unpackRmaKey(const UcxEndpoint&, const UcxRmaKey&) { return std::nullopt; }
    std::optional<UcxMemoryHandle> exchangeRmaKey(const UcxEndpoint&,
                                                   const UcxMemoryHandle&,
                                                   bool,
                                                   const std::function<bool(const UcxRmaKey&)>&,
                                                   std::function<bool(UcxRmaKey&)>&) { return std::nullopt; }
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