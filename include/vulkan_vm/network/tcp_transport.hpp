#pragma once

#include "vulkan_vm/network/network_config.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Forward declare SSL context for TLS support
#if defined(VVM_NETWORK_HAS_TLS)
struct ssl_st;
using SSL_CTX = struct ssl_ctx_st;
using SSL = struct ssl_st;
#endif

namespace vvm {
namespace network {

// ============================================================================
// Control/data plane message types (framed TCP protocol)
// ============================================================================

enum : uint32_t {
    MsgRegisterNode   = 1,
    MsgGetClusterView = 2,
    MsgAllocate       = 3,
    MsgExport         = 4,
    MsgImport         = 5,
    MsgMigratePull    = 6,  // request: read bytes from a remote allocation; response carries stream
    MsgMigratePush    = 7,  // request carries stream; response is an ack
    MsgHeartbeat      = 8,
    MsgLeaveCluster   = 9,
    MsgDeallocate     = 10,  // request: free an allocation by id on its owner
    // Model registry messages (Hugging Face style weight distribution)
    MsgModelList      = 11,  // request list of published models
    MsgModelManifest  = 12,  // request/response with model manifest
    MsgModelChunk     = 13,  // request chunk; response carries streamed chunk data
    // Remote tensor announcement: sender advertises a tensor by name to a
    // peer; the peer stores the descriptor and can later pull the VRAM.
    MsgTensorAnnounce = 14,
    // Remote tensor lookup: request a tensor descriptor by name; response carries
    // the RemoteAllocationDesc so the requester can pull the VRAM.
    MsgTensorLookup   = 15,
};

enum : uint32_t {
    TcpFlagsRequest  = 0,
    TcpFlagsResponse = 1,
    TcpFlagsError    = 2,
};

// One framed message: fixed header + body + optional bulk stream.
struct TcpMessage {
    uint32_t type = 0;
    uint32_t flags = 0;
    uint32_t seq = 0;  // per-connection request sequence number
    std::vector<uint8_t> body;

    // --- Stream (bulk data following the body) ---
    std::vector<uint8_t> stream;
    const void* streamSource = nullptr;
    std::function<void()> streamCleanup;
    uint64_t streamLen = 0;

    // --- Server-side streamed request support (two-phase handler) ---
    bool streamReceived = false;
    void* streamSink = nullptr;
    std::function<void()> streamSinkCleanup;
    void* streamContext = nullptr;  // handler-private state between phases

    // --- Optional windowed streaming (double-buffered pipeline) ---
    // Same contract as StreamIO::onWindowAcquire/onWindowConsumed for
    // inbound request streams, and onWindowProvide for outbound response
    // streams.
    std::function<bool(uint64_t idx, void** dst, uint64_t want, uint64_t& got)> streamWindowAcquire;
    std::function<bool(uint64_t idx, uint64_t len)> streamWindowConsumed;
    std::function<bool(uint64_t idx, const void** src, uint64_t want, uint64_t& got)> streamWindowProvide;
};

// Client-side stream I/O for request(): streams a request payload out of
// writeBuffer and/or a response payload into readBuffer, slice by slice.
// Optional windowed mode (double-buffered pipeline): when the window
// callbacks are set, streaming runs through them instead so the caller can
// overlap GPU copies with network I/O and ride out high-latency links with
// more in-flight data.
struct StreamIO {
    const void* writeBuffer = nullptr;  // bytes to send as the request stream
    uint64_t writeLen = 0;
    void* readBuffer = nullptr;         // where to receive the response stream
    uint64_t readLen = 0;

    // Receive side: acquire() hands the transport a stable destination for
    // the next slice (the caller may block on its own copy fences first so
    // a buffer is only refilled after its previous contents were consumed).
    // consumed() is called once the slice landed, so the caller can submit
    // GPU work on it while later slices are still arriving.
    std::function<bool(uint64_t idx, void** dst, uint64_t want, uint64_t& got)> onWindowAcquire;
    std::function<bool(uint64_t idx, uint64_t len)> onWindowConsumed;

    // Send side: provide() fills the next slice (blocking on GPU copies),
    // so copies of later slices pipeline behind the wire.
    std::function<bool(uint64_t idx, const void** src, uint64_t want, uint64_t& got)> onWindowProvide;
};

// TLS configuration
struct TlsConfig {
    bool enabled = false;
    std::string certPath;
    std::string keyPath;
    std::string caPath;
    bool verifyPeer = true;
    std::string alpnProtocols = "vvm/1.0";
};

// Thread Safety: All public methods are thread-safe.
//                  start()/stop() must not be called concurrently with request().
//                  Request handler must be thread-safe if called concurrently.
class TcpTransport {
public:
    using ConnId = uint64_t;
    using RequestHandler = std::function<void(TcpMessage& request, TcpMessage& response)>;

    TcpTransport();
    ~TcpTransport();

    TcpTransport(const TcpTransport&) = delete;
    TcpTransport& operator=(const TcpTransport&) = delete;
    TcpTransport(TcpTransport&&) noexcept = default;
    TcpTransport& operator=(TcpTransport&&) noexcept = default;

    // ========================================================================
    // Server
    // ========================================================================
    bool start(const std::string& listenHost, uint16_t port, RequestHandler handler,
               const NetworkConfig& netConfig,
               std::chrono::milliseconds idleTimeout = std::chrono::milliseconds(300000));
    void stop();
    bool isRunning() const;
    uint16_t getBoundPort() const;

    // ========================================================================
    // Client
    // ========================================================================
    std::optional<ConnId> connect(const std::string& host, uint16_t port, int32_t timeoutMs = 5000);
    void disconnect(ConnId id);
    bool isConnected(ConnId id) const;

    // Synchronous request/response exchange.
    std::optional<TcpMessage> request(ConnId id, const TcpMessage& req, const StreamIO* streamIO = nullptr);

    // ========================================================================
    // Connection Pool (TCP Stream Striping) - for high-throughput large transfers
    // ========================================================================
    // Creates a pool of parallel connections to a remote endpoint for striped
    // stream transfers. Returns true if poolSize connections were established.
    bool createConnectionPool(const std::string& host, uint16_t port, size_t poolSize = 4);
    
    // Striped stream write: distributes data across pool connections.
    // Use for large outbound transfers (model weights, tensor data).
    bool writeStreamStriped(const std::string& host, uint16_t port, 
                            const void* src, uint64_t len, size_t poolSize = 4);
    
    // Striped stream read: distributes reads across pool connections.
    // Use for large inbound transfers.
    bool readStreamStriped(const std::string& host, uint16_t port,
                           void* dst, uint64_t len, size_t poolSize = 4);
    
    // Shutdown a specific connection pool
    void shutdownConnectionPool(const std::string& host, uint16_t port);
    
    // ========================================================================
    // Dynamic Stripe Scaling (Mathematical Heuristic Matrix)
    // ========================================================================
    // Automatically calculates optimal stripe count based on payload size
    // and hardware concurrency. Overloads that auto-scale:
    
    // Write with auto-scaling: calculates optimal stripes from data size.
    bool writeStreamStripedAuto(const std::string& host, uint16_t port,
                                const void* src, uint64_t len);
    
    // Read with auto-scaling: calculates optimal stripes from data size.
    bool readStreamStripedAuto(const std::string& host, uint16_t port,
                               void* dst, uint64_t len);
    
    // Get the calculated optimal stripe count for a given payload size
    static size_t calculateOptimalStripes(uint64_t dataSize);

    // TLS support
    bool enableTls(const TlsConfig& tlsConfig);
    bool isTlsEnabled() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void acceptLoop();
    void serveConnection(uint64_t connId, uintptr_t socket);
    void cleanupLoop();
};

}  // namespace network
}  // namespace vvm