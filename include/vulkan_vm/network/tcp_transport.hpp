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
    // Model registry messages (Hugging Face–style weight distribution)
    MsgModelList      = 11,  // request list of published models
    MsgModelManifest  = 12,  // request/response with model manifest
    MsgModelChunk     = 13,  // request chunk; response carries streamed chunk data
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
};

// Client-side stream I/O for request(): streams a request payload out of
// writeBuffer and/or a response payload into readBuffer, slice by slice.
struct StreamIO {
    const void* writeBuffer = nullptr;  // bytes to send as the request stream
    uint64_t writeLen = 0;
    void* readBuffer = nullptr;         // where to receive the response stream
    uint64_t readLen = 0;
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