#pragma once

#include "vulkan_vm/network/network_config.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
};

enum : uint32_t {
    TcpFlagsRequest  = 0,
    TcpFlagsResponse = 1,
    TcpFlagsError    = 2,
};

// One framed message: fixed header + body + optional bulk stream.
//
// Protocol framing (Spark-inspired: versioned header, sequence numbers,
// slice-based streaming with zero intermediate copies):
//   [magic "VVMN"][u8 version=1][u8 x3 reserved][u32 type][u32 flags]
//   [u32 bodyLen][u32 seq][u64 streamLen]  -> 32-byte header
//   then body (bodyLen bytes), then stream (streamLen bytes, optional).
// Streams are transferred in fixed-size slices directly between the socket
// and the caller-provided buffer; nothing is materialized in an extra copy.
struct TcpMessage {
    uint32_t type = 0;
    uint32_t flags = 0;
    uint32_t seq = 0;  // per-connection request sequence number
    std::vector<uint8_t> body;

    // --- Stream (bulk data following the body) ---

    // Buffered mode: transport reads/writes the whole stream into this vector.
    std::vector<uint8_t> stream;

    // Slice mode (server side, response): contiguous source bytes for the
    // response stream. The transport streams it out in slices AFTER the
    // handler returns. streamCleanup is invoked when done (to free staging).
    const void* streamSource = nullptr;
    std::function<void()> streamCleanup;

    // Total stream length (slice mode).
    uint64_t streamLen = 0;

    // --- Server-side streamed request support (two-phase handler) ---
    // When a request arrives with a stream, the transport invokes the handler
    // TWICE: first in "prepare" phase (streamReceived == false) so the handler
    // can allocate a sink buffer, then in "finalize" phase (streamReceived ==
    // true) once the stream bytes are in streamSink. Set streamSink in the
    // prepare phase and streamSinkCleanup to free the buffer afterwards.
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

// ============================================================================
// Little-endian serialization helpers
// ============================================================================

namespace detail {

inline void putU8(std::vector<uint8_t>& b, uint8_t v) { b.push_back(v); }

inline void putU16(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back(static_cast<uint8_t>(v & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

inline void putU32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

inline void putU64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

inline void putStr(std::vector<uint8_t>& b, const std::string& s) {
    putU32(b, static_cast<uint32_t>(s.size()));
    b.insert(b.end(), s.begin(), s.end());
}

inline void putBytes(std::vector<uint8_t>& b, const std::vector<uint8_t>& data) {
    putU32(b, static_cast<uint32_t>(data.size()));
    b.insert(b.end(), data.begin(), data.end());
}

inline bool getU8(const uint8_t*& p, const uint8_t* end, uint8_t& out) {
    if (p >= end) return false;
    out = *p++;
    return true;
}

inline bool getU16(const uint8_t*& p, const uint8_t* end, uint16_t& out) {
    if (p + 2 > end) return false;
    out = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    p += 2;
    return true;
}

inline bool getU32(const uint8_t*& p, const uint8_t* end, uint32_t& out) {
    if (p + 4 > end) return false;
    out = 0;
    for (int i = 0; i < 4; ++i) out |= static_cast<uint32_t>(p[i]) << (8 * i);
    p += 4;
    return true;
}

inline bool getU64(const uint8_t*& p, const uint8_t* end, uint64_t& out) {
    if (p + 8 > end) return false;
    out = 0;
    for (int i = 0; i < 8; ++i) out |= static_cast<uint64_t>(p[i]) << (8 * i);
    p += 8;
    return true;
}

inline bool getStr(const uint8_t*& p, const uint8_t* end, std::string& out) {
    uint32_t len = 0;
    if (!getU32(p, end, len)) return false;
    if (p + len > end) return false;
    out.assign(reinterpret_cast<const char*>(p), len);
    p += len;
    return true;
}

inline bool getBytes(const uint8_t*& p, const uint8_t* end, std::vector<uint8_t>& out) {
    uint32_t len = 0;
    if (!getU32(p, end, len)) return false;
    if (p + len > end) return false;
    out.assign(p, p + len);
    p += len;
    return true;
}

}  // namespace detail

// ============================================================================
// Control-plane payload serialization
// ============================================================================

std::vector<uint8_t> serializeNodeId(const NodeId& id);
bool deserializeNodeId(const uint8_t*& p, const uint8_t* end, NodeId& out);

std::vector<uint8_t> serializeNodeInfo(const NodeInfo& info);
bool deserializeNodeInfo(const uint8_t*& p, const uint8_t* end, NodeInfo& out);

std::vector<uint8_t> serializeNodeList(const std::vector<NodeInfo>& list);
bool deserializeNodeList(const std::vector<uint8_t>& data, std::vector<NodeInfo>& out);

std::vector<uint8_t> serializeAllocationDesc(const RemoteAllocationDesc& desc);
bool deserializeAllocationDesc(const uint8_t*& p, const uint8_t* end, RemoteAllocationDesc& out);

// ============================================================================
// TcpTransport - blocking, thread-safe framed TCP server + client
// ============================================================================

class TcpTransport {
public:
    using ConnId = uint64_t;
    // Server-side handler: transform a request message into a response message.
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

    bool start(const std::string& listenHost, uint16_t port, RequestHandler handler);
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
    //  - req.stream is sent after the body (buffered mode), or the bytes in
    //    streamIO->writeBuffer are sent slice by slice (slice mode).
    //  - the response body is returned; the response stream is either read
    //    into streamIO->readBuffer (slice mode) or returned in msg.stream.
    std::optional<TcpMessage> request(ConnId id, const TcpMessage& req, const StreamIO* streamIO = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void acceptLoop();
    void serveConnection(uint64_t connId, uintptr_t socket);
};

}  // namespace network
}  // namespace vvm
