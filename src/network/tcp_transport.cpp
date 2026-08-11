#include "vulkan_vm/network/tcp_transport.hpp"
#include "vulkan_vm/network/network_types.hpp"
#include "vulkan_vm/network/wire_format.hpp"
#include "vulkan_vm/utils.hpp"

#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>

#if defined(VVM_NETWORK_HAS_TLS)
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/bio.h>
#endif

#ifdef VVM_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using SocketType = SOCKET;
static constexpr SocketType kInvalidSocket = INVALID_SOCKET;
static constexpr int kSocketError = SOCKET_ERROR;
inline void closeSocket(SocketType s) { closesocket(s); }
inline void socketShutdown(SocketType s) { shutdown(s, SD_BOTH); }
inline int socketRecv(SocketType s, char* buf, int len) { return recv(s, buf, len, 0); }
inline int socketSend(SocketType s, const char* buf, int len) { return send(s, buf, len, 0); }
using SockLenType = int;
#elif VVM_PLATFORM_MACOS
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
using SocketType = int;
static constexpr SocketType kInvalidSocket = -1;
static constexpr int kSocketError = -1;
inline void closeSocket(SocketType s) { close(s); }
inline void socketShutdown(SocketType s) { shutdown(s, SHUT_RDWR); }
inline int socketRecv(SocketType s, char* buf, int len) { return static_cast<int>(recv(s, buf, len, 0)); }
inline int socketSend(SocketType s, const char* buf, int len) { return static_cast<int>(send(s, buf, len, 0)); }
using SockLenType = socklen_t;
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using SocketType = int;
static constexpr SocketType kInvalidSocket = -1;
static constexpr int kSocketError = -1;
inline void closeSocket(SocketType s) { close(s); }
inline void socketShutdown(SocketType s) { shutdown(s, SHUT_RDWR); }
inline int socketRecv(SocketType s, char* buf, int len) { return static_cast<int>(recv(s, buf, len, 0)); }
inline int socketSend(SocketType s, const char* buf, int len) { return static_cast<int>(send(s, buf, len, 0)); }
using SockLenType = socklen_t;
#endif

inline int socketErr() {
#ifdef VVM_PLATFORM_WINDOWS
    return WSAGetLastError();
#else
    return errno;
#endif
}

inline bool isWouldBlockInProgress(int err) {
#ifdef VVM_PLATFORM_WINDOWS
    return err == WSAEWOULDBLOCK || err == WSAEINPROGRESS;
#else
    return err == EINPROGRESS || err == EALREADY;
#endif
}

// ============================================================================
// Helpers: binary put/get (canonical implementation in wire_format.hpp)
// ============================================================================

namespace detail = ::vvm::network::wire;

namespace vvm {
namespace network {

// Internal implementation details
constexpr char kMagic[4] = {'V', 'V', 'M', 'N'};
constexpr uint8_t kProtocolVersion = 1;
constexpr size_t kHeaderSize = 32;
constexpr uint64_t kStreamSliceSize = 4ull * 1024 * 1024;  // 4MB slices (Spark-style)

// Hard caps per message. These bound the wire protocol so a malformed or
// hostile peer cannot cause unbounded allocations via header length fields.
constexpr uint64_t kMaxBodySize  = 1ull * 1024 * 1024 * 1024;   // 1 GiB
constexpr uint64_t kMaxStreamSize = 16ull * 1024 * 1024 * 1024;  // 16 GiB

// ============================================================================
// Stripe Scaling Heuristic (socket-count selection) & Pooled Sockets tunables
// ============================================================================

// Core network constants for hardware sharding
static constexpr size_t kPoolMinStripes = 1;
static constexpr size_t kPoolMaxStripes = 16;
static constexpr uint64_t kPoolMinBytes = 1ull * 1024 * 1024;        // <1MB: no thread overhead
static constexpr uint64_t kPoolChunkTarget = 64ull * 1024 * 1024;    // 64MB saturation sweet spot

// Payloads under 1MB stay single-socket; larger payloads earn one stripe per
// 64MB of data, capped by host hardware concurrency and kPoolMaxStripes.
size_t calculateOptimalStripeCount(uint64_t totalBytes) {
    if (totalBytes < kPoolMinBytes) {
        return kPoolMinStripes;
    }
    size_t stripes = static_cast<size_t>(totalBytes / kPoolChunkTarget);
    if (stripes < kPoolMinStripes) {
        stripes = 2; // Baseline multi-channel for mid-sized chunks
    }
    size_t hwCores = std::thread::hardware_concurrency();
    size_t hostCap = (hwCores > 0) ? hwCores : kPoolMaxStripes;
    if (stripes > hostCap) stripes = hostCap;
    if (stripes > kPoolMaxStripes) stripes = kPoolMaxStripes;
    return (stripes < kPoolMinStripes) ? kPoolMinStripes : stripes;
}

// Push one stripe (up to 4MB) into the kernel in a single pipeline pass:
// disable Nagle for bulk data and size the socket buffers to the slice size.
inline void tunePooledSocket(SocketType sock) {
    int one = 1;
    int buffered = static_cast<int>(kStreamSliceSize);
#ifdef VVM_PLATFORM_WINDOWS
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&buffered), sizeof(buffered));
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&buffered), sizeof(buffered));
#else
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &buffered, sizeof(buffered));
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &buffered, sizeof(buffered));
#endif
}

// ============================================================================
// Control Plane Coordinator (Handshake Protocol)
// ============================================================================
// 32-byte protocol header:
//   [4 magic][1 version][3 reserved][4 type][4 flags][4 bodyLen][4 seq][8 streamLen]
struct NetHeader {
    char magic[4];
    uint8_t version;
    uint8_t reserved[3];
    uint32_t type;
    uint32_t flags;
    uint32_t bodyLen;
    uint32_t seq;
    uint64_t streamLen;
};

// Validate message type is known to the protocol
bool isValidMessageType(uint32_t type) {
    switch (type) {
        case MsgRegisterNode:
        case MsgGetClusterView:
        case MsgAllocate:
        case MsgExport:
        case MsgImport:
        case MsgMigratePull:
        case MsgMigratePush:
        case MsgHeartbeat:
        case MsgLeaveCluster:
        case MsgDeallocate:
        case MsgModelList:
        case MsgModelManifest:
        case MsgModelChunk:
        case MsgTensorAnnounce:
            return true;
        default:
            return false;
    }
}

// Validate flags are within the known set
bool isValidFlags(uint32_t flags) {
    switch (flags) {
        case TcpFlagsRequest:
        case TcpFlagsResponse:
        case TcpFlagsError:
            return true;
        default:
            return false;
    }
}

bool isValidHeaderLengths(const NetHeader& h, const NetworkConfig& netConfig) {
    // Validate message type and flags semantics
    if (!isValidMessageType(h.type)) return false;
    if (!isValidFlags(h.flags)) return false;

    // First check config limits (settable per-connection)
    if (static_cast<uint64_t>(h.bodyLen) > netConfig.maxBodySize) return false;
    if (h.streamLen > netConfig.maxStreamSize) return false;

    // Then enforce absolute hard caps that CANNOT be bypassed by config
    // These prevent any single message from allocating unbounded memory
    if (static_cast<uint64_t>(h.bodyLen) > kMaxBodySize) return false;
    if (h.streamLen > kMaxStreamSize) return false;

    return true;
}

std::vector<uint8_t> encodeHeader(const NetHeader& h) {
    std::vector<uint8_t> out;
    out.insert(out.end(), kMagic, kMagic + 4);
    out.push_back(kProtocolVersion);
    out.push_back(0);
    out.push_back(0);
    out.push_back(0);
    detail::putU32(out, h.type);
    detail::putU32(out, h.flags);
    detail::putU32(out, h.bodyLen);
    detail::putU32(out, h.seq);
    detail::putU64(out, h.streamLen);
    return out;
}

bool decodeHeader(const uint8_t* data, size_t len, NetHeader& out) {
    if (len < kHeaderSize) return false;
    if (std::memcmp(data, kMagic, 4) != 0) return false;
    if (data[4] != kProtocolVersion) return false;
    const uint8_t* p = data + 8;
    const uint8_t* end = data + len;
    return detail::getU32(p, end, out.type) &&
           detail::getU32(p, end, out.flags) &&
           detail::getU32(p, end, out.bodyLen) &&
           detail::getU32(p, end, out.seq) &&
           detail::getU64(p, end, out.streamLen);
}

// Forward declarations (defined below).
bool writeAll(SocketType s, const void* buf, size_t len);
bool readAll(SocketType s, void* buf, size_t len);

// Stream len bytes from src into the socket, slice by slice (no extra copy).
bool writeStreamSlices(SocketType s, const void* src, uint64_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(src);
    uint64_t remaining = len;
    while (remaining > 0) {
        uint64_t slice = remaining < kStreamSliceSize ? remaining : kStreamSliceSize;
        if (!writeAll(s, p, static_cast<size_t>(slice))) return false;
        p += slice;
        remaining -= slice;
    }
    return true;
}

// Stream len bytes from the socket into dst, slice by slice.
bool readStreamSlices(SocketType s, void* dst, uint64_t len) {
    uint8_t* p = static_cast<uint8_t*>(dst);
    uint64_t remaining = len;
    while (remaining > 0) {
        uint64_t slice = remaining < kStreamSliceSize ? remaining : kStreamSliceSize;
        if (!readAll(s, p, static_cast<size_t>(slice))) return false;
        p += slice;
        remaining -= slice;
    }
    return true;
}

// Windowed receive (double-buffered pipeline): the caller supplies a stable
// destination per slice via acquire() (it may block there on its own copy
// fences), then consumed() is reported so GPU work on the slice can start
// while the next slices are still on the wire.
bool readStreamWindows(SocketType s,
                       const std::function<bool(uint64_t, void**, uint64_t, uint64_t&)>& acquire,
                       const std::function<bool(uint64_t, uint64_t)>& consumed,
                       uint64_t len) {
    uint64_t remaining = len;
    uint64_t idx = 0;
    while (remaining > 0) {
        uint64_t want = remaining < kStreamSliceSize ? remaining : kStreamSliceSize;
        void* dst = nullptr;
        uint64_t got = 0;
        if (!acquire(idx, &dst, want, got) || dst == nullptr || got == 0) return false;
        if (got > want) got = want;
        if (!readAll(s, dst, static_cast<size_t>(got))) return false;
        if (consumed && !consumed(idx, got)) return false;
        remaining -= got;
        ++idx;
    }
    return true;
}

// Windowed send (double-buffered pipeline): pulls each slice from the
// caller via provide(), which may block on GPU copies, so copies of later
// slices pipeline behind the wire.
bool writeStreamWindows(SocketType s,
                        const std::function<bool(uint64_t, const void**, uint64_t, uint64_t&)>& provide,
                        uint64_t len) {
    uint64_t remaining = len;
    uint64_t idx = 0;
    while (remaining > 0) {
        uint64_t want = remaining < kStreamSliceSize ? remaining : kStreamSliceSize;
        const void* src = nullptr;
        uint64_t got = 0;
        if (!provide(idx, &src, want, got) || src == nullptr || got == 0) return false;
        if (got > want) got = want;
        if (!writeAll(s, src, static_cast<size_t>(got))) return false;
        remaining -= got;
        ++idx;
    }
    return true;
}

// Ensures sockets subsystem is initialized once per process.
void ensureSocketsInit() {
#ifdef VVM_PLATFORM_WINDOWS
    static std::once_flag flag;
    std::call_once(flag, []() {
        WSADATA data;
        WSAStartup(MAKEWORD(2, 2), &data);
    });
#endif
}

void setTimeouts(SocketType s, int32_t timeoutMs) {
#ifdef VVM_PLATFORM_WINDOWS
    DWORD t = static_cast<DWORD>(timeoutMs);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&t), sizeof(t));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&t), sizeof(t));
#else
    timeval tv{};
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

bool readAll(SocketType s, void* buf, size_t len) {
    char* p = static_cast<char*>(buf);
    while (len > 0) {
        int chunk = static_cast<int>(len > static_cast<size_t>(INT_MAX) ? static_cast<size_t>(INT_MAX) : len);
        int n = socketRecv(s, p, chunk);
        if (n <= 0) return false;
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

bool writeAll(SocketType s, const void* buf, size_t len) {
    const char* p = static_cast<const char*>(buf);
    while (len > 0) {
        int chunk = static_cast<int>(len > static_cast<size_t>(INT_MAX) ? static_cast<size_t>(INT_MAX) : len);
        int n = socketSend(s, p, chunk);
        if (n <= 0) return false;
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

// ============================================================================
// TLS Support
// ============================================================================

// TLS context — holds only the shared SSL_CTX*. SSL* is per-connection.
struct TlsContext {
#if defined(VVM_NETWORK_HAS_TLS)
    SSL_CTX* ctx = nullptr;
#endif
    bool enabled = false;
    bool serverMode = false;
    std::string lastError;

    TlsContext() = default;
    ~TlsContext() {
        cleanup();
    }

    void cleanup() {
#if defined(VVM_NETWORK_HAS_TLS)
        if (ctx) {
            SSL_CTX_free(ctx);
            ctx = nullptr;
        }
#endif
        enabled = false;
    }

    bool initServer(const TlsConfig& config) {
        cleanup();
#if defined(VVM_NETWORK_HAS_TLS)
        SSL_library_init();
        SSL_load_error_strings();
        OpenSSL_add_all_algorithms();

        ctx = SSL_CTX_new(TLS_server_method());
        if (!ctx) {
            lastError = "Failed to create SSL_CTX";
            return false;
        }

        // Set minimum TLS version
        SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);

        // Load certificate and private key
        if (SSL_CTX_use_certificate_file(ctx, config.certPath.c_str(), SSL_FILETYPE_PEM) <= 0) {
            lastError = "Failed to load certificate: " + getOpenSslError();
            return false;
        }
        if (SSL_CTX_use_PrivateKey_file(ctx, config.keyPath.c_str(), SSL_FILETYPE_PEM) <= 0) {
            lastError = "Failed to load private key: " + getOpenSslError();
            return false;
        }
        if (!SSL_CTX_check_private_key(ctx)) {
            lastError = "Private key does not match certificate";
            return false;
        }

        // Require CA for peer verification — fail closed if missing
        if (config.verifyPeer && config.caPath.empty()) {
            lastError = "TLS peer verification requires a CA path";
            return false;
        }

        // Load CA for client verification
        if (!config.caPath.empty()) {
            if (SSL_CTX_load_verify_locations(ctx, config.caPath.c_str(), nullptr) <= 0) {
                lastError = "Failed to load CA file: " + getOpenSslError();
                return false;
            }
            SSL_CTX_set_verify(ctx, config.verifyPeer ? SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT : SSL_VERIFY_NONE, nullptr);
        }

        // ALPN
        if (!config.alpnProtocols.empty()) {
            std::vector<unsigned char> alpnWire = encodeAlpnProtocols(config.alpnProtocols);
            SSL_CTX_set_alpn_select_cb(ctx, [](SSL* ssl, const unsigned char** out, unsigned char* outlen, const unsigned char* in, unsigned int inlen, void* arg) -> int {
                const std::vector<unsigned char>* protos = static_cast<const std::vector<unsigned char>*>(arg);
                if (SSL_select_next_proto(const_cast<unsigned char**>(out), outlen, protos->data(), static_cast<unsigned int>(protos->size()), in, inlen) == OPENSSL_NPN_NEGOTIATED) {
                    return SSL_TLSEXT_ERR_OK;
                }
                return SSL_TLSEXT_ERR_NOACK;
            }, new std::vector<unsigned char>(std::move(alpnWire)));
        }

        enabled = true;
        serverMode = true;
        return true;
#else
        lastError = "TLS not compiled in (VVM_NETWORK_HAS_TLS=0)";
        return false;
#endif
    }

    bool initClient(const TlsConfig& config) {
        cleanup();
#if defined(VVM_NETWORK_HAS_TLS)
        SSL_library_init();
        SSL_load_error_strings();
        OpenSSL_add_all_algorithms();

        ctx = SSL_CTX_new(TLS_client_method());
        if (!ctx) {
            lastError = "Failed to create SSL_CTX";
            return false;
        }

        SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);

        // Require CA for peer verification — fail closed if missing
        if (config.verifyPeer && config.caPath.empty()) {
            lastError = "TLS peer verification requires a CA path";
            return false;
        }

        if (!config.caPath.empty()) {
            if (SSL_CTX_load_verify_locations(ctx, config.caPath.c_str(), nullptr) <= 0) {
                lastError = "Failed to load CA file: " + getOpenSslError();
                return false;
            }
            SSL_CTX_set_verify(ctx, config.verifyPeer ? SSL_VERIFY_PEER : SSL_VERIFY_NONE, nullptr);
        }

        // ALPN
        if (!config.alpnProtocols.empty()) {
            std::vector<unsigned char> alpnWire = encodeAlpnProtocols(config.alpnProtocols);
            SSL_CTX_set_alpn_protos(ctx, alpnWire.data(), static_cast<unsigned int>(alpnWire.size()));
        }

        enabled = true;
        serverMode = false;
        return true;
#else
        lastError = "TLS not compiled in (VVM_NETWORK_HAS_TLS=0)";
        return false;
#endif
    }

    // Get the shared SSL_CTX* for creating per-connection SSL objects.
#if defined(VVM_NETWORK_HAS_TLS)
    SSL_CTX* context() const {
        return ctx;
    }
#else
    void* context() const {
        return nullptr;
    }
#endif

private:
#if defined(VVM_NETWORK_HAS_TLS)
    static std::vector<unsigned char> encodeAlpnProtocols(const std::string& protocols) {
        std::vector<unsigned char> result;
        size_t start = 0;
        while (start < protocols.size()) {
            size_t end = protocols.find(',', start);
            if (end == std::string::npos) end = protocols.size();
            std::string proto = protocols.substr(start, end - start);
            if (!proto.empty() && proto.size() <= 255) {
                result.push_back(static_cast<unsigned char>(proto.size()));
                result.insert(result.end(), proto.begin(), proto.end());
            }
            start = end + 1;
        }
        return result;
    }

    static std::string getOpenSslError() {
        std::string err;
        unsigned long e;
        while ((e = ERR_get_error()) != 0) {
            char buf[256];
            ERR_error_string_n(e, buf, sizeof(buf));
            err += buf;
            err += "; ";
        }
        return err;
    }
#endif
};

// Per-connection TLS wrapper. Each connection gets its own SSL* object.
// When VVM_NETWORK_HAS_TLS is not defined, this is a no-op wrapper.
class TlsConnection {
public:
#if defined(VVM_NETWORK_HAS_TLS)
    TlsConnection(SSL_CTX* ctx, SocketType sock) : socket_(sock) {
        ssl_ = SSL_new(ctx);
        if (ssl_) {
            SSL_set_fd(ssl_, static_cast<int>(sock));
        }
    }
#else
    TlsConnection(void* /*ctx*/, SocketType sock) : socket_(sock) {}
#endif

    ~TlsConnection() {
        reset();
    }

    // Non-copyable, movable
    TlsConnection(const TlsConnection&) = delete;
    TlsConnection& operator=(const TlsConnection&) = delete;
#if defined(VVM_NETWORK_HAS_TLS)
    TlsConnection(TlsConnection&& other) noexcept
        : socket_(other.socket_), ssl_(other.ssl_) {
        other.ssl_ = nullptr;
    }
    TlsConnection& operator=(TlsConnection&& other) noexcept {
        if (this != &other) {
            reset();
            socket_ = other.socket_;
            ssl_ = other.ssl_;
            other.ssl_ = nullptr;
        }
        return *this;
    }
#else
    TlsConnection(TlsConnection&& other) noexcept : socket_(other.socket_) {}
    TlsConnection& operator=(TlsConnection&& other) noexcept {
        if (this != &other) {
            socket_ = other.socket_;
        }
        return *this;
    }
#endif

    bool isValid() const {
#if defined(VVM_NETWORK_HAS_TLS)
        return ssl_ != nullptr;
#else
        return false;
#endif
    }

    bool accept() {
#if defined(VVM_NETWORK_HAS_TLS)
        if (!ssl_) return false;
        return SSL_accept(ssl_) > 0;
#else
        return false;
#endif
    }

    bool connect(const std::string& host) {
#if defined(VVM_NETWORK_HAS_TLS)
        if (!ssl_) return false;
        if (!SSL_set1_host(ssl_, host.c_str())) {
            return false;
        }
        SSL_set_tlsext_host_name(ssl_, host.c_str());
        if (SSL_connect(ssl_) <= 0) {
            return false;
        }
        if (SSL_get_verify_result(ssl_) != X509_V_OK) {
            return false;
        }
        return true;
#else
        return false;
#endif
    }

    // Verify ALPN negotiation matched expected protocol. Call after handshake.
    bool verifyAlpn(const std::string& expectedProto) const {
#if defined(VVM_NETWORK_HAS_TLS)
        if (expectedProto.empty()) return true;  // no ALPN requirement
        const unsigned char* selected = nullptr;
        unsigned int selectedLen = 0;
        SSL_get0_alpn_selected(ssl_, &selected, &selectedLen);
        if (selectedLen == 0) return false;
        return selectedLen == expectedProto.size() &&
               std::memcmp(selected, expectedProto.data(), selectedLen) == 0;
#else
        return true;
#endif
    }

    int read(void* buf, int len) {
#if defined(VVM_NETWORK_HAS_TLS)
        if (!ssl_) return -1;
        return SSL_read(ssl_, buf, len);
#else
        return -1;
#endif
    }

    int write(const void* buf, int len) {
#if defined(VVM_NETWORK_HAS_TLS)
        if (!ssl_) return -1;
        return SSL_write(ssl_, buf, len);
#else
        return -1;
#endif
    }

    void reset() {
#if defined(VVM_NETWORK_HAS_TLS)
        if (ssl_) {
            SSL_shutdown(ssl_);
            SSL_free(ssl_);
            ssl_ = nullptr;
        }
#endif
    }

private:
    SocketType socket_ = kInvalidSocket;
#if defined(VVM_NETWORK_HAS_TLS)
    SSL* ssl_ = nullptr;
#endif
};

// ============================================================================
// Control-plane payload serialization
// ============================================================================

std::vector<uint8_t> serializeNodeId(const NodeId& id) {
    std::vector<uint8_t> b;
    detail::putStr(b, id.host);
    detail::putU32(b, id.port);
    detail::putU32(b, id.nodeIndex);
    detail::putStr(b, id.uuid);
    return b;
}

bool deserializeNodeId(const uint8_t*& p, const uint8_t* end, NodeId& out) {
    return detail::getStr(p, end, out.host) &&
           detail::getU32(p, end, out.port) &&
           detail::getU32(p, end, out.nodeIndex) &&
           detail::getStr(p, end, out.uuid);
}

std::vector<uint8_t> serializeNodeInfo(const NodeInfo& info) {
    std::vector<uint8_t> b = serializeNodeId(info.id);
    detail::putU32(b, static_cast<uint32_t>(info.gpuDevices.size()));
    for (const auto& gpu : info.gpuDevices) detail::putStr(b, gpu);
    detail::putStr(b, info.nicName);
    detail::putU8(b, info.rdmaCapable ? 1 : 0);
    detail::putU8(b, info.gpuDirectCapable ? 1 : 0);
    detail::putU64(b, info.timestamp);
    return b;
}

bool deserializeNodeInfo(const uint8_t*& p, const uint8_t* end, NodeInfo& out) {
    if (!deserializeNodeId(p, end, out.id)) return false;
    uint32_t gpuCount = 0;
    if (!detail::getU32(p, end, gpuCount)) return false;
    if (gpuCount > 64) return false;  // cap GPUs per node
    out.gpuDevices.clear();
    for (uint32_t i = 0; i < gpuCount; ++i) {
        std::string gpu;
        if (!detail::getStr(p, end, gpu)) return false;
        out.gpuDevices.push_back(std::move(gpu));
    }
    uint8_t rdma = 0, gpuDirect = 0;
    if (!detail::getStr(p, end, out.nicName)) return false;
    if (!detail::getU8(p, end, rdma)) return false;
    if (!detail::getU8(p, end, gpuDirect)) return false;
    if (!detail::getU64(p, end, out.timestamp)) return false;
    out.rdmaCapable = rdma != 0;
    out.gpuDirectCapable = gpuDirect != 0;
    return true;
}

std::vector<uint8_t> serializeNodeList(const std::vector<NodeInfo>& list) {
    std::vector<uint8_t> b;
    detail::putU32(b, static_cast<uint32_t>(list.size()));
    for (const auto& info : list) {
        auto nodeBytes = serializeNodeInfo(info);
        b.insert(b.end(), nodeBytes.begin(), nodeBytes.end());
    }
    return b;
}

bool deserializeNodeList(const std::vector<uint8_t>& data, std::vector<NodeInfo>& out) {
    const uint8_t* p = data.data();
    const uint8_t* end = p + data.size();
    uint32_t count = 0;
    if (!detail::getU32(p, end, count)) return false;
    // Cap semantic count to prevent pathological memory allocation
    if (count > 4096) return false;
    out.clear();
    for (uint32_t i = 0; i < count; ++i) {
        NodeInfo info;
        if (!deserializeNodeInfo(p, end, info)) return false;
        out.push_back(std::move(info));
    }
    return true;
}

std::vector<uint8_t> serializeAllocationDesc(const RemoteAllocationDesc& desc) {
    std::vector<uint8_t> b = serializeNodeId(desc.owner);
    detail::putU64(b, desc.size);
    detail::putU64(b, desc.localAllocId);
    detail::putU8(b, desc.hasRdmaAddr ? 1 : 0);
    detail::putU64(b, desc.rdmaAddr);
    detail::putU32(b, desc.rkey);
    detail::putU8(b, desc.hasUcxAddr ? 1 : 0);
    detail::putBytes(b, desc.ucxWorkerAddr);
    detail::putBytes(b, desc.ucxPackedRkey);
    detail::putU64(b, desc.ucxRemoteAddr);
    detail::putU32(b, desc.ucxDeviceIndex);
    detail::putU8(b, desc.hasHostShadow ? 1 : 0);
    detail::putU32(b, desc.usageFlags);
    detail::putU32(b, desc.memoryTypeIndex);
    detail::putU8(b, desc.dedicatedAllocation ? 1 : 0);
    detail::putU32(b, static_cast<uint32_t>(desc.handleType));
    detail::putBytes(b, desc.externalHandle);
    detail::putU64(b, desc.timestamp);
    detail::putStr(b, desc.allocationName);
    return b;
}

bool deserializeAllocationDesc(const uint8_t*& p, const uint8_t* end, RemoteAllocationDesc& out) {
    uint8_t rdma = 0, ucx = 0, shadow = 0, dedicated = 0;
    uint32_t handleType = 0;
    if (!deserializeNodeId(p, end, out.owner)) return false;
    if (!detail::getU64(p, end, out.size)) return false;
    if (!detail::getU64(p, end, out.localAllocId)) return false;
    if (!detail::getU8(p, end, rdma)) return false;
    if (!detail::getU64(p, end, out.rdmaAddr)) return false;
    if (!detail::getU32(p, end, out.rkey)) return false;
    if (!detail::getU8(p, end, ucx)) return false;
    if (!detail::getBytes(p, end, out.ucxWorkerAddr)) return false;
    if (!detail::getBytes(p, end, out.ucxPackedRkey)) return false;
    if (!detail::getU64(p, end, out.ucxRemoteAddr)) return false;
    if (!detail::getU32(p, end, out.ucxDeviceIndex)) return false;
    if (!detail::getU8(p, end, shadow)) return false;
    if (!detail::getU32(p, end, out.usageFlags)) return false;
    if (!detail::getU32(p, end, out.memoryTypeIndex)) return false;
    if (!detail::getU8(p, end, dedicated)) return false;
    if (!detail::getU32(p, end, handleType)) return false;
    if (!detail::getBytes(p, end, out.externalHandle)) return false;
    if (!detail::getU64(p, end, out.timestamp)) return false;
    if (!detail::getStr(p, end, out.allocationName)) return false;
    out.hasRdmaAddr = rdma != 0;
    out.hasUcxAddr = ucx != 0;
    out.hasHostShadow = shadow != 0;
    out.dedicatedAllocation = dedicated != 0;
    out.handleType = static_cast<ExternalHandleType>(handleType);
    return true;
}

// ============================================================================
// TcpTransport Implementation
// ============================================================================

struct TcpTransport::Impl {
    // Connection info with last activity tracking
    struct ConnectionInfo {
        SocketType socket = kInvalidSocket;
        std::chrono::steady_clock::time_point lastActivity;
        bool isServerSide = false;
        std::unique_ptr<TlsConnection> tlsConn;  // per-connection TLS state
    };
    
    // ----- server -----
    SocketType listener = kInvalidSocket;
    std::atomic<bool> running{false};
    uint16_t boundPort = 0;
    RequestHandler handler;
    std::thread acceptThread;
    std::vector<std::thread> serveThreads;
    std::mutex serveThreadsMutex;

    // ----- client -----
    std::atomic<uint64_t> nextConnId{1};
    uint32_t nextSeq_ = 0;
    std::mutex connsMutex;
    std::unordered_map<uint64_t, ConnectionInfo> conns;
    std::mutex ioMutex;  // serializes request/response exchanges

    // TLS
    std::unique_ptr<TlsContext> tlsContext;
    TlsConfig tlsConfig;
    
    // Connection idle timeout
    std::chrono::milliseconds connectionIdleTimeout{300000};  // 5 min default
    std::thread cleanupThread;
    std::atomic<bool> stopCleanup_{false};
    std::mutex cleanupMutex;
    std::condition_variable cleanupCV;

    // Network config (for caps, rate limits, etc.)
    NetworkConfig netConfig;

    // ============================================================================
    // Connection Pool for Parallel Stream Transfers (TCP Stream Striping)
    // ============================================================================
    struct ConnectionPool {
        struct PooledConnection {
            SocketType socket = kInvalidSocket;
            std::atomic<bool> inUse{false};
            std::string remoteHost;
            uint16_t remotePort = 0;
            
            PooledConnection() = default;
            PooledConnection(const PooledConnection&) = delete;
            PooledConnection& operator=(const PooledConnection&) = delete;
            PooledConnection(PooledConnection&& other) noexcept
                : socket(other.socket), inUse(other.inUse.load()),
                  remoteHost(std::move(other.remoteHost)), remotePort(other.remotePort) {
                other.socket = kInvalidSocket;
                other.inUse.store(false);
                other.remotePort = 0;
            }
            PooledConnection& operator=(PooledConnection&& other) noexcept {
                if (this != &other) {
                    socket = other.socket;
                    inUse.store(other.inUse.load());
                    remoteHost = std::move(other.remoteHost);
                    remotePort = other.remotePort;
                    other.socket = kInvalidSocket;
                    other.inUse.store(false);
                    other.remotePort = 0;
                }
                return *this;
            }
        };
        
        std::string remoteHost;
        uint16_t remotePort = 0;
        size_t poolSize = 4;
        std::vector<PooledConnection> connections;
        std::mutex mutex;
        
        ConnectionPool() = default;
        ConnectionPool(const std::string& host, uint16_t port, size_t size)
            : remoteHost(host), remotePort(port), poolSize(size) {}
        
        bool initialize() {
            connections.reserve(poolSize);
            for (size_t i = 0; i < poolSize; ++i) {
                SocketType sock = socket(AF_INET, SOCK_STREAM, 0);
                if (sock == kInvalidSocket) {
                    VVM_LOG_ERROR("ConnectionPool: socket() failed for connection {}/{}", i + 1, poolSize);
                    return false;
                }
                tunePooledSocket(sock);
                
                sockaddr_in addr{};
                addr.sin_family = AF_INET;
                addr.sin_port = htons(remotePort);
                if (inet_pton(AF_INET, remoteHost.c_str(), &addr.sin_addr) != 1) {
                    VVM_LOG_ERROR("ConnectionPool: invalid remote host {}", remoteHost);
                    closeSocket(sock);
                    return false;
                }
                
                if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == kSocketError) {
                    int err = socketErr();
                    if (!isWouldBlockInProgress(err)) {
                        VVM_LOG_ERROR("ConnectionPool: connect() failed for {}:{}: {}", remoteHost, remotePort, err);
                        closeSocket(sock);
                        return false;
                    }
}

                // Blocking socket: connect() completes synchronously; exact
                // transfer I/O then blocks on kernel buffers (no poll loops).
                PooledConnection pc;
                pc.socket = sock;
                pc.remoteHost = remoteHost;
                pc.remotePort = remotePort;
                connections.push_back(std::move(pc));
            }
            return connections.size() == poolSize;
        }
        
        // Acquire a free connection (round-robin with skip if busy)
        SocketType acquire() {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto& pc : connections) {
                if (!pc.inUse.exchange(true)) {
                    return pc.socket;
                }
            }
            return kInvalidSocket; // all busy
        }
        
        void release(SocketType s) {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto& pc : connections) {
                if (pc.socket == s) {
                    pc.inUse.store(false);
                    break;
                }
            }
        }
        
        void shutdown() {
            for (auto& pc : connections) {
                if (pc.socket != kInvalidSocket) {
                    socketShutdown(pc.socket);
                    closeSocket(pc.socket);
                    pc.socket = kInvalidSocket;
                }
            }
        }
        
        size_t activeCount() const {
            size_t count = 0;
            for (const auto& pc : connections) {
                if (pc.socket != kInvalidSocket) count++;
            }
            return count;
        }
    };
    
    // Pool registry for multiple remote endpoints
    std::unordered_map<std::string, std::unique_ptr<ConnectionPool>> connectionPools;
    std::mutex poolsMutex;
    
    // Create or get a connection pool for a remote endpoint
    ConnectionPool* getOrCreatePool(const std::string& host, uint16_t port, size_t poolSize) {
        std::string key = host + ":" + std::to_string(port);
        std::lock_guard<std::mutex> lock(poolsMutex);
        auto it = connectionPools.find(key);
        if (it != connectionPools.end()) return it->second.get();
        auto pool = std::make_unique<ConnectionPool>(host, port, poolSize);
        if (!pool->initialize()) return nullptr;
        ConnectionPool* ptr = pool.get();
        connectionPools.emplace(std::move(key), std::move(pool));
        return ptr;
    }
    
    // Shared striped-transfer engine: split [buf, len) into contiguous
    // per-socket segments and transfer each segment on its own pooled
    // connection in parallel (1:1 socket-to-segment pairing, like tt-metal
    // connection lists). Blocking sockets avoid poll loops; one stalled peer
    // only stalls its own stripe. All-or-nothing: false if any stripe fails.
    template <bool IsWrite>
    bool transferStripedImpl(ConnectionPool* pool, void* buf, uint64_t len, size_t stripes) {
        if (len == 0) return true;

        // Single stripe: transfer on one socket without spawning a thread.
        if (stripes < 2) {
            SocketType s = pool->acquire();
            if (s == kInvalidSocket) return false;
            bool ok = IsWrite ? writeAllTls(s, buf, static_cast<size_t>(len))
                              : readAllTls(s, buf, static_cast<size_t>(len));
            pool->release(s);
            return ok;
        }

        // Reserve one socket per stripe (pool must hold >= stripes sockets).
        std::vector<SocketType> socks;
        socks.reserve(stripes);
        for (size_t i = 0; i < stripes; ++i) {
            SocketType s = pool->acquire();
            if (s == kInvalidSocket) {
                for (SocketType held : socks) pool->release(held);
                return false;
            }
            socks.push_back(s);
        }

        uint64_t base = len / stripes;
        uint64_t rem = len % stripes;
        std::vector<std::future<bool>> workers;
        workers.reserve(stripes);
        for (size_t i = 0; i < stripes; ++i) {
            uint64_t off = i * base;
            uint64_t segLen = base + (i == stripes - 1 ? rem : 0);
            uint8_t* seg = static_cast<uint8_t*>(buf) + off;
            SocketType s = socks[i];
            workers.push_back(std::async(std::launch::async, [this, s, seg, segLen, pool]() -> bool {
                bool ok;
                if constexpr (IsWrite) {
                    ok = writeAllTls(s, seg, static_cast<size_t>(segLen));
                } else {
                    ok = readAllTls(s, seg, static_cast<size_t>(segLen));
                }
                pool->release(s);
                return ok;
            }));
        }

        bool allOk = true;
        for (auto& w : workers) {
            if (!w.get()) allOk = false;
        }
        return allOk;
    }

    // Striped write: distribute data across pool connections
    bool writeStreamSlicesStriped(const std::string& host, uint16_t port, const void* src, uint64_t len, size_t poolSize = 4) {
        ConnectionPool* pool = getOrCreatePool(host, port, poolSize);
        if (!pool || pool->activeCount() == 0) return false;

        // Choose stripes from the caller-specified pool size, bounded by slice
        // count and hardware concurrency so we never oversubscribe threads.
        size_t stripes = poolSize;
        size_t hwCores = std::thread::hardware_concurrency();
        if (hwCores > 0 && stripes > hwCores) stripes = hwCores;
        uint64_t sliceCount = (len + kStreamSliceSize - 1) / kStreamSliceSize;
        if (sliceCount == 0) sliceCount = 1;
        if (stripes > sliceCount) stripes = static_cast<size_t>(sliceCount);

        return transferStripedImpl<true>(pool, const_cast<uint8_t*>(static_cast<const uint8_t*>(src)), len, stripes);
    }

    // Striped read: distribute reads across pool connections
    bool readStreamSlicesStriped(const std::string& host, uint16_t port, void* dst, uint64_t len, size_t poolSize = 4) {
        ConnectionPool* pool = getOrCreatePool(host, port, poolSize);
        if (!pool || pool->activeCount() == 0) return false;

        size_t stripes = poolSize;
        size_t hwCores = std::thread::hardware_concurrency();
        if (hwCores > 0 && stripes > hwCores) stripes = hwCores;
        uint64_t sliceCount = (len + kStreamSliceSize - 1) / kStreamSliceSize;
        if (sliceCount == 0) sliceCount = 1;
        if (stripes > sliceCount) stripes = static_cast<size_t>(sliceCount);

        return transferStripedImpl<false>(pool, dst, len, stripes);
    }

    ~Impl() { stop(); }

    // TLS-aware read/write — uses per-connection TlsConnection when provided,
    // otherwise falls back to raw socket I/O.
    bool writeAllTls(SocketType s, const void* buf, size_t len, TlsConnection* tlsConn = nullptr) {
        if (tlsConn && tlsConn->isValid()) {
            const char* p = static_cast<const char*>(buf);
            size_t remaining = len;
            while (remaining > 0) {
                int chunk = static_cast<int>(remaining > static_cast<size_t>(INT_MAX) ? static_cast<size_t>(INT_MAX) : remaining);
                int n = tlsConn->write(p, chunk);
                if (n <= 0) return false;
                p += n;
                remaining -= static_cast<size_t>(n);
            }
            return true;
        } else {
            return writeAll(s, buf, len);
        }
    }

    bool readAllTls(SocketType s, void* buf, size_t len, TlsConnection* tlsConn = nullptr) {
        if (tlsConn && tlsConn->isValid()) {
            char* p = static_cast<char*>(buf);
            size_t remaining = len;
            while (remaining > 0) {
                int chunk = static_cast<int>(remaining > static_cast<size_t>(INT_MAX) ? static_cast<size_t>(INT_MAX) : remaining);
                int n = tlsConn->read(p, chunk);
                if (n <= 0) return false;
                p += n;
                remaining -= static_cast<size_t>(n);
            }
            return true;
        } else {
            return readAll(s, buf, len);
        }
    }

    void stop() {
        running = false;
        stopCleanup_ = true;
        cleanupCV.notify_all();
        if (listener != kInvalidSocket) {
            socketShutdown(listener);
            closeSocket(listener);
            listener = kInvalidSocket;
        }
        {
            std::lock_guard<std::mutex> lock(connsMutex);
            for (auto& [id, conn] : conns) {
                if (conn.socket != kInvalidSocket) {
                    socketShutdown(conn.socket);
                    closeSocket(conn.socket);
                }
            }
        }
        if (acceptThread.joinable()) acceptThread.join();
        if (cleanupThread.joinable()) cleanupThread.join();
        {
            std::lock_guard<std::mutex> lock(serveThreadsMutex);
            for (auto& t : serveThreads) {
                if (t.joinable()) t.join();
            }
            serveThreads.clear();
        }
        {
            std::lock_guard<std::mutex> lock(connsMutex);
            conns.clear();
        }
        if (tlsContext) {
            tlsContext->cleanup();
        }
        // Shutdown connection pools
        for (auto& [key, pool] : connectionPools) {
            pool->shutdown();
        }
        connectionPools.clear();
    }
};

TcpTransport::TcpTransport() : impl_(std::make_unique<Impl>()) {}

TcpTransport::~TcpTransport() {
    if (impl_) impl_->stop();
}

bool TcpTransport::start(const std::string& listenHost, uint16_t port, RequestHandler handler,
                           const NetworkConfig& netConfig,
                           std::chrono::milliseconds idleTimeout) {
    ensureSocketsInit();
    if (impl_->running) return false;
    
    impl_->netConfig = netConfig;
    impl_->connectionIdleTimeout = idleTimeout;

    SocketType sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == kInvalidSocket) {
        VVM_LOG_ERROR("socket() failed to create listener");
        return false;
    }

    int opt = 1;
#ifdef VVM_PLATFORM_WINDOWS
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (listenHost.empty() || listenHost == "0.0.0.0" || listenHost == "*") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, listenHost.c_str(), &addr.sin_addr) != 1) {
            VVM_LOG_ERROR("Invalid listen host: {}", listenHost);
            closeSocket(sock);
            return false;
        }
    }

    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == kSocketError) {
        VVM_LOG_ERROR("bind() failed for {}:{}", listenHost, port);
#ifdef VVM_PLATFORM_WINDOWS
        VVM_LOG_ERROR("  WSA error: {}", WSAGetLastError());
#endif
        closeSocket(sock);
        return false;
    }

    if (listen(sock, 32) == kSocketError) {
        VVM_LOG_ERROR("listen() failed on port {}", port);
        closeSocket(sock);
        return false;
    }

    sockaddr_in bound{};
    SockLenType len = sizeof(bound);
#ifdef VVM_PLATFORM_WINDOWS
    len = sizeof(bound);
#endif
    if (getsockname(sock, reinterpret_cast<sockaddr*>(&bound), &len) == 0) {
        impl_->boundPort = ntohs(bound.sin_port);
    } else {
        impl_->boundPort = port;
    }

    impl_->handler = std::move(handler);
    impl_->listener = sock;
    impl_->running = true;

    // Initialize TLS if configured
    if (impl_->tlsConfig.enabled) {
        impl_->tlsContext = std::make_unique<TlsContext>();
        if (impl_->tlsConfig.enabled) {
            if (!impl_->tlsContext->initServer(impl_->tlsConfig)) {
                VVM_LOG_ERROR("Failed to initialize TLS server: {}", impl_->tlsContext->lastError);
                closeSocket(sock);
                return false;
            }
        }
    }

    // Start connection cleanup thread
    impl_->stopCleanup_ = false;
    impl_->cleanupThread = std::thread([this]() { this->cleanupLoop(); });

    impl_->acceptThread = std::thread([this]() { acceptLoop(); });
    VVM_LOG_INFO("TcpTransport listening on {}:{}", listenHost, impl_->boundPort);
    return true;
}

void TcpTransport::stop() {
    impl_->stop();
}

bool TcpTransport::isRunning() const {
    return impl_->running;
}

uint16_t TcpTransport::getBoundPort() const {
    return impl_->boundPort;
}

bool TcpTransport::enableTls(const TlsConfig& tlsConfig) {
    if (impl_->running) {
        VVM_LOG_ERROR("Cannot enable TLS while transport is running");
        return false;
    }
    impl_->tlsConfig = tlsConfig;
    impl_->tlsConfig.enabled = true;
    return true;
}

bool TcpTransport::isTlsEnabled() const {
    return impl_->tlsContext && impl_->tlsContext->enabled;
}

bool TcpTransport::createConnectionPool(const std::string& host, uint16_t port, size_t poolSize) {
    return impl_->getOrCreatePool(host, port, poolSize) != nullptr;
}

bool TcpTransport::writeStreamStriped(const std::string& host, uint16_t port,
                                      const void* src, uint64_t len, size_t poolSize) {
    return impl_->writeStreamSlicesStriped(host, port, src, len, poolSize);
}

bool TcpTransport::readStreamStriped(const std::string& host, uint16_t port,
                                     void* dst, uint64_t len, size_t poolSize) {
    return impl_->readStreamSlicesStriped(host, port, dst, len, poolSize);
}

size_t TcpTransport::calculateOptimalStripes(uint64_t dataSize) {
    return calculateOptimalStripeCount(dataSize);
}

bool TcpTransport::writeStreamStripedAuto(const std::string& host, uint16_t port,
                                          const void* src, uint64_t len) {
    return impl_->writeStreamSlicesStriped(host, port, src, len, calculateOptimalStripeCount(len));
}

bool TcpTransport::readStreamStripedAuto(const std::string& host, uint16_t port,
                                         void* dst, uint64_t len) {
    return impl_->readStreamSlicesStriped(host, port, dst, len, calculateOptimalStripeCount(len));
}

void TcpTransport::shutdownConnectionPool(const std::string& host, uint16_t port) {
    std::string key = host + ":" + std::to_string(port);
    std::lock_guard<std::mutex> lock(impl_->poolsMutex);
    auto it = impl_->connectionPools.find(key);
    if (it != impl_->connectionPools.end()) {
        it->second->shutdown();
        impl_->connectionPools.erase(it);
    }
}

void TcpTransport::acceptLoop() {
    while (impl_->running) {
        sockaddr_in peer{};
        SockLenType addrLen = sizeof(peer);
        SocketType client = accept(impl_->listener, reinterpret_cast<sockaddr*>(&peer), &addrLen);
        if (client == kInvalidSocket) {
            if (!impl_->running) break;
            // Distinguish transient errors (continue) from fatal errors (break).
#ifdef VVM_PLATFORM_WINDOWS
            int err = WSAGetLastError();
            if (err == WSAEWOULDBLOCK || err == WSAEINTR) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            VVM_LOG_ERROR("acceptLoop: fatal error {}, stopping accept loop", err);
            break;
#else
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            VVM_LOG_ERROR("acceptLoop: fatal error {} ({}), stopping accept loop", errno, strerror(errno));
            break;
#endif
        }
        setTimeouts(client, 30000);
        uint64_t id = impl_->nextConnId++;
        {
            std::lock_guard<std::mutex> lock(impl_->connsMutex);
            impl_->conns[id] = {client, std::chrono::steady_clock::now(), true};
        }
        {
            std::lock_guard<std::mutex> lock(impl_->serveThreadsMutex);
            impl_->serveThreads.emplace_back([this, id, client]() { serveConnection(id, client); });
        }
    }
}

void TcpTransport::serveConnection(uint64_t connId, uintptr_t sRaw) {
    SocketType s = static_cast<SocketType>(sRaw);

    // TLS handshake for server — create per-connection SSL object
    std::unique_ptr<TlsConnection> tlsConn;
    if (impl_->tlsContext && impl_->tlsContext->enabled && impl_->tlsContext->serverMode) {
        tlsConn = std::make_unique<TlsConnection>(impl_->tlsContext->context(), s);
        if (!tlsConn->accept()) {
            VVM_LOG_ERROR("TLS handshake failed for conn {}", connId);
            closeSocket(s);
            {
                std::lock_guard<std::mutex> lock(impl_->connsMutex);
                impl_->conns.erase(connId);
            }
            return;
        }
        // Enforce ALPN protocol negotiation
        if (!tlsConn->verifyAlpn(impl_->tlsConfig.alpnProtocols)) {
            VVM_LOG_ERROR("TLS ALPN negotiation failed for conn {} (expected '{}')",
                          connId, impl_->tlsConfig.alpnProtocols);
            closeSocket(s);
            {
                std::lock_guard<std::mutex> lock(impl_->connsMutex);
                impl_->conns.erase(connId);
            }
            return;
        }
    }

    std::vector<uint8_t> header(kHeaderSize);
    std::vector<uint8_t> drain(static_cast<size_t>(kStreamSliceSize));
    while (impl_->running) {
        // Update last activity timestamp
        {
            std::lock_guard<std::mutex> lock(impl_->connsMutex);
            auto it = impl_->conns.find(connId);
            if (it != impl_->conns.end()) {
                it->second.lastActivity = std::chrono::steady_clock::now();
            }
        }

        if (!impl_->readAllTls(s, header.data(), header.size(), tlsConn.get())) break;

        NetHeader nh{};
        if (!decodeHeader(header.data(), header.size(), nh)) break;
        if (!isValidHeaderLengths(nh, impl_->netConfig)) {
            VVM_LOG_ERROR("serveConnection: peer sent oversized message "
                          "(bodyLen={}, streamLen={}); dropping connection",
                          nh.bodyLen, nh.streamLen);
            break;
        }

        TcpMessage req;
        req.type = nh.type;
        req.flags = nh.flags;
        req.seq = nh.seq;
        if (nh.bodyLen > 0) {
            req.body.resize(nh.bodyLen);
            if (!impl_->readAllTls(s, req.body.data(), nh.bodyLen, tlsConn.get())) break;
        }

        TcpMessage resp;
        resp.type = req.type;
        resp.flags = TcpFlagsResponse;
        resp.seq = req.seq;

        if (nh.streamLen > 0) {
            // Streamed request: prepare phase so the handler allocates a sink.
            req.streamLen = nh.streamLen;
            if (impl_->handler) {
                impl_->handler(req, resp);
            } else {
                resp.flags = TcpFlagsError;
            }

            const bool accepted = impl_->handler != nullptr && resp.flags != TcpFlagsError &&
                                  (resp.streamSink != nullptr || resp.streamWindowAcquire != nullptr);
            if (accepted) {
                std::function<void()> sinkCleanup = resp.streamSinkCleanup;
                if (resp.streamWindowAcquire) {
                    if (!readStreamWindows(s, resp.streamWindowAcquire, resp.streamWindowConsumed, nh.streamLen)) {
                        if (sinkCleanup) sinkCleanup();
                        break;
                    }
                } else if (!readStreamSlices(s, resp.streamSink, nh.streamLen)) {
                    if (sinkCleanup) sinkCleanup();
                    break;
                }
                // Finalize phase: stream bytes are now in resp.streamSink.
                TcpMessage finalReq = req;
                finalReq.streamReceived = true;
                TcpMessage finalResp;
                finalResp.type = finalReq.type;
                finalResp.flags = TcpFlagsResponse;
                finalResp.seq = finalReq.seq;
                impl_->handler(finalReq, finalResp);
                if (sinkCleanup) sinkCleanup();
                resp = std::move(finalResp);
            } else {
                // Handler rejected the stream: drain it to preserve framing.
                VVM_LOG_ERROR("serveConnection: streamed request type {} rejected, draining {} bytes",
                              req.type, nh.streamLen);
                uint64_t remaining = nh.streamLen;
                while (remaining > 0) {
                    uint64_t slice = remaining < kStreamSliceSize ? remaining : kStreamSliceSize;
                    if (!impl_->readAllTls(s, drain.data(), static_cast<size_t>(slice), tlsConn.get())) break;
                    remaining -= slice;
                }
            }
            if (resp.streamSinkCleanup) resp.streamSinkCleanup();
        } else {
            // Non-streamed request: single handler invocation.
            if (impl_->handler) {
                impl_->handler(req, resp);
            } else {
                resp.flags = TcpFlagsError;
            }
        }

        const bool sliceOut = (resp.streamSource != nullptr || resp.streamWindowProvide != nullptr) &&
                                   resp.streamLen > 0;
        const uint64_t respStreamLen = sliceOut ? resp.streamLen : static_cast<uint64_t>(resp.stream.size());

        NetHeader out{};
        std::memcpy(out.magic, kMagic, 4);
        out.version = kProtocolVersion;
        out.type = resp.type;
        out.flags = resp.flags;
        out.bodyLen = static_cast<uint32_t>(resp.body.size());
        out.seq = resp.seq;
        out.streamLen = respStreamLen;
        std::vector<uint8_t> outHeader = encodeHeader(out);
        if (!impl_->writeAllTls(s, outHeader.data(), outHeader.size(), tlsConn.get())) break;
        if (!resp.body.empty() && !impl_->writeAllTls(s, resp.body.data(), resp.body.size(), tlsConn.get())) break;
        if (sliceOut) {
            bool streamOk = false;
            if (resp.streamWindowProvide) {
                streamOk = writeStreamWindows(s, resp.streamWindowProvide, respStreamLen);
            } else {
                streamOk = writeStreamSlices(s, resp.streamSource, respStreamLen);
            }
            if (!streamOk) break;
            if (resp.streamCleanup) resp.streamCleanup();
        } else if (!resp.stream.empty()) {
            if (!impl_->writeAllTls(s, resp.stream.data(), resp.stream.size(), tlsConn.get())) break;
        }
    }

    {
        std::lock_guard<std::mutex> lock(impl_->connsMutex);
        impl_->conns.erase(connId);
    }
    closeSocket(s);
}

std::optional<TcpTransport::ConnId> TcpTransport::connect(const std::string& host, uint16_t port, int32_t timeoutMs) {
    ensureSocketsInit();

    if (timeoutMs <= 0) timeoutMs = 5000;

    SocketType s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == kInvalidSocket) {
        VVM_LOG_ERROR("socket() failed for client connection to {}:{}", host, port);
        return std::nullopt;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        VVM_LOG_ERROR("Invalid connect host: {}", host);
        closeSocket(s);
        return std::nullopt;
    }

#ifdef VVM_PLATFORM_WINDOWS
    u_long mode = 1;
    ioctlsocket(s, FIONBIO, &mode);
#else
    int fl = fcntl(s, F_GETFL, 0);
    fcntl(s, F_SETFL, fl | O_NONBLOCK);
#endif

    int rc = ::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    bool connected = (rc == 0);
    if (!connected && rc == kSocketError) {
#ifdef VVM_PLATFORM_WINDOWS
        bool inProgress = WSAGetLastError() == WSAEWOULDBLOCK;
#else
        bool inProgress = errno == EINPROGRESS;
#endif
        if (inProgress) {
            fd_set wfds;
            FD_ZERO(&wfds);
            FD_SET(s, &wfds);
            timeval tv{};
            tv.tv_sec = timeoutMs / 1000;
            tv.tv_usec = (timeoutMs % 1000) * 1000;
            int sel = select(static_cast<int>(s + 1), nullptr, &wfds, nullptr, &tv);
            if (sel == 1 && FD_ISSET(s, &wfds)) {
                int err = 0;
                SockLenType errLen = sizeof(err);
                if (getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&err), &errLen) == 0 && err == 0) {
                    connected = true;
                }
            }
        }
    }

#ifdef VVM_PLATFORM_WINDOWS
    {
        u_long mode = 0;
        ioctlsocket(s, FIONBIO, &mode);
    }
#else
    {
        int fl = fcntl(s, F_GETFL, 0);
        fcntl(s, F_SETFL, fl & ~O_NONBLOCK);
    }
#endif

    if (!connected) {
        VVM_LOG_ERROR("connect() to {}:{} timed out", host, port);
        closeSocket(s);
        return std::nullopt;
    }

    // TLS handshake for client — create per-connection SSL object
    std::unique_ptr<TlsConnection> tlsConn;
    if (impl_->tlsContext && impl_->tlsContext->enabled && !impl_->tlsContext->serverMode) {
        tlsConn = std::make_unique<TlsConnection>(impl_->tlsContext->context(), s);
        if (!tlsConn->connect(host)) {
            VVM_LOG_ERROR("TLS handshake failed for {}:{}", host, port);
            closeSocket(s);
            return std::nullopt;
        }
        // Enforce ALPN protocol negotiation
        if (!tlsConn->verifyAlpn(impl_->tlsConfig.alpnProtocols)) {
            VVM_LOG_ERROR("TLS ALPN negotiation failed for {}:{} (expected '{}')",
                          host, port, impl_->tlsConfig.alpnProtocols);
            closeSocket(s);
            return std::nullopt;
        }
    }

    setTimeouts(s, 60000);

    uint64_t id = impl_->nextConnId++;
    {
        std::lock_guard<std::mutex> lock(impl_->connsMutex);
        impl_->conns[id] = {s, std::chrono::steady_clock::now(), false, std::move(tlsConn)};
    }
    VVM_LOG_INFO("Connected to {}:{} (conn {})", host, port, id);
    return id;
}

std::optional<TcpMessage> TcpTransport::request(ConnId id, const TcpMessage& req, const StreamIO* streamIO) {
    SocketType s = kInvalidSocket;
    TlsConnection* tlsConn = nullptr;
    {
        std::lock_guard<std::mutex> lock(impl_->connsMutex);
        auto it = impl_->conns.find(id);
        if (it == impl_->conns.end()) return std::nullopt;
        s = it->second.socket;
        tlsConn = it->second.tlsConn.get();
        it->second.lastActivity = std::chrono::steady_clock::now();
    }

    std::lock_guard<std::mutex> ioLock(impl_->ioMutex);

    const bool sliceOut = streamIO != nullptr && streamIO->writeLen > 0;
    const uint64_t reqStreamLen = sliceOut ? streamIO->writeLen : static_cast<uint64_t>(req.stream.size());

    NetHeader out{};
    std::memcpy(out.magic, kMagic, 4);
    out.version = kProtocolVersion;
    out.type = req.type;
    out.flags = req.flags;
    out.bodyLen = static_cast<uint32_t>(req.body.size());
    out.seq = impl_->nextSeq_++;
    out.streamLen = reqStreamLen;
    std::vector<uint8_t> outHeader = encodeHeader(out);

    if (!impl_->writeAllTls(s, outHeader.data(), outHeader.size(), tlsConn)) {
        disconnect(id);
        return std::nullopt;
    }
    if (!req.body.empty() && !impl_->writeAllTls(s, req.body.data(), req.body.size(), tlsConn)) {
        disconnect(id);
        return std::nullopt;
    }
    if (sliceOut) {
        if (streamIO->onWindowProvide) {
            if (!writeStreamWindows(s, streamIO->onWindowProvide, reqStreamLen)) {
                disconnect(id);
                return std::nullopt;
            }
        } else if (!writeStreamSlices(s, streamIO->writeBuffer, reqStreamLen)) {
            disconnect(id);
            return std::nullopt;
        }
    } else if (!req.stream.empty()) {
        if (!impl_->writeAllTls(s, req.stream.data(), req.stream.size(), tlsConn)) {
            disconnect(id);
            return std::nullopt;
        }
    }

    std::vector<uint8_t> header(kHeaderSize);
    if (!impl_->readAllTls(s, header.data(), header.size(), tlsConn)) {
        disconnect(id);
        return std::nullopt;
    }

    NetHeader nh{};
    if (!decodeHeader(header.data(), header.size(), nh)) return std::nullopt;
    if (!isValidHeaderLengths(nh, impl_->netConfig)) {
        VVM_LOG_ERROR("TcpTransport::request: peer returned oversized message "
                      "(bodyLen={}, streamLen={}); dropping connection",
                      nh.bodyLen, nh.streamLen);
        disconnect(id);
        return std::nullopt;
    }

    TcpMessage resp;
    resp.type = nh.type;
    resp.flags = nh.flags;
    resp.seq = nh.seq;
    if (nh.bodyLen > 0) {
        resp.body.resize(nh.bodyLen);
        if (!impl_->readAllTls(s, resp.body.data(), nh.bodyLen, tlsConn)) {
            disconnect(id);
            return std::nullopt;
        }
    }
    if (nh.streamLen > 0) {
        if (streamIO != nullptr && streamIO->onWindowAcquire != nullptr) {
            // Windowed mode: stream into caller-supplied per-slice buffers.
            if (!readStreamWindows(s, streamIO->onWindowAcquire, streamIO->onWindowConsumed, nh.streamLen)) {
                disconnect(id);
                return std::nullopt;
            }
            resp.streamLen = nh.streamLen;
        } else if (streamIO != nullptr && streamIO->readBuffer != nullptr && streamIO->readLen >= nh.streamLen) {
            // Slice mode: read the response stream directly into the caller's buffer.
            if (!readStreamSlices(s, streamIO->readBuffer, nh.streamLen)) {
                disconnect(id);
                return std::nullopt;
            }
            resp.streamLen = nh.streamLen;
        } else {
            resp.stream.resize(nh.streamLen);
            if (!impl_->readAllTls(s, resp.stream.data(), resp.stream.size(), tlsConn)) {
                disconnect(id);
                return std::nullopt;
            }
        }
    }
    return resp;
}

void TcpTransport::disconnect(ConnId id) {
    SocketType s = kInvalidSocket;
    {
        std::lock_guard<std::mutex> lock(impl_->connsMutex);
        auto it = impl_->conns.find(id);
        if (it == impl_->conns.end()) return;
        s = it->second.socket;
        impl_->conns.erase(it);
    }
    closeSocket(s);
}

bool TcpTransport::isConnected(ConnId id) const {
    std::lock_guard<std::mutex> lock(impl_->connsMutex);
    return impl_->conns.find(id) != impl_->conns.end();
}

void TcpTransport::cleanupLoop() {
    while (!impl_->stopCleanup_) {
        {
            std::unique_lock<std::mutex> lock(impl_->cleanupMutex);
            impl_->cleanupCV.wait_for(lock, std::chrono::seconds(30),
                                      [this] { return impl_->stopCleanup_.load(); });
        }
        if (impl_->stopCleanup_) break;
        
        auto now = std::chrono::steady_clock::now();
        std::vector<uint64_t> toClose;
        
        {
            std::lock_guard<std::mutex> lock(impl_->connsMutex);
            for (auto& [id, conn] : impl_->conns) {
                auto idleTime = now - conn.lastActivity;
                if (idleTime > impl_->connectionIdleTimeout) {
                    toClose.push_back(id);
                }
            }
        }
        
        for (auto id : toClose) {
            VVM_LOG_INFO("Closing idle connection {}", id);
            disconnect(id);
        }
    }
}

}  // namespace network
}  // namespace vvm