#include "vulkan_vm/network/tcp_transport.hpp"
#include "vulkan_vm/network/network_types.hpp"
#include "vulkan_vm/utils.hpp"

#include <atomic>
#include <cstring>
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
inline int socketRecv(SocketType s, char* buf, int len) { return static_cast<int>(recv(s, buf, len, 0)); }
inline int socketSend(SocketType s, const char* buf, int len) { return static_cast<int>(send(s, buf, len, 0)); }
using SockLenType = socklen_t;
#endif

namespace vvm {
namespace network {

// Internal implementation details
constexpr char kMagic[4] = {'V', 'V', 'M', 'N'};
constexpr uint8_t kProtocolVersion = 1;
constexpr size_t kHeaderSize = 32;
constexpr uint64_t kStreamSliceSize = 4ull * 1024 * 1024;  // 4MB slices (Spark-style)

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

// TLS context - single definition with conditional compilation inside methods
struct TlsContext {
#if defined(VVM_NETWORK_HAS_TLS)
    SSL_CTX* ctx = nullptr;
    SSL* ssl = nullptr;
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
        if (ssl) {
            SSL_shutdown(ssl);
            SSL_free(ssl);
            ssl = nullptr;
        }
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
            SSL_CTX_set_alpn_select_cb(ctx, [](SSL* ssl, const unsigned char** out, unsigned char* outlen, const unsigned char* in, unsigned int inlen, void* arg) -> int {
                const char* proto = static_cast<const char*>(arg);
                if (SSL_select_next_proto(const_cast<unsigned char**>(out), outlen, reinterpret_cast<const unsigned char*>(proto), strlen(proto), in, inlen) == OPENSSL_NPN_NEGOTIATED) {
                    return SSL_TLSEXT_ERR_OK;
                }
                return SSL_TLSEXT_ERR_NOACK;
            }, const_cast<char*>(config.alpnProtocols.c_str()));
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

        if (!config.caPath.empty()) {
            if (SSL_CTX_load_verify_locations(ctx, config.caPath.c_str(), nullptr) <= 0) {
                lastError = "Failed to load CA file: " + getOpenSslError();
                return false;
            }
            SSL_CTX_set_verify(ctx, config.verifyPeer ? SSL_VERIFY_PEER : SSL_VERIFY_NONE, nullptr);
        } else {
            SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);
        }

        // ALPN
        if (!config.alpnProtocols.empty()) {
            SSL_CTX_set_alpn_protos(ctx, reinterpret_cast<const unsigned char*>(config.alpnProtocols.c_str()), config.alpnProtocols.size());
        }

        enabled = true;
        serverMode = false;
        return true;
#else
        lastError = "TLS not compiled in (VVM_NETWORK_HAS_TLS=0)";
        return false;
#endif
    }

    bool accept(SocketType sock) {
#if defined(VVM_NETWORK_HAS_TLS)
        if (!enabled || !ctx) return false;
        ssl = SSL_new(ctx);
        if (!ssl) {
            lastError = "Failed to create SSL";
            return false;
        }
        SSL_set_fd(ssl, static_cast<int>(sock));
        if (SSL_accept(ssl) <= 0) {
            lastError = "SSL_accept failed: " + getOpenSslError();
            return false;
        }
        return true;
#else
        lastError = "TLS not compiled in";
        return false;
#endif
    }

    bool connect(SocketType sock, const std::string& host) {
#if defined(VVM_NETWORK_HAS_TLS)
        if (!enabled || !ctx) return false;
        ssl = SSL_new(ctx);
        if (!ssl) {
            lastError = "Failed to create SSL";
            return false;
        }
        SSL_set_fd(ssl, static_cast<int>(sock));
        SSL_set_tlsext_host_name(ssl, host.c_str());
        if (SSL_connect(ssl) <= 0) {
            lastError = "SSL_connect failed: " + getOpenSslError();
            return false;
        }
        return true;
#else
        lastError = "TLS not compiled in";
        return false;
#endif
    }

    int read(void* buf, int len) {
#if defined(VVM_NETWORK_HAS_TLS)
        if (!ssl) return -1;
        return SSL_read(ssl, buf, len);
#else
        lastError = "TLS not compiled in";
        return -1;
#endif
    }

    int write(const void* buf, int len) {
#if defined(VVM_NETWORK_HAS_TLS)
        if (!ssl) return -1;
        return SSL_write(ssl, buf, len);
#else
        lastError = "TLS not compiled in";
        return -1;
#endif
    }

    void shutdown() {
#if defined(VVM_NETWORK_HAS_TLS)
        if (ssl) {
            SSL_shutdown(ssl);
        }
#endif
    }

private:
#if defined(VVM_NETWORK_HAS_TLS)
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
    uint8_t rdma = 0, shadow = 0, dedicated = 0;
    uint32_t handleType = 0;
    if (!deserializeNodeId(p, end, out.owner)) return false;
    if (!detail::getU64(p, end, out.size)) return false;
    if (!detail::getU64(p, end, out.localAllocId)) return false;
    if (!detail::getU8(p, end, rdma)) return false;
    if (!detail::getU64(p, end, out.rdmaAddr)) return false;
    if (!detail::getU32(p, end, out.rkey)) return false;
    if (!detail::getU8(p, end, shadow)) return false;
    if (!detail::getU32(p, end, out.usageFlags)) return false;
    if (!detail::getU32(p, end, out.memoryTypeIndex)) return false;
    if (!detail::getU8(p, end, dedicated)) return false;
    if (!detail::getU32(p, end, handleType)) return false;
    if (!detail::getBytes(p, end, out.externalHandle)) return false;
    if (!detail::getU64(p, end, out.timestamp)) return false;
    if (!detail::getStr(p, end, out.allocationName)) return false;
    out.hasRdmaAddr = rdma != 0;
    out.hasHostShadow = shadow != 0;
    out.dedicatedAllocation = dedicated != 0;
    out.handleType = static_cast<ExternalHandleType>(handleType);
    return true;
}

// ============================================================================
// TcpTransport Implementation
// ============================================================================

struct TcpTransport::Impl {
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
    std::unordered_map<uint64_t, SocketType> conns;
    std::mutex ioMutex;  // serializes request/response exchanges

    // TLS
    std::unique_ptr<TlsContext> tlsContext;
    TlsConfig tlsConfig;

    ~Impl() { stop(); }

    // TLS-aware read/write
    bool writeAllTls(SocketType s, const void* buf, size_t len) {
        if (tlsContext && tlsContext->enabled) {
            const char* p = static_cast<const char*>(buf);
            size_t remaining = len;
            while (remaining > 0) {
                int chunk = static_cast<int>(remaining > static_cast<size_t>(INT_MAX) ? static_cast<size_t>(INT_MAX) : remaining);
                int n = tlsContext->write(p, chunk);
                if (n <= 0) return false;
                p += n;
                remaining -= static_cast<size_t>(n);
            }
            return true;
        } else {
            return writeAll(s, buf, len);
        }
    }

    bool readAllTls(SocketType s, void* buf, size_t len) {
        if (tlsContext && tlsContext->enabled) {
            char* p = static_cast<char*>(buf);
            size_t remaining = len;
            while (remaining > 0) {
                int chunk = static_cast<int>(remaining > static_cast<size_t>(INT_MAX) ? static_cast<size_t>(INT_MAX) : remaining);
                int n = tlsContext->read(p, chunk);
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
        if (listener != kInvalidSocket) {
            closeSocket(listener);
            listener = kInvalidSocket;
        }
        {
            std::lock_guard<std::mutex> lock(connsMutex);
            for (auto& [id, s] : conns) {
                if (s != kInvalidSocket) closeSocket(s);
            }
        }
        if (acceptThread.joinable()) acceptThread.join();
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
    }
};

TcpTransport::TcpTransport() : impl_(std::make_unique<Impl>()) {}

TcpTransport::~TcpTransport() {
    if (impl_) impl_->stop();
}

bool TcpTransport::start(const std::string& listenHost, uint16_t port, RequestHandler handler) {
    ensureSocketsInit();
    if (impl_->running) return false;

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

void TcpTransport::acceptLoop() {
    while (impl_->running) {
        sockaddr_in peer{};
        SockLenType addrLen = sizeof(peer);
        SocketType client = accept(impl_->listener, reinterpret_cast<sockaddr*>(&peer), &addrLen);
        if (client == kInvalidSocket) {
            if (!impl_->running) break;
            continue;
        }
        setTimeouts(client, 30000);
        uint64_t id = impl_->nextConnId++;
        {
            std::lock_guard<std::mutex> lock(impl_->connsMutex);
            impl_->conns[id] = client;
        }
        {
            std::lock_guard<std::mutex> lock(impl_->serveThreadsMutex);
            impl_->serveThreads.emplace_back([this, id, client]() { serveConnection(id, client); });
        }
    }
}

void TcpTransport::serveConnection(uint64_t connId, uintptr_t sRaw) {
    SocketType s = static_cast<SocketType>(sRaw);

    // TLS handshake for server
    if (impl_->tlsContext && impl_->tlsContext->enabled && impl_->tlsContext->serverMode) {
        if (!impl_->tlsContext->accept(s)) {
            VVM_LOG_ERROR("TLS handshake failed: {}", impl_->tlsContext->lastError);
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
        if (!impl_->readAllTls(s, header.data(), header.size())) break;

        NetHeader nh{};
        if (!decodeHeader(header.data(), header.size(), nh)) break;

        TcpMessage req;
        req.type = nh.type;
        req.flags = nh.flags;
        req.seq = nh.seq;
        if (nh.bodyLen > 0) {
            req.body.resize(nh.bodyLen);
            if (!impl_->readAllTls(s, req.body.data(), nh.bodyLen)) break;
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

            const bool accepted = impl_->handler != nullptr && resp.flags != TcpFlagsError && resp.streamSink != nullptr;
            if (accepted) {
                std::function<void()> sinkCleanup = resp.streamSinkCleanup;
                if (!readStreamSlices(s, resp.streamSink, nh.streamLen)) {
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
                    if (!impl_->readAllTls(s, drain.data(), static_cast<size_t>(slice))) break;
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

        const bool sliceOut = resp.streamSource != nullptr && resp.streamLen > 0;
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
        if (!impl_->writeAllTls(s, outHeader.data(), outHeader.size())) break;
        if (!resp.body.empty() && !impl_->writeAllTls(s, resp.body.data(), resp.body.size())) break;
        if (sliceOut) {
            if (!writeStreamSlices(s, resp.streamSource, respStreamLen)) break;
            if (resp.streamCleanup) resp.streamCleanup();
        } else if (!resp.stream.empty()) {
            if (!impl_->writeAllTls(s, resp.stream.data(), resp.stream.size())) break;
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

    // TLS handshake for client
    if (impl_->tlsContext && impl_->tlsContext->enabled && !impl_->tlsContext->serverMode) {
        if (!impl_->tlsContext->connect(s, host)) {
            VVM_LOG_ERROR("TLS handshake failed: {}", impl_->tlsContext->lastError);
            closeSocket(s);
            return std::nullopt;
        }
    }

    setTimeouts(s, 60000);

    uint64_t id = impl_->nextConnId++;
    {
        std::lock_guard<std::mutex> lock(impl_->connsMutex);
        impl_->conns[id] = s;
    }
    VVM_LOG_INFO("Connected to {}:{} (conn {})", host, port, id);
    return id;
}

std::optional<TcpMessage> TcpTransport::request(ConnId id, const TcpMessage& req, const StreamIO* streamIO) {
    SocketType s = kInvalidSocket;
    {
        std::lock_guard<std::mutex> lock(impl_->connsMutex);
        auto it = impl_->conns.find(id);
        if (it == impl_->conns.end()) return std::nullopt;
        s = it->second;
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

    if (!impl_->writeAllTls(s, outHeader.data(), outHeader.size())) return std::nullopt;
    if (!req.body.empty() && !impl_->writeAllTls(s, req.body.data(), req.body.size())) return std::nullopt;
    if (sliceOut) {
        if (!writeStreamSlices(s, streamIO->writeBuffer, reqStreamLen)) return std::nullopt;
    } else if (!req.stream.empty()) {
        if (!impl_->writeAllTls(s, req.stream.data(), req.stream.size())) return std::nullopt;
    }

    std::vector<uint8_t> header(kHeaderSize);
    if (!impl_->readAllTls(s, header.data(), header.size())) return std::nullopt;

    NetHeader nh{};
    if (!decodeHeader(header.data(), header.size(), nh)) return std::nullopt;

    TcpMessage resp;
    resp.type = nh.type;
    resp.flags = nh.flags;
    resp.seq = nh.seq;
    if (nh.bodyLen > 0) {
        resp.body.resize(nh.bodyLen);
        if (!impl_->readAllTls(s, resp.body.data(), nh.bodyLen)) return std::nullopt;
    }
    if (nh.streamLen > 0) {
        if (streamIO != nullptr && streamIO->readBuffer != nullptr && streamIO->readLen >= nh.streamLen) {
            // Slice mode: read the response stream directly into the caller's buffer.
            if (!readStreamSlices(s, streamIO->readBuffer, nh.streamLen)) return std::nullopt;
            resp.streamLen = nh.streamLen;
        } else {
            resp.stream.resize(nh.streamLen);
            if (!impl_->readAllTls(s, resp.stream.data(), resp.stream.size())) return std::nullopt;
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
        s = it->second;
        impl_->conns.erase(it);
    }
    closeSocket(s);
}

bool TcpTransport::isConnected(ConnId id) const {
    std::lock_guard<std::mutex> lock(impl_->connsMutex);
    return impl_->conns.find(id) != impl_->conns.end();
}

}  // namespace network
}  // namespace vvm