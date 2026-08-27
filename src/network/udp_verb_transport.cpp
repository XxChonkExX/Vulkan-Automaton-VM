// UdpVerbTransport - implementation of docs/udp_mini_verbs.md (UVP v1).
//
// Threading model: one RX thread owns recvfrom() and mutates receive state;
// operation threads own their send loop and block on a per-transfer condvar
// until the RX thread signals ACK progress. All shared maps are guarded by
// mutex_. Memory semantics are host-copy only - no DMA lifetime hazards by
// design (a lost packet or dead peer can never corrupt device state).

#include "vulkan_vm/network/udp_verb_transport.hpp"

#include <algorithm>
#if defined(_WIN32)
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <map>
#include <set>

namespace vvm {
namespace network {

namespace {

// Platform socket primitives (self-contained: tcp_transport's helpers are
// private to that TU).
#if defined(VVM_PLATFORM_WINDOWS)
using SocketType = SOCKET;
static constexpr SocketType kSockInvalid = INVALID_SOCKET;
#else
using SocketType = int;
static constexpr SocketType kSockInvalid = -1;
#endif

constexpr uint16_t kMagic = 0x5556;  // 'UV'
constexpr uint8_t kVersion = 1;

enum class MsgType : uint8_t {
    Data = 1,
    Ack = 2,
    Register = 3,
    Unregister = 4,
    ReadReq = 5,
    Resend = 6,   // requester asks ONE missing DATA packet be re-sent
};

// Datagram sizing: stay within ONE MTU (~1500B Ethernet/Wi-Fi). Oversized
// datagrams shatter into dozens of IP fragments; any lost fragment destroys
// the whole packet and GBN resend-amplifies the loss into collapse.
constexpr size_t kDatagramMax = 1472;   // 1500 - 20 IPv4 - 8 UDP
// 36-byte fixed header (see docs/udp_mini_verbs.md).
constexpr size_t kHeaderSize = 36;
constexpr size_t kChunkPayload = kDatagramMax - kHeaderSize;
// Go-Back-N window in packets (~1.4 MB in flight at 1.4K chunks).
constexpr uint32_t kWindowPkts = 1024;
constexpr int kAckWaitMs = 20;      // per-round wait before resending window

#pragma pack(push, 1)
struct Header {
    uint16_t magic;
    uint8_t version;
    uint8_t type;
    uint32_t xid;
    uint32_t seq;
    uint32_t totalPkts;
    uint16_t payloadLen;
    uint16_t flags;
    uint64_t regionId;
    uint64_t offset;
};
static_assert(sizeof(Header) == 36, "UVP header must be 36 bytes");
#pragma pack(pop)

uint32_t nowMs() {
    return static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

uint64_t nowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace

// ============================================================================
// Per-transfer state shared between the RX thread and the operation thread.
// ============================================================================
struct UdpVerbTransport::Xfer {
    // Send side (initiator of a WRITE, or READREQ origin tracking ACKs)
    std::mutex m;
    std::condition_variable cv;
    uint32_t totalPkts = 0;
    uint32_t acked = 0;          // highest contiguous acked seq
    bool finished = false;
    bool ok = false;

    // Receive side (READ responses land here; WRITEs go straight into regions)
    bool expectData = false;     // true: we are collecting DATA packets
    uint8_t* dst = nullptr;      // read-resp destination buffer
    uint64_t dstOff = 0;         // base offset within dst (slice support)
    // Selective-repeat state for read responses:
    uint32_t contig = 0;                 // highest in-order seq + 1
    std::vector<bool> have;
    uint64_t remoteBase = 0;             // peer region offset of seq 0
    uint64_t reqBytes = 0;
    uint32_t lastProgressMs = 0;
    const uint8_t* src = nullptr;  // send side source buffer (WRITE)
};

struct UdpVerbTransport::Impl {
    explicit Impl(const NetworkConfig& cfg) : config_(cfg) {}

    NetworkConfig config_;
    SocketType sock = kSockInvalid;
    std::atomic<bool> running_{false};
    std::thread rxThread_;
    uint16_t boundPort_ = 0;
    std::atomic<uint64_t> nextXid_{1};
    std::atomic<uint64_t> nextRegionId_{1};
    std::atomic<uint64_t> nextConn_{1000};

    std::mutex mutex_;  // guards peers_/regions_/xfers_

    struct Region {
        void* ptr;
        uint64_t size;
    };
    std::map<uint64_t, Region> regions_;

    struct Peer {
        sockaddr_in addr{};
    };
    std::unordered_map<uint64_t, Peer> peers_;

    std::unordered_map<uint32_t, std::shared_ptr<Xfer>> xfers_;

    // WRITE-path contiguity tracking: xid -> (contiguousPkts, totalPkts,
    // lastMs). The receiver must ACK its TRUE progress or a GBN sender will
    // believe the transfer finished after one window.
    struct WriteProgress {
        uint32_t contig = 0;
        uint32_t total = 0;
        uint32_t lastMs = 0;
        std::set<uint32_t> seen;
    };
    std::unordered_map<uint32_t, WriteProgress> writeContig_;

    void evictWriteProgress() {
        const uint32_t now = nowMs();
        for (auto it = writeContig_.begin(); it != writeContig_.end();) {
            if (it->second.contig >= it->second.total &&
                now - it->second.lastMs > 2000) {
                it = writeContig_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Regions already announced to a given connection handle (REG sent once).
    std::unordered_map<uint64_t, uint64_t> announced_;  // conn -> last regionId? unused v1

    std::atomic<size_t> completionsPolled_{0};

    bool sendTo(const sockaddr_in& to, const void* buf, size_t len) const {
#if defined(VVM_PLATFORM_WINDOWS)
        int n = sendto(sock, static_cast<const char*>(buf), static_cast<int>(len),
                       0, reinterpret_cast<const sockaddr*>(&to), sizeof(to));
        return n == static_cast<int>(len);
#else
        ssize_t n = sendto(sock, static_cast<const char*>(buf),
                           static_cast<ssize_t>(len), 0,
                           reinterpret_cast<const sockaddr*>(&to), sizeof(to));
        return n == static_cast<ssize_t>(len);
#endif
    }

    static Header makeHdr(MsgType t, uint32_t xid, uint32_t seq, uint32_t total,
                          uint16_t plen, uint64_t regionId, uint64_t offset) {
        Header h{};
        h.magic = kMagic;
        h.version = kVersion;
        h.type = static_cast<uint8_t>(t);
        h.xid = xid;
        h.seq = seq;
        h.totalPkts = total;
        h.payloadLen = plen;
        h.flags = 0;
        h.regionId = regionId;
        h.offset = offset;
        return h;
    }

    sockaddr_in peerAddr(uint64_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = peers_.find(handle);
        return it != peers_.end() ? it->second.addr : sockaddr_in{};
    }

    void finish(uint32_t xid, bool ok) {
        auto it = xfers_.find(xid);
        if (it == xfers_.end()) return;
        {
            std::lock_guard<std::mutex> lk(it->second->m);
            it->second->finished = true;
            it->second->ok = ok;
        }
        it->second->cv.notify_all();
    }

    void rxThreadMain() {
        std::vector<uint8_t> buf(kDatagramMax + 64);
        while (running_) {
            sockaddr_in from{};
#if defined(VVM_PLATFORM_WINDOWS)
            int fromLen = sizeof(from);
            int n = recvfrom(sock, reinterpret_cast<char*>(buf.data()),
                             static_cast<int>(buf.size()), 0,
                             reinterpret_cast<sockaddr*>(&from), &fromLen);
#else
            socklen_t fromLen = sizeof(from);
            ssize_t n = recvfrom(sock, buf.data(),
                                 static_cast<ssize_t>(buf.size()), 0,
                                 reinterpret_cast<sockaddr*>(&from), &fromLen);
#endif
            if (!running_) break;
            if (n < static_cast<ssize_t>(kHeaderSize)) continue;

            Header h{};
            std::memcpy(&h, buf.data(), sizeof(h));
            if (h.magic != kMagic || h.version != kVersion) continue;
            const uint8_t* payload = buf.data() + kHeaderSize;

            std::lock_guard<std::mutex> lock(mutex_);
            switch (static_cast<MsgType>(h.type)) {
                case MsgType::Register: {
                    if (writeContig_.size() > 512) evictWriteProgress();
                    uint64_t sz = 0;
                    std::memcpy(&sz, payload, sizeof(sz));
                    // Receiver learns the remote staging size. v1: writes into
                    // registered host regions need no local staging; this map
                    // exists so future shadow-sync can validate sizes.
                    remoteRegionSizes_[h.regionId] = sz;
                    break;
                }
                case MsgType::Unregister:
                    remoteRegionSizes_.erase(h.regionId);
                    break;

                case MsgType::Data: {
                    std::shared_ptr<Xfer> xf;
                    auto xit = xfers_.find(h.xid);
                    if (xit != xfers_.end()) xf = xit->second;

                    uint8_t* dst = nullptr;
                    if (xf && xf->expectData && xf->dst) {
                        dst = xf->dst + xf->dstOff +
                              static_cast<uint64_t>(h.seq) * kChunkPayload;
                    } else {
                        auto rit = regions_.find(h.regionId);
                        if (rit != regions_.end()) {
                            const uint64_t end =
                                h.offset + h.payloadLen;
                            if (end <= rit->second.size) {
                                dst = static_cast<uint8_t*>(rit->second.ptr) +
                                      h.offset;
                            } else {
                                VVM_LOG_WARN("udp-verb: WRITE past region {} end "
                                             "(off={}+len={} > {}) - dropped",
                                             static_cast<unsigned long long>(h.regionId),
                                             static_cast<unsigned long long>(h.offset),
                                             h.payloadLen,
                                             static_cast<unsigned long long>(rit->second.size));
                            }
                        }
                    }
                    if (!dst) {
                        std::string keys;
                        for (auto& [k, v] : regions_) keys += std::to_string(k) + " ";
                        VVM_LOG_WARN("udp-verb[{}] DATA xid={} unknown region {} "
                                     "(have: {})", boundPort_, h.xid,
                                     static_cast<unsigned long long>(h.regionId),
                                     keys.empty() ? "<none>" : keys.c_str());
                        break;
                    }
                    std::memcpy(dst, payload, h.payloadLen);
                    if (xf && xf->expectData && h.seq < 3)
                        VVM_LOG_INFO("udp-verb[{}] RX s={} plen={} d0..3={:02x} {:02x} {:02x} {:02x}",
                                     boundPort_, h.seq, h.payloadLen,
                                     h.payloadLen>0?dst[0]:0, h.payloadLen>1?dst[1]:0,
                                     h.payloadLen>2?dst[2]:0, h.payloadLen>3?dst[3]:0);

                    if (xf && xf->expectData && xf->totalPkts == h.totalPkts) {
                        // Read-response path: bitmap + contiguous ACK.
                        if (h.seq < xf->totalPkts && !xf->have[h.seq]) {
                            xf->have[h.seq] = true;
                            while (xf->contig < xf->totalPkts &&
                                   xf->have[xf->contig])
                                ++xf->contig;
                            xf->lastProgressMs = nowMs();
                        }
                        if (xf->contig >= xf->totalPkts) finish(h.xid, true);
                        Header ack = makeHdr(MsgType::Ack, h.xid, xf->contig,
                                             xf->totalPkts, 0, h.regionId, 0);
                        sendTo(from, &ack, sizeof(ack));
                    } else {
                        // WRITE path: ACK true contiguous progress so the
                        // Go-Back-N sender knows what to resend.
                        auto& wp = writeContig_[h.xid];
                        wp.total = h.totalPkts;
                        wp.lastMs = nowMs();
                        if (wp.contig == h.seq) {
                            uint32_t expect = h.seq;
                            // Fast-forward over any already-seen tail packets.
                            while (expect < wp.total &&
                                   writeContig_[h.xid].seen.count(expect)) {
                                ++expect;
                            }
                            wp.contig = expect + 1;
                        }
                        wp.seen.insert(h.seq);
                        Header ack = makeHdr(MsgType::Ack, h.xid, wp.contig,
                                             h.totalPkts, 0, h.regionId, 0);
                        sendTo(from, &ack, sizeof(ack));
                    }
                    break;
                }

                case MsgType::Ack: {
                    auto it = xfers_.find(h.xid);
                    if (it == xfers_.end()) break;
                    {
                        std::lock_guard<std::mutex> lk(it->second->m);
                        if (h.seq > it->second->acked) it->second->acked = h.seq;
                        if (!it->second->finished &&
                            it->second->acked >= it->second->totalPkts) {
                            it->second->finished = true;
                            it->second->ok = true;
                        }
                    }
                    it->second->cv.notify_all();
                    break;
                }

                case MsgType::Resend: {
                    auto rit2 = regions_.find(h.regionId);
                    if (rit2 == regions_.end()) break;
                    const uint64_t off = h.offset;
if (off >= rit2->second.size) break;
size_t len = static_cast<size_t>((std::min)(kChunkPayload, rit2->second.size - off));
                    Header dh = makeHdr(MsgType::Data, h.xid, h.seq, h.totalPkts,
                                        static_cast<uint16_t>(len),
                                        h.regionId, off);
                    std::vector<uint8_t> pkt(sizeof(dh) + len);
                    std::memcpy(pkt.data(), &dh, sizeof(dh));
                    std::memcpy(pkt.data() + sizeof(dh),
                                static_cast<const uint8_t*>(rit2->second.ptr) +
                                    off,
                                len);
                    sendTo(from, pkt.data(), pkt.size());
                    break;
                }
                case MsgType::ReadReq: {
                    VVM_LOG_INFO("udp-verb[{}] READREQ recv xid={} region={}",
                                 boundPort_, h.xid,
                                 static_cast<unsigned long long>(h.regionId));
                    // We own the region: stream requested bytes to requester.
                    // Body carries the REQUESTED total length (u64 LE): the
                    // requester sizes its sink by this, not by our region.
                    uint64_t reqBytes = static_cast<uint64_t>(h.totalPkts) *
                                        kChunkPayload;
                    if (h.payloadLen == sizeof(reqBytes)) {
                        std::memcpy(&reqBytes, payload, sizeof(reqBytes));
                    }
                    auto rit = regions_.find(h.regionId);
                    if (rit == regions_.end()) {
                        VVM_LOG_WARN("udp-verb: READREQ for unknown region {}",
                                     static_cast<unsigned long long>(h.regionId));
                        break;
                    }
                    const auto* src = static_cast<const uint8_t*>(rit->second.ptr);
                    uint32_t streamed = 0;
                    for (uint32_t s = 0; s < h.totalPkts; ++s) {
                        const uint64_t rel =
                            static_cast<uint64_t>(s) * kChunkPayload;
                        if (rel >= reqBytes) break;
                        uint64_t off = h.offset + rel;
if (off >= rit->second.size) break;
uint64_t a = kChunkPayload;
uint64_t b = reqBytes - rel;
uint64_t c = rit->second.size - off;
uint64_t min_val = (std::min)((std::min)(a, b), c);
size_t len = static_cast<size_t>(min_val);
                        Header dh = makeHdr(MsgType::Data, h.xid, s, h.totalPkts,
                                            static_cast<uint16_t>(len),
                                            h.regionId, off);
                        std::vector<uint8_t> pkt(sizeof(dh) + len);
                        std::memcpy(pkt.data(), &dh, sizeof(dh));
                        if (len) std::memcpy(pkt.data() + sizeof(dh), src + off, len);
                        if (s < 3)
                            VVM_LOG_INFO("udp-verb[{}] TX s={} len={} b0..3={:02x} {:02x} {:02x} {:02x}",
                                         boundPort_, s, len,
                                         len>0?src[off]:0, len>1?src[off+1]:0,
                                         len>2?src[off+2]:0, len>3?src[off+3]:0);
                        sendTo(from, pkt.data(), pkt.size());
                        if ((s % kWindowPkts) == (kWindowPkts - 1))
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(1));  // pacing
                        ++streamed;
                        (void)streamed;
                    }
                    break;
                }

                default:
                    break;
            }
        }
    }

    std::unordered_map<uint64_t, uint64_t> remoteRegionSizes_;
};

UdpVerbTransport::UdpVerbTransport(const NetworkConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

UdpVerbTransport::~UdpVerbTransport() { shutdown(); }

bool UdpVerbTransport::initialize() {
    if (impl_->running_) return true;

    impl_->sock = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (impl_->sock < 0) {
        VVM_LOG_ERROR("udp-verb: socket() failed: {}", strerror(errno));
        return false;
    }

    int rcvbuf = 4 * 1024 * 1024;  // absorb bursts on Wi-Fi links
    setsockopt(impl_->sock, SOL_SOCKET, SO_RCVBUF,
               reinterpret_cast<const char*>(&rcvbuf), sizeof(rcvbuf));

    // Port convention matches the verbs transport: control port + offset.
    std::string listen = impl_->config_.listenAddress.empty()
                             ? "0.0.0.0"
                             : impl_->config_.listenAddress.substr(
                                   0, impl_->config_.listenAddress.rfind(':'));
    if (listen.empty()) listen = "0.0.0.0";
    (void)listen;  // v1 binds ANY; multi-address refinement later
    uint16_t basePort = 50050;
    auto colon = impl_->config_.listenAddress.rfind(':');
    if (colon != std::string::npos) {
        try {
            basePort = static_cast<uint16_t>(std::stoul(
                impl_->config_.listenAddress.substr(colon + 1)));
        } catch (...) {
            basePort = 50050;
        }
    }
    const uint16_t port = static_cast<uint16_t>(basePort + kRdmaPortOffset);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (::bind(impl_->sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        VVM_LOG_ERROR("udp-verb: bind on udp/{} failed: {}", port, strerror(errno));
        return false;
    }
    impl_->boundPort_ = port;
    localPort_ = port;

    impl_->running_ = true;
    impl_->rxThread_ = std::thread([this]() { impl_->rxThreadMain(); });

    VVM_LOG_INFO("udp-verb transport initialized on udp/{}", port);
    return true;
}

void UdpVerbTransport::shutdown() {
    if (!impl_->running_.exchange(false)) return;
    // ORDER MATTERS: closing the fd does NOT wake a thread blocked in
    // recvfrom() on Linux (observed: shutdown() joined forever). shutdown()
    // on the socket makes the blocked recvfrom return immediately.
#if defined(VVM_PLATFORM_WINDOWS)
    closesocket(impl_->sock);
#else
    ::shutdown(impl_->sock, SHUT_RDWR);
#endif
    if (impl_->rxThread_.joinable()) impl_->rxThread_.join();
#if defined(VVM_PLATFORM_WINDOWS)
    impl_->sock = kSockInvalid;
#else
    ::close(impl_->sock);
    impl_->sock = kSockInvalid;
#endif
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->peers_.clear();
    impl_->regions_.clear();
    impl_->remoteRegionSizes_.clear();
}

bool UdpVerbTransport::isReady() const {
    return impl_ && impl_->running_;
}

std::optional<RdmaMemoryRegion> UdpVerbTransport::registerGpuMemory(
    VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkBuffer buffer) {
    (void)memory; (void)offset; (void)buffer;
    // Device contents are unreachable from CLI contexts on locked platforms;
    // returning nullopt makes the cluster manager fall back to host-staged
    // automatically rather than advertising a region full of garbage.
    VVM_LOG_WARN("udp-verb: registerGpuMemory unsupported (host-staged fallback)");
    (void)size;
    return std::nullopt;
}

std::optional<RdmaMemoryRegion> UdpVerbTransport::registerHostMemory(
    void* ptr, size_t size) {
    if (!ptr || size == 0) return std::nullopt;
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    RdmaMemoryRegion r;
    r.addr = ptr;
    r.length = size;
    r.lkey = 0;
    r.rkey = static_cast<uint32_t>(impl_->nextRegionId_++);
    impl_->regions_[r.rkey] = {ptr, size};
    VVM_LOG_INFO("udp-verb: current regions: {}", [&]{ std::string out; for (auto& [k,v] : impl_->regions_) out += std::to_string(k) + " "; return out; }());
    return r;
}

void UdpVerbTransport::unregisterMemory(const RdmaMemoryRegion& region) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->regions_.erase(region.rkey);
}

std::optional<RdmaConnection> UdpVerbTransport::connect(
    const std::string& host, uint32_t port, uint32_t nodeIndex) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    // NOTE: `port` IS the peer's fabric data port (exact) - matching the
    // verbs backend where callers advertise concrete RDMA listener ports.
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        VVM_LOG_ERROR("udp-verb: invalid host {}", host);
        return std::nullopt;
    }
    uint64_t handle = impl_->nextConn_++;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        impl_->peers_[handle] = {addr};
    }
    RdmaConnection c;
    c.remoteHost = host;
    c.remotePort = port;
    c.remoteNodeIndex = nodeIndex;
    c.connected = true;
    c.qpNum = static_cast<uint32_t>(handle);
    c.internalId_ = reinterpret_cast<void*>(static_cast<uintptr_t>(handle));
    VVM_LOG_INFO("udp-verb: connected to {}:{}", host, port);
    return c;
}

void UdpVerbTransport::disconnect(const RdmaConnection& conn) {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    impl_->peers_.erase(reinterpret_cast<uintptr_t>(conn.internalId_));
}

std::vector<RdmaConnection> UdpVerbTransport::getConnections() const {
    return {};
}

bool UdpVerbTransport::rdmaWrite(const RdmaConnection& conn,
                                 const RdmaMemoryRegion& localRegion,
                                 uint64_t remoteAddr, uint32_t remoteRkey,
                                 VkDeviceSize size, uint64_t timeoutNs) {
    if (!isReady() || !conn.connected) return false;
    if (!localRegion.addr || size == 0) return size == 0;

    sockaddr_in to = impl_->peerAddr(reinterpret_cast<uintptr_t>(conn.internalId_));

    const uint32_t total =
        static_cast<uint32_t>((size + kChunkPayload - 1) / kChunkPayload);
    auto xf = std::make_shared<Xfer>();
    xf->totalPkts = total;
    xf->src = nullptr;
    const auto* src = static_cast<const uint8_t*>(localRegion.addr);

    const uint32_t xid = static_cast<uint32_t>(impl_->nextXid_++);
    {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        impl_->xfers_[xid] = xf;
    }

    // Announce region size once per write (cheap, idempotent at receiver).
    Header reg = Impl::makeHdr(MsgType::Register, xid, 0, total, sizeof(uint64_t),
                               remoteRkey, 0);
    uint64_t szBytes = size;
    std::vector<uint8_t> regPkt(sizeof(reg) + sizeof(szBytes));
    std::memcpy(regPkt.data(), &reg, sizeof(reg));
    std::memcpy(regPkt.data() + sizeof(reg), &szBytes, sizeof(szBytes));
    impl_->sendTo(to, regPkt.data(), regPkt.size());

    const uint64_t deadlineNs =
        timeoutNs == UINT64_MAX ? UINT64_MAX : nowNs() + timeoutNs;

    bool ok = false;
    uint32_t sentUpTo = 0;
    while (true) {
        {
            std::lock_guard<std::mutex> lk(xf->m);
            sentUpTo = (std::min)(xf->acked + kWindowPkts, total);
        }
        for (uint32_t s = (sentUpTo > kWindowPkts
                               ? sentUpTo - kWindowPkts
                               : 0);
             s < sentUpTo; ++s) {
uint64_t off = static_cast<uint64_t>(s) * kChunkPayload;
            size_t len = static_cast<size_t>(
                (std::min)(kChunkPayload, size - off));
            Header dh = Impl::makeHdr(MsgType::Data, xid, s, total,
                                      static_cast<uint16_t>(len),
                                      remoteRkey, remoteAddr + off);
            std::vector<uint8_t> pkt(sizeof(dh) + len);
            std::memcpy(pkt.data(), &dh, sizeof(dh));
            std::memcpy(pkt.data() + sizeof(dh), src + off, len);
            impl_->sendTo(to, pkt.data(), pkt.size());
        }

        std::unique_lock<std::mutex> lk(xf->m);
        if (xf->finished) {
            ok = xf->ok;
            break;
        }
        if (nowNs() >= deadlineNs) {
            xf->finished = true;
            ok = false;
            VVM_LOG_ERROR("udp-verb: WRITE timed out ({} bytes to region {})",
                          static_cast<unsigned long long>(size),
                          static_cast<unsigned long long>(remoteRkey));
            break;
        }
        xf->cv.wait_for(lk, std::chrono::milliseconds(kAckWaitMs));
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        impl_->xfers_.erase(xid);
    }
    impl_->completionsPolled_++;
    return ok;
}

bool UdpVerbTransport::rdmaRead(const RdmaConnection& conn,
                                const RdmaMemoryRegion& localRegion,
                                uint64_t remoteAddr, uint32_t remoteRkey,
                                VkDeviceSize size, uint64_t timeoutNs) {
    if (!isReady() || !conn.connected) return false;
    if (!localRegion.addr || size == 0) return size == 0;

    sockaddr_in to = impl_->peerAddr(reinterpret_cast<uintptr_t>(conn.internalId_));
    const uint32_t total =
        static_cast<uint32_t>((size + kChunkPayload - 1) / kChunkPayload);

    auto xf = std::make_shared<Xfer>();
    xf->totalPkts = total;
    xf->expectData = true;
    xf->dst = static_cast<uint8_t*>(localRegion.addr);
    xf->dstOff = remoteAddr;  // slice base: packets land at dst+doff+seq*chunk
    xf->have.assign(total, false);
    xf->remoteBase = remoteAddr;
    xf->reqBytes = static_cast<uint64_t>(size);
    xf->lastProgressMs = nowMs();

    const uint32_t xid = static_cast<uint32_t>(impl_->nextXid_++);
    {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        impl_->xfers_[xid] = xf;
    }

    Header req = Impl::makeHdr(MsgType::ReadReq, xid, 0, total,
                               sizeof(uint64_t), remoteRkey, remoteAddr);
    std::vector<uint8_t> reqPkt(sizeof(req) + sizeof(uint64_t));
    std::memcpy(reqPkt.data(), &req, sizeof(req));
    uint64_t reqBytes = static_cast<uint64_t>(size);
    std::memcpy(reqPkt.data() + sizeof(req), &reqBytes, sizeof(reqBytes));
    const bool sent = impl_->sendTo(to, reqPkt.data(), reqPkt.size());
    VVM_LOG_INFO("udp-verb[{}] READREQ xid={} sent={} -> udp/{}",
                 localPort_, xid, sent ? "yes" : "NO", ntohs(to.sin_port));

    const uint64_t deadlineNs =
        timeoutNs == UINT64_MAX ? UINT64_MAX : nowNs() + timeoutNs;

    // Selective-repeat driver: when progress stalls, explicitly request the
    // missing packets (READ responses are fire-and-forget on the wire; this
    // is what makes them reliable).
    bool ok = false;
    {
        std::unique_lock<std::mutex> lk(xf->m);
        while (!xf->finished) {
            if (nowNs() >= deadlineNs) {
                xf->finished = true;
                ok = false;
                VVM_LOG_ERROR("udp-verb: READ timed out ({}/{} packets from "
                              "region {})",
                              xf->contig, xf->totalPkts,
                              static_cast<unsigned long long>(remoteRkey));
                break;
            }
            if (xf->cv.wait_for(lk, std::chrono::milliseconds(kAckWaitMs)) ==
                std::cv_status::timeout) {
                // No fresh progress this tick: request up to 64 missing
                // packets, oldest-first.
                uint32_t requested = 0;
                for (uint32_t sq = xf->contig;
                     sq < xf->totalPkts && requested < 64; ++sq) {
                    if (!xf->have[sq]) {
                        uint64_t off = remoteAddr +
                                       static_cast<uint64_t>(sq) *
                                           kChunkPayload;
                        Header rr =
                            Impl::makeHdr(MsgType::Resend, xid, sq, total, 0,
                                          remoteRkey, off);
                        impl_->sendTo(to, &rr, sizeof(rr));
                        ++requested;
                    }
                }
            }
        }
        ok = xf->finished && xf->ok;
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        impl_->xfers_.erase(xid);
    }
    impl_->completionsPolled_++;
    return ok;
}

bool UdpVerbTransport::rdmaWriteAsync(
    const RdmaConnection& conn, const RdmaMemoryRegion& localRegion,
    uint64_t remoteAddr, uint32_t remoteRkey, VkDeviceSize size,
    CompletionCallback callback, uint64_t timeoutNs) {
    std::thread([=]() {
        bool ok = rdmaWrite(conn, localRegion, remoteAddr, remoteRkey, size,
                            timeoutNs);
        if (callback) callback(ok, ok ? "" : "udp-verb write failed");
    }).detach();
    return true;
}

bool UdpVerbTransport::rdmaReadAsync(
    const RdmaConnection& conn, const RdmaMemoryRegion& localRegion,
    uint64_t remoteAddr, uint32_t remoteRkey, VkDeviceSize size,
    CompletionCallback callback, uint64_t timeoutNs) {
    std::thread([=]() {
        bool ok = rdmaRead(conn, localRegion, remoteAddr, remoteRkey, size,
                           timeoutNs);
        if (callback) callback(ok, ok ? "" : "udp-verb read failed");
    }).detach();
    return true;
}

void UdpVerbTransport::flush() {}
size_t UdpVerbTransport::pollCompletions() {
    return impl_->completionsPolled_.exchange(0);
}

std::string UdpVerbTransport::getDeviceGuid() const {
    return "udp-" + std::to_string(localPort_);
}

std::unique_ptr<RdmaTransport> createUdpVerbRdmaTransport(
    const NetworkConfig& config, VkPhysicalDevice physicalDevice,
    VkDevice device) {
    (void)physicalDevice;
    (void)device;
    return std::make_unique<UdpVerbTransport>(config);
}

}  // namespace network
}  // namespace vvm

