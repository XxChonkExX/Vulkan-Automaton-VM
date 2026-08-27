// vvm_daemon - local control daemon exposing fabric/pool operations over a
// line protocol (docs/vvm_daemon_spec.md). Built for Android arm64 and
// desktop alike; zero dependencies beyond the network module.

#include "vulkan_vm/network/udp_verb_transport.hpp"

#if defined(VVM_PLATFORM_WINDOWS)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

using vvm::network::NetworkConfig;
using vvm::network::RdmaConnection;
using vvm::network::RdmaMemoryRegion;
using vvm::network::RdmaTransport;

namespace {

struct Region {
    std::vector<uint8_t> mem;
    uint32_t rkey = 0;   // 0 = not registered with fabric
    // File-backed variant: mem points at an mmap of filePath.
    bool fileBacked = false;
    void* mapAddr = nullptr;
    size_t mapLen = 0;
    std::string filePath;
    uint8_t* filePtr = nullptr;
    uint64_t fileSize = 0;

#if defined(VVM_PLATFORM_WINDOWS)
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMapping = nullptr;
#endif
};

std::mutex g_mutex;
std::map<uint32_t, Region> g_regions;
std::atomic<uint32_t> g_nextRegion{1};
std::atomic<uint64_t> g_nextConn{1};
std::unique_ptr<RdmaTransport> g_transport;
std::mutex g_connMutex;
std::map<uint64_t, RdmaConnection> g_conns;

static bool fromHex(const std::string& s, std::vector<uint8_t>& out) {
    if (s.size() % 2) return false;
    out.resize(s.size() / 2);
    for (size_t i = 0, o = 0; i < s.size(); i += 2, ++o) {
        char hi = s[i], lo = s[i + 1];
        auto val = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        int hv = val(hi), lv = val(lo);
        if (hv < 0 || lv < 0) return false;
        out[o] = static_cast<uint8_t>((hv << 4) | lv);
    }
    return true;
}

static std::string toHex(const uint8_t* data, size_t len) {
    static const char* d = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out.push_back(d[data[i] >> 4]);
        out.push_back(d[data[i] & 0xF]);
    }
    return out;
}

static bool needU32(const std::map<std::string, std::string>& kv,
                    const std::string& key, bool& ok,
                    uint32_t& out) {
    auto it = kv.find(key);
    if (it == kv.end()) { ok = false; return false; }
    char* end = nullptr;
    unsigned long v = std::strtoul(it->second.c_str(), &end, 10);
    if (end == it->second.c_str() || v > UINT32_MAX) { ok = false; return false; }
    out = static_cast<uint32_t>(v);
    return true;
}

void unmapFileRegion(Region& r) {
    if (r.fileBacked && r.mapAddr) {
#if defined(VVM_PLATFORM_WINDOWS)
        if (r.hMapping) {
            UnmapViewOfFile(r.mapAddr);
            CloseHandle(r.hMapping);
            r.hMapping = nullptr;
        }
        if (r.hFile != INVALID_HANDLE_VALUE) {
            CloseHandle(r.hFile);
            r.hFile = INVALID_HANDLE_VALUE;
        }
#else
        munmap(r.mapAddr, r.mapLen);
#endif
        r.mapAddr = nullptr;
        r.mapLen = 0;
        r.filePtr = nullptr;
        r.fileSize = 0;
    }
}

std::string handleClient(intptr_t fd) {
#if defined(VVM_PLATFORM_WINDOWS)
    SOCKET sock = static_cast<SOCKET>(fd);
#else
    int sock = fd;
#endif
    char rbuf[4096];
    std::string inbuf;
    bool run = true;
    std::string resp;

    while (run) {
#if defined(VVM_PLATFORM_WINDOWS)
        int n = recv(sock, rbuf, static_cast<int>(sizeof(rbuf)), 0);
#else
        ssize_t n = read(sock, rbuf, sizeof(rbuf));
#endif
        if (n <= 0) break;
        inbuf.append(rbuf, static_cast<size_t>(n));

        size_t pos;
        while ((pos = inbuf.find('\n')) != std::string::npos) {
            std::string line = inbuf.substr(0, pos);
            inbuf.erase(0, pos + 1);
            if (line.empty()) continue;

            std::map<std::string, std::string> kv;
            size_t start = 0;
            while (start < line.size()) {
                size_t eq = line.find('=', start);
                if (eq == std::string::npos) break;
                size_t sp = line.find(' ', eq);
                std::string k = line.substr(start, eq - start);
                std::string v = (sp == std::string::npos)
                                ? line.substr(eq + 1)
                                : line.substr(eq + 1, sp - eq - 1);
                kv[k] = v;
                start = (sp == std::string::npos) ? line.size() : sp + 1;
            }

            std::string verb = kv["verb"];
            if (verb == "STATUS") {
                resp = "OK version=1 "
                       "backend=" + (g_transport ? g_transport->getBackendName() : "none") +
                       " regions=" + std::to_string(g_regions.size()) +
                       " conns=" + std::to_string(g_conns.size());
            } else if (verb == "REGION" && kv.count("sub") && kv["sub"]=="file") {
                auto pit = kv.find("path");
                if (pit == kv.end()) { resp = "ERR code=2 msg=missing-path"; }
                else {
#if defined(VVM_PLATFORM_WINDOWS)
                    HANDLE hFile = CreateFileA(pit->second.c_str(), GENERIC_READ,
                                               FILE_SHARE_READ, nullptr,
                                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                    if (hFile == INVALID_HANDLE_VALUE) {
                        resp = "ERR code=4 msg=cannot-open-file";
                    } else {
                        LARGE_INTEGER size;
                        if (!GetFileSizeEx(hFile, &size) || size.QuadPart <= 0) {
                            CloseHandle(hFile);
                            resp = "ERR code=4 msg=cannot-open-file";
                        } else {
                            HANDLE hMap = CreateFileMapping(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
                            if (!hMap) {
                                CloseHandle(hFile);
                                resp = "ERR code=6 msg=mmap-failed";
                            } else {
                                void* m = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
                                if (!m) {
                                    CloseHandle(hMap);
                                    CloseHandle(hFile);
                                    resp = "ERR code=6 msg=mmap-failed";
                                } else {
                                    std::lock_guard<std::mutex> lk(g_mutex);
                                    Region r;
                                    r.fileBacked = true;
                                    r.mapAddr = m;
                                    r.mapLen = static_cast<size_t>(size.QuadPart);
                                    r.mem = {};
                                    uint8_t* p8 = static_cast<uint8_t*>(m);
                                    r.filePtr = p8;
                                    r.fileSize = size.QuadPart;
                                    r.hFile = hFile;
                                    r.hMapping = hMap;
                                    uint32_t id = g_nextRegion++;
                                    if (g_transport) {
                                        if (auto reg = g_transport->registerHostMemory(p8, r.mapLen))
                                            r.rkey = reg->rkey;
                                    }
                                    g_regions[id] = std::move(r);
                                    resp = "OK id=" + std::to_string(id) +
                                           " rkey=" + std::to_string(g_regions[id].rkey) +
                                           " bytes=" + std::to_string(r.mapLen);
                                }
                            }
                        }
                    }
#else
                    int ffd = ::open(pit->second.c_str(), O_RDONLY);
                    struct stat st{};
                    if (ffd < 0 || ::fstat(ffd, &st) != 0 || st.st_size <= 0) {
                        if (ffd >= 0) ::close(ffd);
                        resp = "ERR code=4 msg=cannot-open-file";
                    } else {
                        void* m = ::mmap(nullptr, st.st_size, PROT_READ,
                                         MAP_PRIVATE, ffd, 0);
                        ::close(ffd);
                        if (m == MAP_FAILED) {
                            resp = "ERR code=6 msg=mmap-failed";
                        } else {
                            std::lock_guard<std::mutex> lk(g_mutex);
                            Region r;
                            r.fileBacked = true;
                            r.mapAddr = m;
                            r.mapLen = st.st_size;
                            r.mem = {};
                            uint8_t* p8 = static_cast<uint8_t*>(m);
                            r.mem.assign(0, 0);
                            r.filePath = pit->second;
                            r.filePtr = p8;
                            r.fileSize = st.st_size;
                            uint32_t id = g_nextRegion++;
                            if (g_transport) {
                                if (auto reg = g_transport->registerHostMemory(p8, st.st_size))
                                    r.rkey = reg->rkey;
                            }
                            g_regions[id] = std::move(r);
                            resp = "OK id=" + std::to_string(id) +
                                   " rkey=" + std::to_string(g_regions[id].rkey) +
                                   " bytes=" + std::to_string(st.st_size);
                        }
                    }
#endif
                }
            } else if (verb == "SAVE") {
                bool ok = true;
                uint32_t id = 0;
                if (!needU32(kv, "id", ok, id)) { ok = false; }
                uint32_t off = 0;
                if (!needU32(kv, "off", ok, off)) { ok = false; }
                uint64_t len = 0;
                auto lit = kv.find("len");
                if (lit != kv.end()) {
                    len = strtoull(lit->second.c_str(), nullptr, 10);
                }
                auto fit = kv.find("file");
                std::lock_guard<std::mutex> lk(g_mutex);
                auto it = g_regions.find(id);
                const uint8_t* srcp =
                    it != g_regions.end()
                        ? (it->second.fileBacked ? it->second.filePtr
                                                 : it->second.mem.data())
                        : nullptr;
                const uint64_t totalsz =
                    it != g_regions.end()
                        ? (it->second.fileBacked ? it->second.fileSize
                                                 : it->second.mem.size())
                        : 0;
                if (!ok || fit == kv.end() || !srcp ||
                    off + len > totalsz) {
                    resp = "ERR code=4 msg=no-such-region-or-oob";
                } else {
                    std::ofstream f(fit->second, std::ios::binary | std::ios::trunc);
                    f.write(reinterpret_cast<const char*>(srcp) + off,
                            static_cast<std::streamsize>(len));
                    f.close();
                    resp = f ? "OK bytes=" + std::to_string(len)
                             : "ERR code=6 msg=file-write-failed";
                }
            } else if (verb == "REGION" && kv.count("sub") && kv["sub"]=="new") {
                size_t sz = std::strtoul(kv.count("size") ? kv["size"].c_str() : "0", nullptr, 10);
                if (!sz || sz > (8ull << 30)) {
                    resp = "ERR code=2 msg=bad-size";
                } else {
                    std::lock_guard<std::mutex> lk(g_mutex);
                    Region r;
                    r.mem.assign(sz, 0x00);
                    uint32_t id = g_nextRegion++;
                    if (g_transport) {
                        if (auto reg = g_transport->registerHostMemory(
                                r.mem.data(), r.mem.size())) {
                            r.rkey = reg->rkey;
                        }
                    }
                    g_regions[id] = std::move(r);
                    resp = "OK id=" + std::to_string(id) +
                           " rkey=" + std::to_string(g_regions[id].rkey) +
                           " bytes=" + std::to_string(sz);
                }
            } else if (verb == "REGION" && kv.count("sub") && kv["sub"]=="list") {
                std::lock_guard<std::mutex> lk(g_mutex);
                resp += "BEGIN\n";
                for (auto& [id, r] : g_regions)
                    resp += "id=" + std::to_string(id) +
                           " rkey=" + std::to_string(r.rkey) +
                           " bytes=" + std::to_string(r.fileBacked ? r.fileSize : r.mem.size()) + "\n";
                resp += "END\n";
                continue;
            } else if (verb == "PUT") {
                bool ok = true;
                uint32_t id = 0;
                if (!needU32(kv, "id", ok, id)) { ok = false; }
                uint32_t off = 0;
                if (!needU32(kv, "off", ok, off)) { ok = false; }
                auto hit = kv.find("hex");
                if (!ok || hit == kv.end()) { resp = "ERR code=2 msg=missing-arg"; }
                else {
                    std::vector<uint8_t> bytes;
                    if (!fromHex(hit->second, bytes)) {
                        resp = "ERR code=3 msg=bad-hex";
                    } else {
                        std::lock_guard<std::mutex> lk(g_mutex);
                        auto it = g_regions.find(id);
                        if (it == g_regions.end() ||
                            off + bytes.size() > (it->second.fileBacked ? it->second.fileSize : it->second.mem.size())) {
                            resp = "ERR code=4 msg=no-such-region-or-oob";
                        } else {
                            uint8_t* dst = it->second.fileBacked ? it->second.filePtr + off
                                                                  : it->second.mem.data() + off;
                            std::memcpy(dst, bytes.data(), bytes.size());
                            resp = "OK bytes=" + std::to_string(bytes.size());
                        }
                    }
                }
            } else if (verb == "GET") {
                bool ok = true;
                uint32_t id = 0;
                if (!needU32(kv, "id", ok, id)) { ok = false; }
                uint32_t off = 0;
                if (!needU32(kv, "off", ok, off)) { ok = false; }
                uint32_t len = 0;
                if (!needU32(kv, "len", ok, len)) { ok = false; }
                std::lock_guard<std::mutex> lk(g_mutex);
                auto it = g_regions.find(id);
                const uint8_t* base = nullptr; uint64_t cap = 0;
                if (it != g_regions.end()) {
                    if (it->second.fileBacked) { base = it->second.filePtr; cap = it->second.fileSize; }
                    else { base = it->second.mem.data(); cap = it->second.mem.size(); }
                }
                if (!ok || !base || off + len > cap) {
                    resp = "ERR code=4 msg=no-such-region-or-oob";
                } else {
                    resp = "OK hex=" + toHex(base + off,
                                             std::min<size_t>(len, 1024));
                }
            } else if (verb == "CONN" && kv.count("sub") && kv["sub"]=="add") {
                
                auto ipit = kv.find("ip");
                auto pit = kv.find("dataport");
                if (ipit == kv.end() || pit == kv.end()) {
                    resp = "ERR code=2 msg=missing-arg";
                } else {
                    auto c = g_transport->connect(
                        ipit->second,
                        std::strtoul(pit->second.c_str(), nullptr, 10));
                    if (!c) {
                        resp = "ERR code=5 msg=connect-failed";
                    } else {
                        uint64_t cid = g_nextConn++;
                        std::lock_guard<std::mutex> lk(g_connMutex);
                        g_conns[cid] = *c;
                        resp = "OK id=" + std::to_string(cid);
                    }
                }
            } else if (verb == "PUSH" || verb == "PULL") {
                bool ok = true;
                uint64_t connId = 0;
                auto cit = kv.find("conn");
                if (cit != kv.end()) {
                    connId = strtoull(cit->second.c_str(), nullptr, 10);
                }
                uint32_t rkey = 0;
                if (!needU32(kv, "rkey", ok, rkey)) { ok = false; }
                uint32_t reg = 0;
                std::string regKey = kv.count("src") ? "src" : "dst";
                if (!needU32(kv, regKey, ok, reg)) { ok = false; }
                uint32_t soff = 0;
                if (!needU32(kv, "soff", ok, soff)) { ok = false; }
                uint32_t doff = 0;
                if (!needU32(kv, "doff", ok, doff)) { ok = false; }
                uint32_t len = 0;
                if (!needU32(kv, "len", ok, len)) { ok = false; }

                RdmaConnection c;
                {
                    std::lock_guard<std::mutex> lk(g_connMutex);
                    auto cit2 = g_conns.find(connId);
                    if (cit2 != g_conns.end()) c = cit2->second;
                }
                std::lock_guard<std::mutex> lk(g_mutex);
                auto rit = g_regions.find(reg);
                uint8_t* rbase = nullptr; uint64_t rcap = 0;
                if (rit != g_regions.end()) {
                    if (rit->second.fileBacked) { rbase = rit->second.filePtr; rcap = rit->second.fileSize; }
                    else { rbase = rit->second.mem.data(); rcap = rit->second.mem.size(); }
                }
                uint64_t tmoNs = 30ull*1000*1000*1000;
                if (kv.count("tmo")) tmoNs = strtoull(kv["tmo"].c_str(), nullptr, 10)*1000000ull*1000ull;
                if (!ok || !c.connected || rit == g_regions.end() || !rbase ||
                    soff + len > rcap) {
                    resp = "ERR code=4 msg=bad-args";
                } else {
                    RdmaMemoryRegion rr;
                    rr.addr = rbase + soff;
                    rr.length = len;
                    bool okXfer =
                        verb == "PUSH"
                            ? g_transport->rdmaWrite(c, rr, doff, rkey, len, tmoNs)
                            : g_transport->rdmaRead(c, rr, doff, rkey, len, tmoNs);
                    resp = okXfer ? "OK bytes=" + std::to_string(len)
                                  : "ERR code=6 msg=fabric-xfer-failed";
                }
            } else {
                resp = "ERR code=1 msg=unknown-verb";
            }

            if (!resp.empty()) {
                resp += "\n";
                size_t off = 0;
                while (off < resp.size()) {
#if defined(VVM_PLATFORM_WINDOWS)
                    int w = send(sock, resp.data() + off, static_cast<int>(resp.size() - off), 0);
#else
                    ssize_t w = ::send(sock, resp.data() + off, resp.size() - off, 0);
#endif
                    if (w <= 0) { run = false; break; }
                    off += static_cast<size_t>(w);
                }
            }
            if (!run) break;
        }
    }
#if defined(VVM_PLATFORM_WINDOWS)
    closesocket(sock);
#else
    ::close(fd);
#endif
    return {};
}

}  // namespace

int main(int argc, char** argv) {
#if defined(VVM_PLATFORM_WINDOWS)
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::fprintf(stderr, "vvm-daemon: WSAStartup failed\n");
        return 1;
    }
#endif

    uint16_t ctrlPort = 53250;
    uint16_t dataPort = 53260;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--ctrl") && i + 1 < argc)
            ctrlPort = static_cast<uint16_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--data-port") && i + 1 < argc)
            dataPort = static_cast<uint16_t>(std::atoi(argv[++i]));
    }

    NetworkConfig cfg;
    cfg.listenAddress = "0.0.0.0:" + std::to_string(dataPort);
    g_transport = vvm::network::createUdpVerbRdmaTransport(cfg, nullptr, nullptr);
    if (!g_transport || !g_transport->initialize()) {
        std::fprintf(stderr, "vvm-daemon: transport init failed\n");
        return 1;
    }
    std::printf("vvm-daemon: ctrl tcp/%d  fabric udp/%d (%s)\n", ctrlPort,
                g_transport->getLocalPort(),
                g_transport->getBackendName().c_str());
    std::fflush(stdout);

#if defined(VVM_PLATFORM_WINDOWS)
    SOCKET srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv == INVALID_SOCKET) {
        std::fprintf(stderr, "vvm-daemon: socket failed\n");
        return 1;
    }
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(ctrlPort);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // device-local only
    if (bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR ||
        listen(srv, 4) == SOCKET_ERROR) {
        std::fprintf(stderr, "vvm-daemon: bind/listen failed\n");
        closesocket(srv);
        return 1;
    }

    while (true) {
        SOCKET fd = accept(srv, nullptr, nullptr);
        if (fd == INVALID_SOCKET) break;
        std::thread(handleClient, static_cast<intptr_t>(fd)).detach();
    }
    closesocket(srv);
#else
    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(ctrlPort);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // device-local only
    if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) ||
        ::listen(srv, 4)) {
        std::fprintf(stderr, "vvm-daemon: bind/listen failed\n");
        return 1;
    }

    while (true) {
        int fd = ::accept(srv, nullptr, nullptr);
        if (fd < 0) break;
        std::thread(handleClient, fd).detach();
    }
    ::close(srv);
#endif

    g_transport->shutdown();

#if defined(VVM_PLATFORM_WINDOWS)
    WSACleanup();
#endif
    return 0;
}