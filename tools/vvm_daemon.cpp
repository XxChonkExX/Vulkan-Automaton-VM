// vvm_daemon - local control daemon exposing fabric/pool operations over a
// line protocol (docs/vvm_daemon_spec.md). Built for Android arm64 and
// desktop alike; zero dependencies beyond the network module.

#include "vulkan_vm/network/udp_verb_transport.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <fstream>
#include <sys/stat.h>
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
};

std::mutex g_mutex;
std::map<uint32_t, Region> g_regions;
std::atomic<uint32_t> g_nextRegion{1};
std::atomic<uint64_t> g_nextConn{1};
std::unique_ptr<RdmaTransport> g_transport;
std::mutex g_connMutex;
std::map<uint64_t, RdmaConnection> g_conns;

std::string toHex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        s += d[p[i] >> 4];
        s += d[p[i] & 15];
    }
    return s;
}

bool fromHex(const std::string& h, std::vector<uint8_t>& out) {
    if (h.size() % 2) return false;
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    out.clear();
    out.reserve(h.size() / 2);
    for (size_t i = 0; i < h.size(); i += 2) {
        int hi = nib(h[i]), lo = nib(h[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return true;
}

// Split "CMD k=v k=v" into verb + kv map.
std::string parseLine(const std::string& line, std::map<std::string, std::string>& kv) {
    size_t i = 0;
    while (i < line.size() && line[i] == ' ') ++i;
    size_t start = i;
    while (i < line.size() && line[i] != ' ') ++i;
    std::string verb = line.substr(start, i - start);
    bool firstExtra = true;
    while (i < line.size()) {
        while (i < line.size() && line[i] == ' ') ++i;
        start = i;
        while (i < line.size() && line[i] != ' ') ++i;
        std::string tok = line.substr(start, i - start);
        // path= values may contain spaces: consume the rest of the line.
        if (tok.rfind("path=", 0) == 0) {
            kv["path"] = line.substr(start + 5);
            break;
        }
        auto eq = tok.find('=');
        if (eq != std::string::npos)
            kv[tok.substr(0, eq)] = tok.substr(eq + 1);
        else if (firstExtra) { kv["sub"] = tok; firstExtra = false; }
    }
    return verb;
}

uint32_t needU32(const std::map<std::string, std::string>& kv, const char* k,
                 bool& ok) {
    auto it = kv.find(k);
    if (it == kv.end() || it->second.empty()) { ok = false; return 0; }
    return static_cast<uint32_t>(std::strtoul(it->second.c_str(), nullptr, 10));
}

std::string handleClient(int fd) {
    std::string inbuf, out;
    char rbuf[4096];
    bool run = true;
    while (run) {
        ssize_t n = ::recv(fd, rbuf, sizeof(rbuf), 0);
        if (n <= 0) break;
        inbuf.append(rbuf, static_cast<size_t>(n));
        size_t nl;
        while ((nl = inbuf.find('\n')) != std::string::npos) {
            std::string line = inbuf.substr(0, nl);
            inbuf.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;

            std::map<std::string, std::string> kv;
            std::string verb = parseLine(line, kv);
            std::string resp;

            if (verb == "QUIT") {
                out += "OK bye\n";
                run = false;
            } else if (verb == "PING") {
                resp = "OK pong=1";
            } else if (verb == "STATUS") {
                std::lock_guard<std::mutex> lk(g_mutex);
                resp = "OK backend=" +
                       (g_transport ? g_transport->getBackendName() : "none") +
                       " regions=" + std::to_string(g_regions.size()) +
                       " conns=" + std::to_string(g_conns.size());
            } else if (verb == "REGION" && kv.count("sub") && kv["sub"]=="file") {
                auto pit = kv.find("path");
                if (pit == kv.end()) { resp = "ERR code=2 msg=missing-path"; }
                else {
                    int fd = ::open(pit->second.c_str(), O_RDONLY);
                    struct stat st{};
                    if (fd < 0 || ::fstat(fd, &st) != 0 || st.st_size <= 0) {
                        if (fd >= 0) ::close(fd);
                        resp = "ERR code=4 msg=cannot-open-file";
                    } else {
                        void* m = ::mmap(nullptr, st.st_size, PROT_READ,
                                         MAP_PRIVATE, fd, 0);
                        ::close(fd);
                        if (m == MAP_FAILED) {
                            resp = "ERR code=6 msg=mmap-failed";
                        } else {
                            std::lock_guard<std::mutex> lk(g_mutex);
                            Region r;
                            r.fileBacked = true;
                            r.mapAddr = m; r.mapLen = st.st_size;
                            r.mem = {};  // unused for file-backed
                            // Point mem at the mapping so READ/PULL see it.
                            uint8_t* p8 = static_cast<uint8_t*>(m);
                            r.mem.assign(0, 0);
                            r.filePath = pit->second;
                            // emulate vector view without copying:
                            r.filePtr = p8; r.fileSize = st.st_size;
                            uint32_t id = g_nextRegion++;
                            if (g_transport) {
                                if (auto reg = g_transport->registerHostMemory(
                                        p8, st.st_size)) r.rkey = reg->rkey;
                            }
                            g_regions[id] = std::move(r);
                            resp = "OK id=" + std::to_string(id) +
                                   " rkey=" + std::to_string(g_regions[id].rkey) +
                                   " bytes=" + std::to_string(st.st_size);
                        }
                    }
                }
            } else if (verb == "SAVE") {
                bool ok = true;
                uint32_t id = needU32(kv, "id", ok);
                uint32_t off = needU32(kv, "off", ok);
                uint64_t len = strtoull(kv.count("len") ? kv["len"].c_str() : "0",
                                        nullptr, 10);
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
                out += "BEGIN\n";
                for (auto& [id, r] : g_regions)
                    out += "id=" + std::to_string(id) +
                           " rkey=" + std::to_string(r.rkey) +
                           " bytes=" + std::to_string(r.mem.size()) + "\n";
                out += "END\n";
                continue;
            } else if (verb == "PUT") {
                bool ok = true;
                uint32_t id = needU32(kv, "id", ok);
                uint32_t off = needU32(kv, "off", ok);
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
                            off + bytes.size() > it->second.mem.size()) {
                            resp = "ERR code=4 msg=no-such-region-or-oob";
                        } else {
                            std::memcpy(it->second.mem.data() + off, bytes.data(),
                                        bytes.size());
                            resp = "OK bytes=" + std::to_string(bytes.size());
                        }
                    }
                }
            } else if (verb == "GET") {
                bool ok = true;
                uint32_t id = needU32(kv, "id", ok);
                uint32_t off = needU32(kv, "off", ok);
                uint32_t len = needU32(kv, "len", ok);
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
                uint64_t connId =
                    std::strtoul(kv.count("conn") ? kv["conn"].c_str() : "",
                                 nullptr, 10);
                uint32_t rkey = needU32(kv, "rkey", ok);
                uint32_t reg = needU32(kv, kv.count("src") ? "src" : "dst", ok);
                uint32_t soff = needU32(kv, "soff", ok);
                uint32_t doff = needU32(kv, "doff", ok);
                uint32_t len = needU32(kv, "len", ok);

                RdmaConnection c;
                {
                    std::lock_guard<std::mutex> lk(g_connMutex);
                    auto cit = g_conns.find(connId);
                    if (cit != g_conns.end()) c = cit->second;
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
                    ssize_t w = ::send(fd, resp.data() + off, resp.size() - off, 0);
                    if (w <= 0) { run = false; break; }
                    off += static_cast<size_t>(w);
                }
            }
            if (!run) break;
        }
    }
    ::close(fd);
    return {};
}

}  // namespace

int main(int argc, char** argv) {
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
    g_transport->shutdown();
    return 0;
}
