// udp_verb_test.cpp - loopback validation of the UDP mini-verb transport.
// Two UdpVerbTransport instances talk over 127.0.0.1: register host regions,
// WRITE push, READ pull, verify byte-exactness at sizes spanning one packet
// to hundreds of datagrams (Go-Back-N window exercise).

#include "vulkan_vm/network/udp_verb_transport.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

using vvm::network::NetworkConfig;
using vvm::network::RdmaConnection;
using vvm::network::RdmaMemoryRegion;
using vvm::network::RdmaTransport;

static NetworkConfig makeCfg(uint16_t port) {
    NetworkConfig cfg;
    cfg.listenAddress = "127.0.0.1:" + std::to_string(port);
    return cfg;
}

int main() {
    // This suite exercises the software fabric specifically; never let the
    // factory pick a hardware backend even where one is compiled in.
    setenv("VVM_RDMA_BACKEND", "udp", 1);
    constexpr uint16_t kPortA = 53201;
    constexpr uint16_t kPortB = 53202;

    auto ta = RdmaTransport::create(makeCfg(kPortA), nullptr, nullptr);
    auto tb = RdmaTransport::create(makeCfg(kPortB), nullptr, nullptr);
    CHECK(ta && tb);
    CHECK(ta->getBackendName() == "udp-mini-verb");
    CHECK(ta->initialize());
    CHECK(tb->initialize());

    // connect() derives the peer's data port by adding kRdmaPortOffset
    // itself - pass the CONTROL port, same as the cluster manager does.
    auto connAB = ta->connect("127.0.0.1", kPortB);
    CHECK(connAB.has_value());
    if (!connAB) return 1;

    // ---- Case matrix: single packet, exact window boundary-ish, multi-window
    const size_t sizes[] = {
        1024,                    // one datagram
        60000 * 3 + 12345,       // crosses several datagrams, ragged tail
        60000 * 300 + 7,         // ~18 MB: forces multiple GBN rounds
    };

    std::mt19937 rng(4242);

    for (size_t sz : sizes) {
        // +8 KiB headroom: writes start at remoteAddr=4096 inside the region.
        std::vector<uint8_t> src(sz), mirror(sz + 8192, 0xEE);
        for (size_t i = 0; i < sz; ++i) src[i] = static_cast<uint8_t>(rng());

        // B registers its receive buffer; A pushes into it.
        auto regionB = tb->registerHostMemory(mirror.data(), mirror.size());
        CHECK(regionB.has_value());
        auto scratchA = ta->registerHostMemory(src.data(), src.size());
        CHECK(scratchA.has_value());

        // WRITE: A -> B region
        bool okW = ta->rdmaWrite(*connAB, *scratchA, /*remoteAddr=*/4096,
                                 regionB->rkey, sz, /*timeoutNs=*/30ull * 1000 * 1000 * 1000);
        CHECK(okW);
        {
            const size_t cmpLen = std::min(sz, mirror.size() - 4096);
            const uint8_t* got = mirror.data() + 4096;
            for (size_t i = 0; i < cmpLen; ++i) {
                if (got[i] != src[i]) {
                    std::printf("WRITE DIFF @%zu: got %02x want %02x\n",
                                i, got[i], src[i]);
                    break;
                }
            }
        }
        CHECK(std::memcmp(mirror.data() + 4096, src.data(),
                          std::min(sz, mirror.size() - 4096)) == 0);

        // READ: pull back a subrange into fresh scratch on A from B's region
        const size_t readLen = std::min<size_t>(sz, 5 * 1024 * 1024 + 999);
        std::vector<uint8_t> sink(readLen + 8192, 0x11);
        auto sinkReg = ta->registerHostMemory(sink.data(), sink.size());
        CHECK(sinkReg.has_value());
        bool okR = ta->rdmaRead(*connAB, *sinkReg, /*remoteAddr=*/8192,
                                regionB->rkey, readLen,
                                /*timeoutNs=*/30ull * 1000 * 1000 * 1000);
        CHECK(okR);
        // rdmaRead fills the sink region starting at ITS offset 0.
        CHECK(std::memcmp(sink.data(), mirror.data() + 8192,
                          readLen) == 0);

        tb->unregisterMemory(*regionB);
        ta->unregisterMemory(*scratchA);
        ta->unregisterMemory(*sinkReg);
        std::printf("  case %zu bytes: write+read+verify OK\n", sz);
    }

    ta->shutdown();
    tb->shutdown();

    if (failures == 0) {
        std::printf("=== ALL UDP VERB TESTS PASSED ===\n");
        return 0;
    }
    std::printf("=== UDP VERB TESTS FAILED (%d) ===\n", failures);
    return 1;
}
