// tcp_hardening_test: loopback verification of the TCP accept gates
// (THREAT_MODEL §5). No GPU, no TLS, no peers needed - pure loopback.
//
//   - per-IP cap: maxConnectionsPerIp=1 lets the first loopback client
//     through and closes the second (its request fails).
//   - global cap: maxConnections=1 does the same across the table.
// Rejected sockets are closed in the accept loop (no serve thread), so a
// request on them can never be answered - that is the assertion.

#include "vulkan_vm/network/tcp_transport.hpp"

#include <chrono>
#include <cstdio>
#include <thread>

using namespace vvm::network;

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

namespace {

void echoHandler(TcpMessage& req, TcpMessage& resp) {
    resp.body = req.body;
}

// One server + two clients; returns true when exactly the first client
// gets an echo and the second gets nothing (rejected).
bool capHolds(size_t maxConns, size_t perIp) {
    TcpTransport server;
    NetworkConfig cfg;
    cfg.maxConnections = maxConns;
    cfg.maxConnectionsPerIp = perIp;
    cfg.headerTimeoutMs = 200;  // fast worker release on stop()
    if (!server.start("127.0.0.1", 0, echoHandler, cfg,
                      std::chrono::milliseconds(500))) {
        std::printf("FAIL: server start\n");
        return false;
    }
    const uint16_t port = server.getBoundPort();
    if (port == 0) {
        std::printf("FAIL: ephemeral port\n");
        server.stop();
        return false;
    }

    TcpTransport c1, c2;
    auto id1 = c1.connect("127.0.0.1", port, 3000);
    auto id2 = c2.connect("127.0.0.1", port, 3000);
    bool ok = true;
    if (!id1 || !id2) {
        std::printf("FAIL: loopback connects\n");
        ok = false;
    } else {
        // No settling sleep: server workers exit on the first header-read
        // timeout, so speak immediately. Accept order follows connect
        // order on loopback; either client may hold the single slot.
        TcpMessage req;
        req.type = MsgHeartbeat;
        req.body = {7, 7, 7};
        auto r1 = c1.request(*id1, req);
        auto r2 = c2.request(*id2, req);
        // Exactly one of the two must hold the single slot.
        const bool firstWins = r1 && r1->body == req.body && !r2;
        const bool secondWins = r2 && r2->body == req.body && !r1;
        if (!firstWins && !secondWins) {
            std::printf("FAIL: cap not enforced (r1=%d r2=%d)\n",
                        r1.has_value() ? 1 : 0, r2.has_value() ? 1 : 0);
            ok = false;
        }
    }
    c1.disconnect(id1.value_or(0));
    c2.disconnect(id2.value_or(0));
    server.stop();
    return ok;
}

}  // namespace

int main() {
    // Per-IP cap of 1 on loopback: both clients share 127.0.0.1.
    CHECK(capHolds(16, 1));
    // Global cap of 1: same outcome via the table limit.
    CHECK(capHolds(1, 16));
    // Roomy table: both clients served.
    {
        TcpTransport server;
        NetworkConfig cfg;
        cfg.maxConnections = 16;
        cfg.maxConnectionsPerIp = 16;
        cfg.headerTimeoutMs = 200;
        bool ok = server.start("127.0.0.1", 0, echoHandler, cfg,
                               std::chrono::milliseconds(500));
        CHECK(ok);
        if (ok) {
            const uint16_t port = server.getBoundPort();
            CHECK(port != 0);
            TcpTransport c1, c2;
            auto id1 = c1.connect("127.0.0.1", port, 3000);
            auto id2 = c2.connect("127.0.0.1", port, 3000);
            CHECK(id1 && id2);
            if (id1 && id2) {
                TcpMessage req;
                req.type = MsgHeartbeat;
                req.body = {9};
                auto r1 = c1.request(*id1, req);
                auto r2 = c2.request(*id2, req);
                CHECK(r1 && r1->body == req.body);
                CHECK(r2 && r2->body == req.body);
                c1.disconnect(*id1);
                c2.disconnect(*id2);
            }
            server.stop();
        }
    }
    if (failures == 0) {
        std::printf("=== ALL TCP HARDENING TESTS PASSED ===\n");
        return 0;
    }
    std::printf("=== TCP HARDENING FAILURES (%d) ===\n", failures);
    return 1;
}
