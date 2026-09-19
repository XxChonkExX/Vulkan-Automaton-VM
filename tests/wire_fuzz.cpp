// wire_fuzz: parser hardening for the VVM TCP wire format
// (THREAT_MODEL §5: fuzz the wire parser with the hard caps asserted).
//
// Two harnesses in one TU:
//   - Deterministic self-test (default main): primitive round-trips,
//     truncation matrices, adversarial lengths, and NodeInfo/NodeList
//     cap checks. Runs on every CI runner, no sanitizer needed.
//   - libFuzzer entry (FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION):
//       clang++ -std=c++20 -fsanitize=fuzzer,address \
//         tests/wire_fuzz.cpp -Iinclude -o wire_fuzz_libfuzzer
//       ./wire_fuzz_libfuzzer -max_total_time=300
//
// Invariants asserted on EVERY input: getters never advance the cursor
// past `end`, never read out of bounds, and NodeInfo/NodeList semantic
// caps (64 GPUs, 4096 nodes) hold.

#include "vulkan_vm/utils.hpp"
#include "vulkan_vm/network/wire_format.hpp"
#include "vulkan_vm/network/network_types.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace detail = ::vvm::detail;
namespace wire = ::vvm::network::wire;

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

namespace {

// Feed arbitrary bytes through every NodeInfo/NodeList entry point and
// assert the hard-cap + bounds invariants (libFuzzer body + corpus core).
int fuzzOne(const uint8_t* data, size_t size) {
    int local = 0;
    // NodeInfo parse: must not crash; cursor stays within [begin, end].
    {
        ::vvm::network::NodeInfo info;
        const uint8_t* p = data;
        const uint8_t* end = data + size;
        (void)deserializeNodeInfo(p, end, info);
        if (p < data || p > end) ++local;
        if (info.gpuDevices.size() > 64) ++local;  // semantic cap
    }
    // NodeList parse.
    {
        std::vector<uint8_t> buf(data, data + size);
        std::vector<::vvm::network::NodeInfo> out;
        (void)deserializeNodeList(buf, out);
        if (out.size() > 4096) ++local;  // semantic cap
    }
    // Primitive getters: truncated reads fail, cursor never escapes.
    {
        const uint8_t* p = data;
        const uint8_t* end = data + size;
        uint64_t u64 = 0;
        uint32_t u32 = 0;
        std::string s;
        std::vector<uint8_t> b;
        (void)detail::getU64(p, end, u64);
        (void)detail::getU32(p, end, u32);
        (void)detail::getStr(p, end, s);
        (void)detail::getBytes(p, end, b);
        if (p < data || p > end) ++local;
        if (s.size() > size || b.size() > size) ++local;  // never over-read
    }
    return local;
}

void deterministicCorpus() {
    // 1. Primitive round-trips (little-endian detail:: dialect).
    {
        std::vector<uint8_t> buf;
        detail::putU8(buf, 0xAB);
        detail::putU32(buf, 0xDEADBEEFu);
        detail::putU64(buf, 0x0123456789ABCDEFull);
        detail::putStr(buf, "mlx5_0");
        const std::vector<uint8_t> payload = {1, 2, 3, 4, 5};
        detail::putBytes(buf, payload);
        const uint8_t* p = buf.data();
        const uint8_t* end = p + buf.size();
        uint8_t u8 = 0;
        uint32_t u32 = 0;
        uint64_t u64 = 0;
        std::string s;
        std::vector<uint8_t> b;
        CHECK(detail::getU8(p, end, u8) && u8 == 0xAB);
        CHECK(detail::getU32(p, end, u32) && u32 == 0xDEADBEEFu);
        CHECK(detail::getU64(p, end, u64) && u64 == 0x0123456789ABCDEFull);
        CHECK(detail::getStr(p, end, s) && s == "mlx5_0");
        CHECK(detail::getBytes(p, end, b) && b == payload);
        CHECK(p == end);
    }
    // 2. Big-endian wire:: dialect round-trip.
    {
        std::vector<uint8_t> buf;
        wire::putU32(buf, 0x01020304u);
        wire::putU64(buf, 0x1112131415161718ull);
        const uint8_t* p = buf.data();
        const uint8_t* end = p + buf.size();
        uint32_t u32 = 0;
        uint64_t u64 = 0;
        CHECK(wire::getU32(p, end, u32) && u32 == 0x01020304u);
        CHECK(wire::getU64(p, end, u64) && u64 == 0x1112131415161718ull);
        CHECK(p == end);
    }
    // 3. Truncation matrix: every strict prefix must fail cleanly
    // (no crash, cursor contained); only the full buffer parses.
    {
        std::vector<uint8_t> buf;
        detail::putU64(buf, 0xFFFFFFFFFFFFFFFFull);
        detail::putStr(buf, "hello-cluster-node-00");
        for (size_t n = 0; n < buf.size(); ++n) {
            const uint8_t* p = buf.data();
            const uint8_t* end = p + n;
            uint64_t u64 = 0;
            const bool ok64 = detail::getU64(p, end, u64);
            CHECK(ok64 == (n >= 8));
            if (ok64) CHECK(u64 == 0xFFFFFFFFFFFFFFFFull);
            std::string s;
            const bool okStr = detail::getStr(p, end, s);
            CHECK(okStr == (n == buf.size()));
            if (okStr) CHECK(s == "hello-cluster-node-00");
            CHECK(p >= buf.data() && p <= end);
        }
    }
    // 4. Adversarial lengths: u32-max string/bytes length, empty tail.
    // The length prefix itself is consumed (4 bytes); the body is refused
    // and the cursor stops right after the prefix.
    {
        const uint8_t evil[] = {0xFF, 0xFF, 0xFF, 0xFF};
        const uint8_t* p = evil;
        const uint8_t* end = evil + sizeof(evil);
        std::string s;
        std::vector<uint8_t> b;
        CHECK(!detail::getStr(p, end, s));
        CHECK(p == evil + 4);
        p = evil;
        CHECK(!detail::getBytes(p, end, b));
        CHECK(p == evil + 4);
    }
    // 5. NodeInfo round-trip + cap rejections.
    {
        ::vvm::network::NodeInfo info;
        info.id.host = "10.0.0.7";
        info.id.port = 50051;
        info.id.nodeIndex = 3;
        info.id.uuid = "node-uuid-7";
        info.gpuDevices = {"AMD Radeon RX 7900 XTX", "Intel Arc Pro B70"};
        info.nicName = "mlx5_0";
        info.rdmaCapable = true;
        info.timestamp = 123456789ull;
        const std::vector<uint8_t> buf = serializeNodeInfo(info);
        ::vvm::network::NodeInfo back;
        const uint8_t* p = buf.data();
        CHECK(deserializeNodeInfo(p, buf.data() + buf.size(), back));
        CHECK(back.id.host == "10.0.0.7" && back.id.port == 50051);
        CHECK(back.gpuDevices == info.gpuDevices);
        CHECK(back.nicName == "mlx5_0" && back.rdmaCapable);
        CHECK(back.timestamp == 123456789ull);
        // gpuCount = 65 must be rejected (cap 64). Layout: str(host)
        // + u32(port) + u32(index) + str(uuid), then the count.
        std::vector<uint8_t> evil;
        detail::putStr(evil, "");
        detail::putU32(evil, 0);
        detail::putU32(evil, 0);
        detail::putStr(evil, "");
        detail::putU32(evil, 65);  // gpuCount over cap
        ::vvm::network::NodeInfo no;
        const uint8_t* q = evil.data();
        CHECK(!deserializeNodeInfo(q, evil.data() + evil.size(), no));
        // NodeList count = 5000 must be rejected (cap 4096).
        std::vector<uint8_t> evilList;
        detail::putU32(evilList, 5000);
        std::vector<::vvm::network::NodeInfo> out;
        CHECK(!deserializeNodeList(evilList, out));
    }
    // 6. Byte-at-a-time garbage sweep over the fuzz body.
    {
        uint8_t Rhough[256];
        for (int i = 0; i < 256; ++i) Rhough[i] = static_cast<uint8_t>(i * 31 + 7);
        CHECK(fuzzOne(Rhough, sizeof(Rhough)) == 0);
        for (size_t n = 0; n <= sizeof(Rhough); n += 7) {
            CHECK(fuzzOne(Rhough, n) == 0);
        }
    }
}

}  // namespace

#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (fuzzOne(data, size) != 0) std::abort();
    return 0;
}
#else
int main() {
    deterministicCorpus();
    if (failures == 0) {
        std::printf("=== ALL WIRE FUZZ SELF-TESTS PASSED ===\n");
        return 0;
    }
    std::printf("=== WIRE FUZZ FAILURES (%d) ===\n", failures);
    return 1;
}
#endif
