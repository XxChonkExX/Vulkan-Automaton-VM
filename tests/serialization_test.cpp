// Regression test for the canonical VVM wire format (wire_format.hpp).
// Catches byte-shuffle bugs like the getU64 p[4]/p[5]/p[6]/p[7] mixup that
// silently corrupted streamLen framing, plus ModelManifest round-trips.
// Pure CPU - no Vulkan device required.

#include "vulkan_vm/network/wire_format.hpp"
#include "vulkan_vm/network/model_types.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vvm::network;

static int failures = 0;
static int checks = 0;

static void expect(bool ok, const char* what) {
    checks++;
    if (!ok) {
        failures++;
        std::printf("FAIL: %s\n", what);
    }
}

static void roundTripU64(uint64_t value) {
    std::vector<uint8_t> buf;
    wire::putU64(buf, value);
    expect(buf.size() == 8, "putU64 emits 8 bytes");
    const uint8_t* p = buf.data();
    const uint8_t* end = p + buf.size();
    uint64_t out = 0;
    expect(wire::getU64(p, end, out) && out == value,
           "getU64 round-trips");
}

static void roundTripU32(uint32_t value) {
    std::vector<uint8_t> buf;
    wire::putU32(buf, value);
    const uint8_t* p = buf.data();
    const uint8_t* end = p + buf.size();
    uint32_t out = 0;
    expect(wire::getU32(p, end, out) && out == value,
           "getU32 round-trips");
}

int main() {
    // Boundary values for getU64 (the p[4..7] byte-shuffle bug broke these).
    roundTripU64(0x0000000000000000ull);
    roundTripU64(0x0000000000000001ull);
    roundTripU64(0x0102030405060708ull);  // every byte position distinct
    roundTripU64(0x000000000000ffffull);
    roundTripU64(0x00000000ffffffffull);  // 4 GB boundary
    roundTripU64(0x0000000100000000ull);  // >4 GB
    roundTripU64(0x123456789abcdef0ull);
    roundTripU64(0x7fffffffffffffffull);
    roundTripU64(0xffffffffffffffffull);

    roundTripU32(0);
    roundTripU32(1);
    roundTripU32(0x01020304);
    roundTripU32(0xffffffff);

    // String + bytes round trip.
    {
        std::vector<uint8_t> buf;
        wire::putStr(buf, std::string("hello/world"));
        const uint8_t* p = buf.data();
        const uint8_t* end = p + buf.size();
        std::string out;
        expect(wire::getStr(p, end, out) && out == "hello/world",
               "getStr round-trips");
        expect(p == end, "getStr consumes exactly the buffer");
    }
    {
        std::vector<uint8_t> bytes = {0xde, 0xad, 0xbe, 0xef};
        std::vector<uint8_t> buf;
        wire::putBytes(buf, bytes);
        const uint8_t* p = buf.data();
        const uint8_t* end = p + buf.size();
        std::vector<uint8_t> out;
        expect(wire::getBytes(p, end, out) && out == bytes,
               "getBytes round-trips");
    }

    // Truncation safety: every get* must refuse short buffers without
    // reading past the end or advancing the cursor.
    {
        uint8_t shortBuf[4] = {1, 2, 3, 4};
        const uint8_t* p = shortBuf;
        const uint8_t* end = p + 4;
        uint64_t out64 = 0;
        expect(!wire::getU64(p, end, out64), "getU64 refuses 4-byte buffer");
        expect(p == shortBuf, "getU64 does not advance on failure");
        uint32_t out32 = 0;
        expect(wire::getU32(p, end, out32) && out32 == 0x01020304,
               "getU32 succeeds on exact-size buffer");
    }
    {
        std::vector<uint8_t> buf;
        wire::putStr(buf, std::string("x"));
        buf.resize(1);  // drop the content, keep the 4-byte length
        const uint8_t* p = buf.data();
        const uint8_t* end = p + buf.size();
        std::string out;
        expect(!wire::getStr(p, end, out), "getStr refuses truncated string");
    }

    // ModelManifest schema round trip (the ModelHub wire contract).
    {
        ModelManifest m;
        m.modelId = "chonk/llama-3b";
        m.version = "v1";
        m.chunkSize = 4u * 1024 * 1024;
        ModelFileEntry f;
        f.path = "weights/model.safetensors";
        f.size = 0x0000000100000000ull;  // >4 GB: exercises getU64 in manifest
        for (int i = 0; i < 32; ++i) f.sha256[i] = static_cast<uint8_t>(i * 7);
        m.files.push_back(f);
        m.totalSize = f.size;

        auto blob = m.serialize();
        ModelManifest m2;
        expect(m2.deserialize(blob.data(), blob.size()),
               "ModelManifest deserializes");
        expect(m2.modelId == m.modelId && m2.version == m.version &&
                   m2.chunkSize == m.chunkSize && m2.files.size() == 1 &&
                   m2.files[0].path == f.path && m2.files[0].size == f.size &&
                   std::memcmp(m2.files[0].sha256, f.sha256, 32) == 0,
               "ModelManifest fields survive the wire round trip");
    }

    std::printf("=== WIRE FORMAT TESTS: %d checks, %d failures ===\n",
                checks, failures);
    return failures == 0 ? 0 : 1;
}
