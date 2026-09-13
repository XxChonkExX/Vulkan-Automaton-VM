// storage_pack_test: L0 pack format round-trip, width validation, checksum.
// CPU-only (no GPU). Exercises both PackTraits (64-bit offsets) and
// PackTraits32 (32-bit offsets).
//
// IMPORTANT: assert() compiles out under NDEBUG (Release). Every call that
// produces or consumes data (write/open/readShard) is a REAL call bound to a
// bool first; only the bool is asserted. Never put work inside assert().

#include "vulkan_vm/storage/pack_format.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

using namespace vvm::storage::pack;

static void fill(std::vector<uint8_t>& v, uint8_t seed) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint8_t>(seed + i);
}

template <class Traits>
static int roundTrip(const char* tag) {
    const std::string path = std::string("pack_test_") + tag + ".bin";

    std::vector<uint8_t> b0(1u << 20), b1(4096 + 13), b2(3);
    fill(b0, 0x10);
    fill(b1, 0x80);
    fill(b2, 0xF0);

    PackWriter<Traits> w(12); // 4 KiB nominal shards
    w.addShard(7, b0.data(), b0.size());
    w.addShard(3, b1.data(), b1.size());
    w.addShard(9, b2.data(), b2.size());
    const bool wrote = w.write(path);
    if (!wrote) { std::cerr << "  write failed\n"; return 1; }

    PackReader<Traits> r;
    const bool opened = r.open(path);
    if (!opened) { std::cerr << "  open failed\n"; return 1; }
    if (r.header().shardCount != 3) { std::cerr << "  bad count\n"; return 1; }
    if (r.header().magic != kPackMagic) { std::cerr << "  bad magic\n"; return 1; }
    if (r.header().version != kPackVersion) { std::cerr << "  bad version\n"; return 1; }
    if (r.table().size() != 3) { std::cerr << "  bad table size\n"; return 1; }

    for (const auto& e : r.table()) {
        if (e.offset % kBlobAlign != 0) { std::cerr << "  offset misaligned\n"; return 1; }
        std::vector<uint8_t> out;
        const bool read = r.readShard(e, out);
        if (!read) { std::cerr << "  shard read failed\n"; return 1; }
        if (PackReader<Traits>::hash(out.data(), out.size()) != e.hash) {
            std::cerr << "  hash mismatch\n"; return 1;
        }
        if (e.id == 7 && (out.size() != (1u << 20) || out[0] != 0x10)) { std::cerr << "  blob7 bad\n"; return 1; }
        if (e.id == 3 && (out.size() != 4096 + 13 || out[0] != 0x80)) { std::cerr << "  blob3 bad\n"; return 1; }
        if (e.id == 9 && (out.size() != 3 || out[0] != 0xF0)) { std::cerr << "  blob9 bad\n"; return 1; }
    }

    // A corrupted byte must change the checksum.
    {
        std::vector<uint8_t> out;
        const bool read = r.readShard(r.table()[0], out);
        if (!read) { std::cerr << "  corrupt-read failed\n"; return 1; }
        out[out.size() / 2] ^= 0xFF;
        if (PackReader<Traits>::hash(out.data(), out.size()) == r.table()[0].hash) {
            std::cerr << "  corruption not detected\n"; return 1;
        }
    }

    std::remove(path.c_str());
    std::cout << "  round-trip [" << tag << "] ok\n";
    return 0;
}

static int testWidthMismatch() {
    const std::string path = "pack_test_mismatch.bin";
    std::vector<uint8_t> b(1024, 0x55);
    PackWriter<PackTraits32> w32;
    w32.addShard(1, b.data(), b.size());
    if (!w32.write(path)) { std::cerr << "  w32 write failed\n"; return 1; }

    PackReader<PackTraits> r64;
    if (r64.open(path)) { std::cerr << "  width mismatch NOT rejected\n"; return 1; }

    PackReader<PackTraits32> r32;
    if (!r32.open(path)) { std::cerr << "  correct traits rejected\n"; return 1; }

    std::remove(path.c_str());
    std::cout << "  width mismatch reject ok\n";
    return 0;
}

static int testHashIdentity() {
    const uint8_t empty[1] = {0};
    if (fnv1a64(empty, 0) != 1469598103934665603ull) { std::cerr << "  fnv1a64 basis\n"; return 1; }
    if (fnv1a32(empty, 0) != 2166136261u) { std::cerr << "  fnv1a32 basis\n"; return 1; }

    const char msg[] = "vulkanvm";
    const uint8_t* p = reinterpret_cast<const uint8_t*>(msg);
    const size_t n = std::strlen(msg);
    if (fnv1a64(p, n) != fnv1a64(p, n)) { std::cerr << "  fnv1a64 nondet\n"; return 1; }
    if (fnv1a64(p, n) == fnv1a64(empty, 0)) { std::cerr << "  fnv1a64 collision\n"; return 1; }
    if (fnv1a32(p, n) != fnv1a32(p, n)) { std::cerr << "  fnv1a32 nondet\n"; return 1; }
    if (crc32(p, n) != crc32(p, n)) { std::cerr << "  crc32 nondet\n"; return 1; }

    const char other[] = "vulkanvm!";
    if (fnv1a64(p, n) == fnv1a64(reinterpret_cast<const uint8_t*>(other), std::strlen(other))) {
        std::cerr << "  fnv1a64 distinct-input collision\n"; return 1;
    }

    std::cout << "  hash identity ok\n";
    return 0;
}

int main() {
    std::cout << "storage_pack_test (CPU-only)\n";
    int rc = 0;
    rc |= roundTrip<PackTraits>("traits64");
    rc |= roundTrip<PackTraits32>("traits32");
    rc |= testWidthMismatch();
    rc |= testHashIdentity();

    // Self-describing flag round-trip.
    if (packFlags(offsetWidth(PackTraits::flags()), sizeWidth(PackTraits::flags()),
                   idWidth(PackTraits::flags()), hashWidth(PackTraits::flags())) != PackTraits::flags()) {
        std::cerr << "  flag round-trip failed\n";
        rc |= 1;
    }

    std::cout << (rc == 0 ? "All storage_pack tests passed!\n" : "storage_pack tests FAILED\n");
    return rc;
}