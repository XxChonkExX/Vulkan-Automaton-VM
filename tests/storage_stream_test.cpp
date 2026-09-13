// storage_stream_test: CPU-only pack + registry tests (no GPU needed).
// GPU streamer path is exercised in basic_test-style runs when a device
// exists; here we only verify layout math, LRU, and file round-trip so CI
// stays green on machines without Vulkan hardware.

#include "vulkan_vm/storage_stream.hpp"

#include <cassert>
#include <cstdio>
#include <iostream>

using namespace vvm::storage;

static void testAlign() {
    assert(alignUp(0, 4096) == 0);
    assert(alignUp(1, 4096) == 4096);
    assert(alignUp(4096, 4096) == 4096);
    assert(alignUp(4097, 4096) == 8192);
    std::cout << "  align ok\n";
}

static void testRegistry() {
    ExpertRegistry reg(2);
    reg.registerPack({{0, 0, 16ull << 20}, {1, 1ull << 20, 32ull << 20}, {2, 2ull << 20, 8ull << 20}});
    assert(!reg.touch(0).has_value());
    assert(!reg.touch(1).has_value());
    auto victim = reg.touch(2); // evicts 0 (LRU)
    assert(victim && *victim == 0);
    assert(reg.residentCount() == 2);
    assert(!reg.isResident(0) && reg.isResident(2));
    // prefetch plan: skips resident, largest-first
    auto plan = reg.planPrefetch({0, 1, 2});
    assert(plan.size() == 1 && plan[0] == 0);
    reg.evict(1);
    assert(reg.residentCount() == 1);
    std::cout << "  registry ok\n";
}

static void testPackRoundTrip() {
    const std::string path = "moe_pack_test.bin";
    std::vector<std::pair<uint32_t, std::vector<uint8_t>>> experts;
    experts.emplace_back(7, std::vector<uint8_t>(1 << 20, 0xAB));
    experts.emplace_back(3, std::vector<uint8_t>((1 << 20) + 13, 0xCD));
    experts.emplace_back(9, std::vector<uint8_t>{1, 2, 3, 4, 5});
    std::vector<ExpertSpec> specs;
    assert(writePackFile(path, experts, &specs));
    assert(specs.size() == 3);

    auto table = readPackTable(path);
    assert(table && table->size() == 3);
    for (auto& s : *table) {
        assert(s.fileOffset % kPackAlign == 0);
        auto blob = readPackBlob(path, s);
        assert(blob);
        if (s.expertId == 7) assert(blob->size() == (1u << 20) && (*blob)[0] == 0xAB);
        if (s.expertId == 9) assert(blob->size() == 5 && (*blob)[4] == 5);
    }
    std::remove(path.c_str());
    std::cout << "  pack round-trip ok\n";
}

int main() {
    std::cout << "storage_stream_test (CPU-only)\n";
    testAlign();
    testRegistry();
    testPackRoundTrip();

    // Capability probe must never throw, even with null device.
    StreamCaps caps = queryStreamCaps(VK_NULL_HANDLE);
    assert(caps.pureAvailable);
    std::cout << "  caps ok (dstorage=" << caps.directStorageAvailable << ")\n";

    std::cout << "All storage_stream tests passed!\n";
    return 0;
}
