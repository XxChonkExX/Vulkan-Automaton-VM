// CPU-only unit tests for the buddy allocator (no Vulkan device required).
//
// Covers: power-of-two enforcement, merge-on-deallocate, double-free
// validation, largest-free tracking, and fragmentation accounting.

#include <vulkan_vm/buddy_allocator.hpp>

#include <cstdio>

using vvm::BuddyAllocator;

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

int main() {
    constexpr VkDeviceSize kBlock = 1024 * 1024;      // 1 MB
    constexpr VkDeviceSize kMin = 256 * 1024;         // 256 KB

    BuddyAllocator alloc(kBlock, kMin);

    // --- 1. Power-of-two rounding ---
    // 400 KB request rounds up to 512 KB.
    auto a = alloc.allocate(400 * 1024);
    CHECK(a.has_value());
    CHECK(*a == 0);
    CHECK(alloc.getLargestFree() == 512 * 1024);

    // --- 2. Deterministic buddy placement ---
    auto b = alloc.allocate(256 * 1024);
    CHECK(b.has_value());
    CHECK(*b == 512 * 1024);

    auto c = alloc.allocate(256 * 1024);
    CHECK(c.has_value());
    CHECK(*c == 768 * 1024);

    // --- 3. Block is now full; further allocation must fail ---
    CHECK(!alloc.allocate(256 * 1024).has_value());
    CHECK(alloc.getLargestFree() == 0);

    // --- 4. Fragmentation: free a middle chunk, largest free stays small ---
    alloc.deallocate(*b, 256 * 1024);
    CHECK(alloc.getLargestFree() == 256 * 1024);

    // --- 5. Double-free tolerated (logged, no crash, no corruption) ---
    alloc.deallocate(*b, 256 * 1024);  // second free is a no-op
    CHECK(alloc.getLargestFree() == 256 * 1024);

    // --- 6. Coalescing: freeing both halves merges into one 512 KB region ---
    alloc.deallocate(*c, 256 * 1024);
    CHECK(alloc.getLargestFree() == 512 * 1024);

    // --- 7. Full coalesce back to the block size ---
    alloc.deallocate(*a, 512 * 1024);
    CHECK(alloc.getLargestFree() == kBlock);
    CHECK(alloc.getFragmentation() == 0.0f);

    // --- 8. Fragmentation ratio in a fragmented state ---
    {
        BuddyAllocator frag(kBlock, kMin);
        auto f1 = frag.allocate(256 * 1024);   // offset 0
        auto f2 = frag.allocate(256 * 1024);   // offset 256 KB
        auto f3 = frag.allocate(256 * 1024);   // offset 512 KB
        CHECK(f1 && f2 && f3);
        // Free the left-most; the split remainder at 768 KB stays but cannot
        // be merged past the busy 256-512 region, so largest free < total free.
        frag.deallocate(*f1, 256 * 1024);
        const VkDeviceSize largest = frag.getLargestFree();
        CHECK(largest != 0);
        CHECK(frag.getFragmentation() > 0.0f);
        CHECK(frag.getLargestFree() <= 256 * 1024);
    }

    // --- 9. Oversized + empty requests ---
    {
        BuddyAllocator o(kBlock, kMin);
        CHECK(!o.allocate(kBlock * 2).has_value());
        CHECK(!o.allocate(0).has_value());
    }

    if (failures == 0) {
        std::printf("=== ALL BUDDY TESTS PASSED (0 failures) ===\n");
        return 0;
    }
    std::printf("=== SOME TESTS FAILED (%d failures) ===\n", failures);
    return 1;
}