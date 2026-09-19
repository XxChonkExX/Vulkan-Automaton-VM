// memory_type_test: CPU-only unit tests for largest-heap pure
// DEVICE_LOCAL selection (no Vulkan device required).
//
// Covers: skips ReBAR (HOST_VISIBLE) types even on bigger heaps, picks the
// pure type on the largest heap (not first-match), ignores out-of-range
// heap indices, and returns nullopt when no pure type exists.

#include <vulkan_vm/utils.hpp>

#include <cstdio>

using vvm::findLargestHeapPureDeviceLocal;

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

namespace {

constexpr VkMemoryPropertyFlags kPure =
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
constexpr VkMemoryPropertyFlags kRebar =
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

VkPhysicalDeviceMemoryProperties makeProps() {
    VkPhysicalDeviceMemoryProperties p{};
    p.memoryHeapCount = 2;
    p.memoryHeaps[0].size = 256ull * 1024ull * 1024ull;   // small heap
    p.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    p.memoryHeaps[1].size = 8ull * 1024ull * 1024ull * 1024ull;  // big heap
    p.memoryHeaps[1].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    return p;
}

}  // namespace

int main() {
    // --- 1. NVIDIA-like: early pure type on a small heap must NOT win ---
    {
        auto p = makeProps();
        p.memoryTypeCount = 3;
        p.memoryTypes[0].propertyFlags = kPure;   // 256 MB heap
        p.memoryTypes[0].heapIndex = 0;
        p.memoryTypes[1].propertyFlags = kRebar;  // 8 GB heap, skipped
        p.memoryTypes[1].heapIndex = 1;
        p.memoryTypes[2].propertyFlags = kPure;   // 8 GB heap, winner
        p.memoryTypes[2].heapIndex = 1;
        const auto got = findLargestHeapPureDeviceLocal(p);
        CHECK(got.has_value());
        CHECK(got.value_or(99) == 2);
    }

    // --- 2. All ReBAR (Intel-style): nullopt, caller keeps first-match ---
    {
        auto p = makeProps();
        p.memoryTypeCount = 2;
        p.memoryTypes[0].propertyFlags = kRebar;
        p.memoryTypes[0].heapIndex = 0;
        p.memoryTypes[1].propertyFlags = kRebar;
        p.memoryTypes[1].heapIndex = 1;
        CHECK(!findLargestHeapPureDeviceLocal(p).has_value());
    }

    // --- 3. Out-of-range heap index is skipped, not read ---
    {
        auto p = makeProps();
        p.memoryTypeCount = 2;
        p.memoryTypes[0].propertyFlags = kPure;
        p.memoryTypes[0].heapIndex = 17;  // bogus
        p.memoryTypes[1].propertyFlags = kPure;
        p.memoryTypes[1].heapIndex = 0;
        const auto got = findLargestHeapPureDeviceLocal(p);
        CHECK(got.has_value());
        CHECK(got.value_or(99) == 1);
    }

    // --- 4. No DEVICE_LOCAL at all: nullopt ---
    {
        auto p = makeProps();
        p.memoryTypeCount = 1;
        p.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        p.memoryTypes[0].heapIndex = 0;
        CHECK(!findLargestHeapPureDeviceLocal(p).has_value());
    }

    // --- 5. Empty properties: nullopt (no crash on zero counts) ---
    {
        VkPhysicalDeviceMemoryProperties p{};
        CHECK(!findLargestHeapPureDeviceLocal(p).has_value());
    }

    if (failures == 0) {
        std::printf("=== ALL MEMORY TYPE TESTS PASSED ===\n");
        return 0;
    }
    std::printf("=== MEMORY TYPE TEST FAILURES (%d) ===\n", failures);
    return 1;
}
