// completion_token_test: CPU-only unit tests for backend-neutral completion
// tokens (no Vulkan device required).
//
// Covers: Ready consults true; VulkanTimeline with null timeline/device
// consults false (fail closed); Foreign delegates to the callback (empty
// consult is never complete); factory helpers set the right kind/fields.

#include <vulkan_vm/completion_token.hpp>

#include <cstdio>

using vvm::CompletionToken;
using vvm::isTokenComplete;

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

int main() {
    // --- 1. Ready is complete (any device, including none) ---
    CHECK(isTokenComplete(CompletionToken::ready()));
    CHECK(isTokenComplete(CompletionToken::ready(), VK_NULL_HANDLE));
    CHECK(isTokenComplete(CompletionToken{}));

    // --- 2. VulkanTimeline fails closed without a timeline or device ---
    CHECK(!isTokenComplete(CompletionToken::vulkanTimeline(VK_NULL_HANDLE, 1)));
    CHECK(!isTokenComplete(CompletionToken::vulkanTimeline(
        reinterpret_cast<VkSemaphore>(0x1234), 1, VK_NULL_HANDLE), VK_NULL_HANDLE));

    // --- 3. Foreign delegates to the consult callback ---
    CHECK(!isTokenComplete(CompletionToken::foreign({})));  // empty: never
    int calls = 0;
    auto countdown = CompletionToken::foreign([&] { return ++calls >= 3; });
    CHECK(!isTokenComplete(countdown));
    CHECK(!isTokenComplete(countdown));
    CHECK(isTokenComplete(countdown));
    CHECK(calls == 3);  // consulted on every pass, no caching

    // --- 4. Factories set kind + payload ---
    const auto r = CompletionToken::ready();
    CHECK(r.kind == CompletionToken::Kind::Ready);
    const auto t = CompletionToken::vulkanTimeline(
        reinterpret_cast<VkSemaphore>(0x5678), 42);
    CHECK(t.kind == CompletionToken::Kind::VulkanTimeline);
    CHECK(t.timeline == reinterpret_cast<VkSemaphore>(0x5678));
    CHECK(t.value == 42);
    CHECK(t.device == VK_NULL_HANDLE);
    const auto f = CompletionToken::foreign([] { return true; });
    CHECK(f.kind == CompletionToken::Kind::Foreign);
    CHECK(isTokenComplete(f));

    if (failures == 0) {
        std::printf("=== ALL COMPLETION TOKEN TESTS PASSED ===\n");
        return 0;
    }
    std::printf("=== COMPLETION TOKEN TEST FAILURES (%d) ===\n", failures);
    return 1;
}
