#pragma once

// Backend-neutral completion tokens for GPU-lifetime-safe reclamation.
//
// Problem: UnifiedMemoryPool::retire() speaks only Vulkan timelines, so
// HIP/L0/CUDA pools (and any foreign stream) cannot retire - they return
// false and fall back to synchronous waits. The token abstracts the
// *consult* step ("is this work done?") so one retire()/collect() gate
// serves every backend:
//
//   Kind::Ready          - already complete (synchronous paths, host work).
//                          collect() reclaims on the first pass.
//   Kind::VulkanTimeline - classic retire(): (device, timeline, value),
//                          consulted with vkGetSemaphoreCounterValue.
//   Kind::Foreign        - caller-supplied non-blocking consult, e.g. a
//                          hipEventQuery / cudaEventQuery / zeFence poll
//                          wrapped in a lambda. Invoked under the pool
//                          mutex: it MUST never block.
//
// Tokens are values (cheap to copy); the pool copies them into its retired
// queue. The referenced objects (timeline, foreign event) stay caller-owned
// exactly like the semaphore in retire().

#include "vulkan_vm/utils.hpp"

#include <cstdint>
#include <functional>

namespace vvm {

struct CompletionToken {
    enum class Kind : uint8_t { Ready, VulkanTimeline, Foreign };

    Kind kind = Kind::Ready;

    // VulkanTimeline payload. device may be null when the token is used
    // with the pool that owns the timeline - collect() then falls back
    // to the pool's device.
    VkDevice device = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;  // caller-owned, never destroyed
    uint64_t value = 0;

    // Foreign payload: non-blocking consult. Empty consult == never
    // complete (fail closed, same as a lost-device query).
    std::function<bool()> isComplete;

    static CompletionToken ready() { return CompletionToken{}; }

    static CompletionToken vulkanTimeline(VkSemaphore timeline, uint64_t value,
                                          VkDevice device = VK_NULL_HANDLE) {
        CompletionToken t;
        t.kind = Kind::VulkanTimeline;
        t.timeline = timeline;
        t.value = value;
        t.device = device;
        return t;
    }

    static CompletionToken foreign(std::function<bool()> consult) {
        CompletionToken t;
        t.kind = Kind::Foreign;
        t.isComplete = std::move(consult);
        return t;
    }
};

// Consult a token (non-blocking). defaultDevice fills in a null token
// device for the VulkanTimeline kind. Query failure (lost device, null
// timeline) returns false: leaking is safer than freeing against a dead
// device, and collect() keeps the item queued.
[[nodiscard]] VVM_API bool isTokenComplete(const CompletionToken& token,
                                                  VkDevice defaultDevice = VK_NULL_HANDLE);

}  // namespace vvm
