#pragma once

// Common constants for Vulkan-VM
// Centralizes magic numbers for timeouts, limits, and tuning knobs

#include <cstdint>

namespace vvm {
namespace constants {

// Timeouts (nanoseconds)
constexpr uint64_t kNsPerMs = 1'000'000;
constexpr uint64_t kNsPerSec = 1'000'000'000;

constexpr uint64_t kCopyTimeoutNs = 10 * kNsPerSec;          // 10 seconds
constexpr uint64_t kCopyBudgetNs = 60 * kNsPerSec;           // 60 seconds
constexpr uint64_t kDefaultConnectTimeoutMs = 5000;          // 5 seconds
constexpr uint64_t kDefaultRpcTimeoutMs = 30000;             // 30 seconds
constexpr uint64_t kDefaultMigrationTimeoutMs = 60000;       // 60 seconds
constexpr uint64_t kDefaultHeartbeatIntervalMs = 5000;       // 5 seconds
constexpr uint64_t kDefaultConnectionIdleTimeoutMs = 300000; // 5 minutes

// Buffer sizes
constexpr uint32_t kDefaultRdmaMtu = 4096;
constexpr uint64_t kDefaultStreamWindowBytes = 8 * 1024 * 1024;  // 8 MB

// Pipeline limits
constexpr uint32_t kMinPipelineBuffers = 2;
constexpr uint32_t kMaxPipelineBuffers = 4;
constexpr uint32_t kDefaultPipelineBuffers = 3;

// Message size limits (for untrusted peers)
constexpr uint64_t kDefaultMaxBodySize = 1024ull * 1024 * 1024;        // 1 GiB
constexpr uint64_t kDefaultMaxStreamSize = 16ull * 1024 * 1024 * 1024; // 16 GiB
constexpr uint32_t kDefaultMaxConcurrentStreams = 8;
constexpr uint64_t kDefaultMaxBytesPerSecond = 0;        // unlimited
constexpr uint32_t kDefaultMaxMessageRate = 1000;

// Alignment
constexpr VkDeviceSize kPageSize = 4096;

// Windowed pipeline
constexpr uint32_t kWindowedMinDepth = 2;
constexpr uint32_t kWindowedMaxDepth = 4;

// Copy context pool
constexpr uint32_t kMinCopyContexts = 4;
constexpr uint32_t kMaxCopyContexts = 16;

// Wire protocol version
constexpr uint32_t kProtocolVersionMajor = 1;
constexpr uint32_t kProtocolVersionMinor = 0;
constexpr uint32_t kProtocolVersionPatch = 0;
constexpr uint32_t kProtocolVersion = (kProtocolVersionMajor << 16) | (kProtocolVersionMinor << 8) | kProtocolVersionPatch;

} // namespace constants
} // namespace vvm