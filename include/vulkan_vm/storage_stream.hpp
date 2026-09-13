#pragma once

// VulkanVM SSD -> GPU streaming for MoE expert offload (Windows-first, cross-vendor).
//
// Two paths, one destination (the Chonk Buffer):
//   Path A (pure Vulkan, default): portable file read (overlapped/IoRing/BypassIO
//            where available, plain pread/ReadFile fallback) -> host-visible
//            staging Allocation -> copyBuffer -> device-local expert Allocation.
//            Read+write, works on AMD/Intel/NVIDIA today.
//   Path B (DirectStorage + D3D12, experimental): NVMe DMA -> D3D12 shared heap
//            -> NT handle -> vkImportMemoryWin32HandleKHR into the pool.
//            Read-only, Win11 + NVMe + DX12 only.
//
// Lifetime follows docs/LIFETIME_CONTRACT.md:
//   R1 pool owns every Allocation. R2 embedder owns VkDevice. R5 alias must not
//   outlive source. R8 offloaded must reload before use. Expert handles are
//   UniqueAllocation-backed; evict() deallocates back to the pool.

#include "vulkan_vm/core.hpp"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <list>

namespace vvm::storage {

// ============================================================================
// On-disk pack layout (v1, little-endian, no compression by default)
//
//   [PackHeader][ExpertEntry x N][blob_0][blob_1]...[blob_N-1]
// Each blob is 4K-aligned. Readers must use entry.fileOffset/entry.bytes,
// never assume packing order == expert id order.
// ============================================================================

constexpr uint32_t kPackMagic = 0x564D4558u; // 'VMEX'
constexpr uint32_t kPackVersion = 1;
constexpr uint64_t kPackAlign = 4096;

struct PackHeader {
    uint32_t magic = kPackMagic;
    uint32_t version = kPackVersion;
    uint32_t expertCount = 0;
    uint32_t reserved = 0;
};

struct ExpertEntry {
    uint32_t expertId = 0;
    uint32_t reserved = 0;
    uint64_t fileOffset = 0; // absolute offset in pack file
    uint64_t bytes = 0;      // uncompressed bytes
};

struct ExpertSpec {
    uint32_t expertId = 0;
    uint64_t fileOffset = 0;
    uint64_t bytes = 0;
};

// Which backend moves bytes SSD -> staging.
enum class StreamBackend {
    Auto,        // DirectStorage if available && requested, else Pure
    PureVulkan,  // portable, read+write, all vendors
    DirectStorage, // Win11 NVMe DMA, read-only, needs D3D12 interop
};

struct StorageStreamConfig {
    std::string packPath;              // MoE pack file
    StreamBackend backend = StreamBackend::Auto;
    uint32_t maxResidentExperts = 8;   // LRU cap in VRAM (keep cache/actives free)
    VkDeviceSize stagingBytes = 128ull * 1024 * 1024; // pure-path staging ring
    VkDeviceSize minBatchBytes = 1ull * 1024 * 1024;  // <1MB reads kill throughput
    bool useGDeflate = false;          // future: VK_EXT_memory_decompression
    bool allowDirectStorage = true;    // Auto may pick DirectStorage if present
};

// Runtime capability probe (no throw, no init side effects beyond query).
struct StreamCaps {
    bool pureAvailable = true;
    bool directStorageAvailable = false; // true only with SDK + Win11 + NVMe
    bool gdeflateAvailable = false;      // VK_NV/EXT_memory_decompression
    std::string notes;
};

VVM_API StreamCaps queryStreamCaps(VkPhysicalDevice physicalDevice);

// CPU-only LRU registry: which experts are resident, prefetch order, eviction.
// No Vulkan calls — fully unit-testable in CI without a GPU.
class VVM_API ExpertRegistry {
public:
    explicit ExpertRegistry(uint32_t maxResident);

    void registerPack(std::vector<ExpertSpec> specs);
    std::optional<ExpertSpec> specFor(uint32_t expertId) const;

    // Mark use (refresh LRU). Returns evicted id if cap exceeded.
    std::optional<uint32_t> touch(uint32_t expertId);
    bool isResident(uint32_t expertId) const;
    void evict(uint32_t expertId);

    // Order a prefetch batch: resident-first-skip, largest-first for throughput.
    std::vector<uint32_t> planPrefetch(const std::vector<uint32_t>& want) const;

    size_t residentCount() const;
    std::vector<uint32_t> residentIds() const; // MRU -> LRU

private:
    uint32_t maxResident_;
    std::unordered_map<uint32_t, ExpertSpec> specs_;
    std::list<uint32_t> lru_; // front = MRU
    std::unordered_map<uint32_t, std::list<uint32_t>::iterator> pos_;
};

// Pack file helpers (portable, CPU-only — used by tools + tests).
VVM_API uint64_t alignUp(uint64_t v, uint64_t align);
VVM_API bool writePackFile(const std::string& path,
                           const std::vector<std::pair<uint32_t, std::vector<uint8_t>>>& experts,
                           std::vector<ExpertSpec>* outSpecs = nullptr);
VVM_API std::optional<std::vector<ExpertSpec>> readPackTable(const std::string& path);
VVM_API std::optional<std::vector<uint8_t>> readPackBlob(const std::string& path,
                                                         const ExpertSpec& spec);

// GPU streamer: owns staging, fills device-local expert Allocations from pack.
// Thread-safe for prefetch/ensure from one loader thread + ensure from infer
// thread (internally mutex-guarded). Allocations live in caller pool (R1).
class VVM_API ExpertStreamer {
public:
    ExpertStreamer(UnifiedMemoryPool* pool, StorageStreamConfig config);
    ~ExpertStreamer();

    ExpertStreamer(const ExpertStreamer&) = delete;
    ExpertStreamer& operator=(const ExpertStreamer&) = delete;

    Result open();  // validates pack table, allocates staging ring
    void close();

    bool isOpen() const;
    StreamBackend activeBackend() const { return activeBackend_; }
    const std::vector<ExpertSpec>& specs() const { return specs_; }

    // Async hint: start loading these experts (no block on completion).
    Result prefetch(const std::vector<uint32_t>& expertIds);

    // Blocking: expert resident in VRAM on return. prefetched ? fast : load now.
    ResultT<Allocation*> ensureResident(uint32_t expertId);

    void evict(uint32_t expertId); // LRU spill, frees VRAM (R1: back to pool)
    void evictAll();

    struct Stats {
        uint64_t bytesStreamed = 0;
        uint64_t prefetchHits = 0;
        uint64_t prefetchMisses = 0;
        uint64_t evictions = 0;
    };
    Stats stats() const;

private:
    Result loadLocked(const ExpertSpec& spec); // caller holds mutex_

    UnifiedMemoryPool* pool_ = nullptr; // embedder-owned (R2)
    StorageStreamConfig config_;
    StreamBackend activeBackend_ = StreamBackend::PureVulkan;
    bool open_ = false;

    std::vector<ExpertSpec> specs_;
    ExpertRegistry registry_{8};

    // residentId -> device-local Allocation (pool-owned, R1)
    std::unordered_map<uint32_t, Allocation> resident_;
    // single staging buffer for pure path (host-visible, reused ring-style)
    std::optional<Allocation> staging_;
    VkDeviceSize stagingCursor_ = 0;

    mutable std::mutex mutex_;
    Stats stats_;
};

// DirectStorage interop helpers (Path B). Implemented in
// storage_stream_dstorage.cpp. Without VVM_HAS_DSTORAGE these report
// unsupported — same API, no link dependency on dstorage.lib.
VVM_API bool dstorageRuntimeAvailable();
VVM_API Result dstorageImportToPool(UnifiedMemoryPool* pool,
                                    void* d3d12SharedHeapNtHandle,
                                    VkDeviceSize bytes,
                                    Allocation* out);

} // namespace vvm::storage
