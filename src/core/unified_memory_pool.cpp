#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/buddy_allocator.hpp"
#include "vulkan_vm/utils.hpp"
#include "vulkan_vm/mem_backend.hpp"
#include "vulkan_vm/vulkan_mem_backend.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace vvm {

// OffloadManager deleter: declared in core.hpp (keeps OffloadManager
// incomplete there), defined here where the complete type is available.
void OffloadManagerDeleter::operator()(OffloadManager* p) const {
    delete p;
}

// ============================================================================
// Pimpl state: every private member + private method of UnifiedMemoryPool.
// The public façade (core.hpp) is exactly one unique_ptr: it never changes
// size, so std::optional<UnifiedMemoryPool> crossing the DLL boundary stays
// valid across library updates. Nothing in this struct is visible to
// consumers; it can grow freely. All method bodies below (renamed from
// UnifiedMemoryPool:: to UnifiedMemoryPoolImpl::) are unchanged.
// ============================================================================
struct UnifiedMemoryPoolImpl {
    // Back-pointer for Impl-internal users of the outer object
    // (OffloadManager takes UnifiedMemoryPool*). Repointed on move.
    UnifiedMemoryPool* outer_ = nullptr;

    // Vendor seam (mem_backend.hpp): the memory plane behind the pool's
    // allocation policy. Created in initialize(), destroyed with the Impl.
    std::unique_ptr<IDeviceMemoryBackend> backend_;

    // Thread safety: all methods are guarded by this mutex
    mutable std::mutex mutex_;

    DeviceConfig deviceConfig_;
    PoolConfig config_;
    VkDevice device_ = VK_NULL_HANDLE;
    std::vector<BlockInfo> blocks_;
    // Dedicated allocations (exportable/imported) tracked separately
    std::vector<Allocation> dedicatedAllocations_;
    // Budget held via reserve(): counts as committed in wouldExceedBudget
    // until unreserve()d. Guarded by mutex_ like everything else.
    VkDeviceSize reservedBytes_ = 0;
    uint32_t deviceLocalMemoryType_ = UINT32_MAX;
    uint32_t hostVisibleMemoryType_ = UINT32_MAX;
    uint32_t deviceLocalHeapIndex_ = UINT32_MAX;
    bool memoryBudgetAvailable_ = false;
    VkCommandPool transferCmdPool_ = VK_NULL_HANDLE;
    bool debugUtilsEnabled_ = false;
    PFN_vkSetDebugUtilsObjectNameEXT fnSetDebugName_ = nullptr;

    // Retirement queue (GPU-lifetime-safe reclamation): allocations handed to
    // retire() wait here until collect() observes their timeline value. The
    // timeline stays caller-owned (pool never destroys it) except
    // retireTimeline_, the pool-owned ticket timeline for internal async use.
    struct RetirementItem {
        Allocation allocation;
        VkCommandBuffer cmd = VK_NULL_HANDLE;  // freed at reclaim (see below)
        VkCommandPool cmdPool = VK_NULL_HANDLE;  // destroyed at reclaim (or NULL)
        VkSemaphore timeline = VK_NULL_HANDLE; // caller-owned (or retireTimeline_)
        uint64_t value = 0;
    };
    std::vector<RetirementItem> retired_;
    VkSemaphore retireTimeline_ = VK_NULL_HANDLE;
    uint64_t retireNextValue_ = 0;

    // Offload manager for host swap
    std::unique_ptr<OffloadManager, OffloadManagerDeleter> offloadManager_;

    // VRAM high-water warning fired? (see VRAM_OVERFLOW_FINDINGS.md)
    bool warnedHighWater_ = false;
    // Generation counter for handle validation (prevents stale handle use).
    // Validity is tracked via the LIVE SET, not the counter (comparing
    // against the counter only matches the most-recent allocation, so
    // out-of-order frees were wrongly rejected and leaked memory).
    uint64_t generationCounter_ = 0;
    std::unordered_set<uint64_t> liveGenerations_;

    // Adaptive sizing history: ring of recent buddy-path request sizes.
    static constexpr uint32_t kSizeHistory = 8;
    VkDeviceSize recentSizes_[kSizeHistory] = {};
    uint32_t recentIdx_ = 0;
    uint32_t recentCount_ = 0;

    ~UnifiedMemoryPoolImpl();

    bool initialize(const DeviceConfig& device, const PoolConfig& config);
    bool isVulkanBackend() const;
    bool validateConfig() const;
    bool selectMemoryTypes();
    bool initTransferPool();
    bool bootstrapFirstBlock();
    bool validateDeviceCapabilities() const;

    uint64_t nextGeneration() {
        uint64_t g = ++generationCounter_;
        liveGenerations_.insert(g);
        return g;
    }
    uint64_t getCurrentGeneration() const { return generationCounter_; }
    bool isValidGeneration(uint64_t generation) const {
        return liveGenerations_.count(generation) != 0;
    }
    void retireGeneration(uint64_t generation) {
        liveGenerations_.erase(generation);
    }

    std::optional<uint32_t> findMemoryType(VkMemoryPropertyFlags required,
                                           VkMemoryPropertyFlags preferred);
    std::optional<VkDeviceMemory> allocateBlock(VkDeviceSize size,
                                                uint32_t memoryTypeIndex,
                                                bool isChunk = false,
                                                uint32_t chunkTier = 0);
    std::optional<Allocation> subAllocate(VkDeviceSize size,
                                          VkDeviceSize alignment,
                                          uint32_t blockIndex,
                                          VkBufferUsageFlags usage);
    void subDeallocate(Allocation&& alloc);
    VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment);
    VkMemoryPropertyFlags usageToFlags(MemoryUsage usage) const;
    bool wouldExceedBudget(VkDeviceSize additionalBytes) const;
    VkDeviceSize heapSizeBytes() const;
    void setDebugName(VkObjectType objectType, uint64_t objectHandle,
                      const char* name) const;
    VkDeviceSize adaptiveBlockSizeFor(VkDeviceSize requestSize, VkDeviceSize staticPick);

    // Former public-API bodies, now one hop behind the façade forwarders.
    std::optional<Allocation> allocate(VkDeviceSize size,
                                       VkBufferUsageFlags usage,
                                       VkMemoryPropertyFlags flags = 0);
    std::optional<Allocation> allocate(const AllocDesc& desc);
    std::optional<Allocation> allocateDedicatedExportable(
        VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags flags = 0);
    std::optional<Allocation> allocateDedicated(
        VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags flags = 0);
    std::optional<Allocation> allocateDedicatedBackend(
        VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags flags = 0);
    std::optional<Allocation> allocateTensor(VkDeviceSize size,
                                             VkBufferUsageFlags usage);
    void deallocate(Allocation&& alloc);
    void deallocate(UniqueAllocation&& alloc);
    // Lock-free body of deallocate(); caller must hold mutex_.
    void deallocateLocked(Allocation&& alloc);
    bool retire(Allocation&& alloc, VkSemaphore timeline, uint64_t value,
                VkCommandBuffer cmd, VkCommandPool cmdPool);
    uint32_t collect();
    std::pair<VkSemaphore, uint64_t> retireTicket();
    bool ensureRetireTimeline();
    bool reserve(VkDeviceSize bytes);
    void unreserve(VkDeviceSize bytes);
    VkDeviceSize reservedBytes() const;
    [[nodiscard]] std::optional<ExternalMemoryInfo> exportMemory(const Allocation& alloc,
                                                        ExternalHandleType type);
    [[nodiscard]] std::optional<Allocation> importMemory(ExternalMemoryInfo&& info,
                                                VkBufferUsageFlags usage);
    std::optional<MigrationOperation> offloadToHost(Allocation& alloc);
    std::optional<MigrationOperation> reloadToDevice(Allocation& alloc);
    void waitMigration(const MigrationOperation& op);
    std::optional<Allocation> importMemoryHostPointer(void* hostPtr,
                                                      VkDeviceSize size,
                                                      VkBufferUsageFlags usage);
    bool copyBuffer(const Allocation& src, const Allocation& dst,
                    VkDeviceSize srcOffset, VkDeviceSize dstOffset,
                    VkDeviceSize size, VkFence fence = VK_NULL_HANDLE);
    PoolStats getStats() const;
    DeviceMemoryInfo getDeviceMemoryInfo() const;
    void defragment();
    void trim();
};

// ============================================================================
// UnifiedMemoryPool Implementation
// ============================================================================

// Out-of-line default ctor: see core.hpp note. Destroys the unique_ptr
// member with the complete backend type.
UnifiedMemoryPool::UnifiedMemoryPool() = default;

MemoryTopologyType detectMemoryTopology(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);

    bool hasUnifiedType = false;   // DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
    bool hasDevLocal = false;
    uint32_t devLocalHeaps = 0;
    for (uint32_t h = 0; h < props.memoryHeapCount; ++h) {
        const auto& heap = props.memoryHeaps[h];
        const bool heapDevLocal = (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        if (heapDevLocal) devLocalHeaps++;
        for (uint32_t t = 0; t < props.memoryTypeCount; ++t) {
            const auto& mt = props.memoryTypes[t];
            if (mt.heapIndex != h) continue;
            if (heapDevLocal) hasDevLocal = true;
            const VkMemoryPropertyFlags f = mt.propertyFlags;
            if (heapDevLocal &&
                (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                hasUnifiedType = true;
            }
        }
    }

    if (!hasDevLocal) return MemoryTopologyType::Discrete;
    if (devLocalHeaps <= 1 && hasUnifiedType) return MemoryTopologyType::Unified;
    if (hasUnifiedType) return MemoryTopologyType::Hybrid;
    return MemoryTopologyType::Discrete;
}

VkDeviceSize totalDeviceVRAM(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &props);
    VkDeviceSize total = 0;
    for (uint32_t h = 0; h < props.memoryHeapCount; ++h) {
        if (props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total += props.memoryHeaps[h].size;
        }
    }
    return total;
}

// Largest-heap pure DEVICE_LOCAL type. findMemoryTypeIndex returns the
// FIRST match, which is wrong on NVIDIA: early pure types can sit on small
// heaps (a 1080 Ti pool landed on a 256 MB heap and OOMed its first block).
// Used by the preferPureDeviceLocal swap; the initial MemoryTypeSelector
// pass is score-based and unaffected.
static std::optional<uint32_t> findLargestHeapPureDeviceLocal(
    const VkPhysicalDeviceMemoryProperties& memProps) {
    std::optional<uint32_t> best;
    uint64_t bestHeap = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f = memProps.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) continue;
        if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) continue;
        const uint32_t h = memProps.memoryTypes[i].heapIndex;
        if (h >= memProps.memoryHeapCount) continue;
        const uint64_t hs = memProps.memoryHeaps[h].size;
        if (!best.has_value() || hs > bestHeap) {
            best = i;
            bestHeap = hs;
        }
    }
    return best;
}

// Driver single-allocation cap (Vulkan 1.1 maxMemoryAllocationSize).
// Windows AMD/Intel drivers refuse single vkAllocateMemory at/above ~4 GiB
// (spec explicitly permits refusing >= 4 GiB), so pool growth must never
// attempt one backend block above this. No class state (core.hpp is
// ABI-frozen): per-physical-device cache behind its own lock. Growth path
// only — never on a hot path. Non-pow2 caps are floored (buddy blocks are
// always pow2).
VkDeviceSize driverMaxSingleAlloc(VkPhysicalDevice phys) {
    static std::mutex mtx;
    static std::unordered_map<VkPhysicalDevice, VkDeviceSize> cache;
    if (phys == VK_NULL_HANDLE) return UINT64_MAX;
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(phys);
        if (it != cache.end()) return it->second;
    }
    VkPhysicalDeviceVulkan11Properties props11{};
    props11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &props11;
    vkGetPhysicalDeviceProperties2(phys, &props2);
    VkDeviceSize cap = props11.maxMemoryAllocationSize;
    if (cap == 0) cap = UINT64_MAX;  // pre-1.1 driver without the struct
    while (cap != UINT64_MAX && (cap & (cap - 1)) != 0) {
        cap &= cap - 1;  // floor to pow2
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        cache[phys] = cap;
    }
    return cap;
}

PoolConfig PoolConfig::forDevice(VkPhysicalDevice physicalDevice) {
    PoolConfig cfg;
    const VkDeviceSize vram = totalDeviceVRAM(physicalDevice);
    const bool isHighVRAM = vram >= 24ull * 1024 * 1024 * 1024;
    switch (detectMemoryTopology(physicalDevice)) {
        case MemoryTopologyType::Unified:
            // APU / Strix Halo style: one shared heap. Bigger blocks, fewer of
            // them; host shadow is largely unnecessary since VRAM is
            // host-visible. Still cap the fraction so we don't starve the OS.
            cfg.blockSize = isHighVRAM ? 2048ull * 1024 * 1024 : 1024ull * 1024 * 1024;
            cfg.maxBlocks = isHighVRAM ? 16 : 8;
            cfg.enableHostVisible = true;
            // maxHeapFraction caps the pool at a fraction of the reported
            // DEVICE_LOCAL heap. On Strix Halo (UMA, 129GB physical) the
            // reported heap under-counts the true unified RAM, so the default
            // 0.8 can leave a false ~84GB wall. Allow override via
            // CHONK_MAX_HEAP_FRACTION (e.g. 0.92) to use more physical RAM.
            {
                const char* hf = getenv("CHONK_MAX_HEAP_FRACTION");
                cfg.maxHeapFraction = hf ? (float)atof(hf) : (isHighVRAM ? 0.8f : 0.7f);
            }
            cfg.hostShadowMultiplier = 0.0f;  // VRAM is already host-visible
            cfg.maxHostShadowBytes = 0;
            cfg.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            cfg.blockSizes = {
                256ull * 1024 * 1024,
                512ull * 1024 * 1024,
                1024ull * 1024 * 1024,
                2048ull * 1024 * 1024,
                4096ull * 1024 * 1024,
                8192ull * 1024 * 1024,
                16384ull * 1024 * 1024,
                32768ull * 1024 * 1024,
            };
            break;
        case MemoryTopologyType::Hybrid:
            cfg.blockSize = isHighVRAM ? 2048ull * 1024 * 1024 : 512ull * 1024 * 1024;
            cfg.maxBlocks = isHighVRAM ? 24 : 12;
            cfg.maxHeapFraction = 0.75f;
            cfg.hostShadowMultiplier = 2.0f;  // Smaller shadow for hybrid
            cfg.blockSizes = {
                256ull * 1024 * 1024,
                512ull * 1024 * 1024,
                1024ull * 1024 * 1024,
                2048ull * 1024 * 1024,
                4096ull * 1024 * 1024,
                8192ull * 1024 * 1024,
                16384ull * 1024 * 1024,
            };
            break;
        case MemoryTopologyType::Discrete:
        default:
            // High-VRAM discrete cards (RTX 4090 24GB, RTX 6000 Ada 48GB):
            // 2GB blocks up to 64 blocks covers 128GB; heap fraction 0.8 leaves
            // headroom for the driver/other consumers.
            cfg.blockSize = isHighVRAM ? 2048ull * 1024 * 1024 : 512ull * 1024 * 1024;
            cfg.maxBlocks = isHighVRAM ? 64 : 16;
            cfg.maxHeapFraction = isHighVRAM ? 0.8f : 0.75f;
            cfg.hostShadowMultiplier = isHighVRAM ? 2.0f : 4.0f;
            cfg.blockSizes = {
                512ull * 1024 * 1024,
                1024ull * 1024 * 1024,
                2048ull * 1024 * 1024,
                4096ull * 1024 * 1024,
                8192ull * 1024 * 1024,
                16384ull * 1024 * 1024,
                32768ull * 1024 * 1024,
            };
            break;
    }
    return cfg;
}

PoolConfig PoolConfig::forAPU(VkDeviceSize totalSystemRAM) {
    PoolConfig cfg;
    cfg.enableHostVisible = true;
    cfg.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    // APUs share system RAM with the GPU. Bigger blocks, modest count, and a
    // conservative heap fraction so we never starve the OS.
    if (totalSystemRAM >= 64ull * 1024 * 1024 * 1024) {
        cfg.blockSize = 1024ull * 1024 * 1024;  // 1 GB
        cfg.maxBlocks = 16;
        cfg.maxHeapFraction = 0.6f;
    } else {
        cfg.blockSize = 512ull * 1024 * 1024;
        cfg.maxBlocks = 8;
        cfg.maxHeapFraction = 0.5f;
    }
    // APU VRAM is host-visible; shadow buffer is redundant
    cfg.hostShadowMultiplier = 0.0f;
    cfg.maxHostShadowBytes = 0;
    // Multi-block sizes for flexible routing on APUs (scaled for future hardware)
    cfg.blockSizes = {
        256ull * 1024 * 1024,   // 256 MB
        512ull * 1024 * 1024,   // 512 MB
        1024ull * 1024 * 1024,  // 1 GB
        2048ull * 1024 * 1024,  // 2 GB
        4096ull * 1024 * 1024,  // 4 GB
        8192ull * 1024 * 1024,  // 8 GB
        16384ull * 1024 * 1024, // 16 GB
        32768ull * 1024 * 1024, // 32 GB
    };
    return cfg;
}

PoolConfig PoolConfig::forHighVRAM(VkPhysicalDevice physicalDevice) {
    PoolConfig cfg;
    const VkDeviceSize vram = totalDeviceVRAM(physicalDevice);
    cfg.blockSize = 2048ull * 1024 * 1024;  // 2 GB
    cfg.maxBlocks = 64;
    // Cap at 85% so we never commit the whole heap; heap sizes are often the
    // full VRAM minus a small reserved carve-out.
    cfg.maxHeapFraction = 0.85f;
    cfg.enableHostVisible = false;
    if (detectMemoryTopology(physicalDevice) == MemoryTopologyType::Unified) {
        cfg.enableHostVisible = true;
        cfg.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    cfg.hostShadowMultiplier = 2.0f;
    cfg.maxHostShadowBytes = 4ull * 1024 * 1024 * 1024;  // Cap at 4GB
    cfg.blockSizes = {
        512ull * 1024 * 1024,
        1024ull * 1024 * 1024,
        2048ull * 1024 * 1024,
        4096ull * 1024 * 1024,
        8192ull * 1024 * 1024,
        16384ull * 1024 * 1024,
        32768ull * 1024 * 1024,
    };
    (void)vram;  // kept for future heuristics (e.g. block count scaling)
    return cfg;
}

std::optional<UnifiedMemoryPool> UnifiedMemoryPool::create(
    const DeviceConfig& device, const PoolConfig& config) {
    UnifiedMemoryPool pool;
    pool.impl_ = std::make_unique<UnifiedMemoryPoolImpl>();
    pool.impl_->outer_ = &pool;
    if (pool.impl_->initialize(device, config)) {
        return pool;
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Pimpl forwarders: the entire public API is one pointer hop into Impl.
// Nothing here touches state, so this TU is the only one that must rebuild
// when private members change.
// ---------------------------------------------------------------------------
std::optional<Allocation> UnifiedMemoryPool::allocate(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) { return impl_->allocate(size, usage, flags); }
std::optional<Allocation> UnifiedMemoryPool::allocate(const AllocDesc& desc) { return impl_->allocate(desc); }
std::optional<Allocation> UnifiedMemoryPool::allocateDedicatedExportable(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) { return impl_->allocateDedicatedExportable(size, usage, flags); }
std::optional<Allocation> UnifiedMemoryPool::allocateDedicated(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) { return impl_->allocateDedicated(size, usage, flags); }
std::optional<Allocation> UnifiedMemoryPool::allocateDedicatedBackend(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) { return impl_->allocateDedicatedBackend(size, usage, flags); }
std::optional<Allocation> UnifiedMemoryPool::allocateTensor(VkDeviceSize size, VkBufferUsageFlags usage) { return impl_->allocateTensor(size, usage); }
void UnifiedMemoryPool::deallocate(Allocation&& alloc) { impl_->deallocate(std::move(alloc)); }
void UnifiedMemoryPool::deallocate(UniqueAllocation&& alloc) { impl_->deallocate(std::move(alloc)); }
bool UnifiedMemoryPool::reserve(VkDeviceSize bytes) { return impl_->reserve(bytes); }
bool UnifiedMemoryPool::retire(Allocation&& alloc, VkSemaphore timeline, uint64_t value,
                               VkCommandBuffer cmd, VkCommandPool cmdPool) {
    return impl_->retire(std::move(alloc), timeline, value, cmd, cmdPool);
}
uint32_t UnifiedMemoryPool::collect() { return impl_->collect(); }
std::pair<VkSemaphore, uint64_t> UnifiedMemoryPool::retireTicket() { return impl_->retireTicket(); }
void UnifiedMemoryPool::unreserve(VkDeviceSize bytes) { impl_->unreserve(bytes); }
VkDeviceSize UnifiedMemoryPool::reservedBytes() const { return impl_->reservedBytes(); }
std::optional<ExternalMemoryInfo> UnifiedMemoryPool::exportMemory(const Allocation& alloc, ExternalHandleType type) { return impl_->exportMemory(alloc, type); }
std::optional<Allocation> UnifiedMemoryPool::importMemory(ExternalMemoryInfo&& info, VkBufferUsageFlags usage) { return impl_->importMemory(std::move(info), usage); }
std::optional<MigrationOperation> UnifiedMemoryPool::offloadToHost(Allocation& alloc) { return impl_->offloadToHost(alloc); }
std::optional<MigrationOperation> UnifiedMemoryPool::reloadToDevice(Allocation& alloc) { return impl_->reloadToDevice(alloc); }
void UnifiedMemoryPool::waitMigration(const MigrationOperation& op) { impl_->waitMigration(op); }
bool UnifiedMemoryPool::copyBuffer(const Allocation& src, const Allocation& dst, VkDeviceSize srcOffset, VkDeviceSize dstOffset, VkDeviceSize size, VkFence fence) { return impl_->copyBuffer(src, dst, srcOffset, dstOffset, size, fence); }
PoolStats UnifiedMemoryPool::getStats() const { return impl_->getStats(); }
DeviceMemoryInfo UnifiedMemoryPool::getDeviceMemoryInfo() const { return impl_->getDeviceMemoryInfo(); }
std::optional<Allocation> UnifiedMemoryPool::importMemoryHostPointer(void* hostPtr, VkDeviceSize size, VkBufferUsageFlags usage) { return impl_->importMemoryHostPointer(hostPtr, size, usage); }
const PoolConfig& UnifiedMemoryPool::getConfig() const { return impl_->config_; }
const DeviceConfig& UnifiedMemoryPool::getDeviceConfig() const { return impl_->deviceConfig_; }
VkDevice UnifiedMemoryPool::getDevice() const { return impl_->device_; }
VkPhysicalDevice UnifiedMemoryPool::getPhysicalDevice() const { return impl_->deviceConfig_.physicalDevice; }
void UnifiedMemoryPool::defragment() { impl_->defragment(); }
void UnifiedMemoryPool::trim() { impl_->trim(); }

UnifiedMemoryPool::UnifiedMemoryPool(UnifiedMemoryPool&& other) noexcept
    : impl_(std::move(other.impl_)) {
    // Pointer transfer: no partial state exists to audit (the whole pool
    // moves as one). Repoint the back-pointer so Impl-internal users of the
    // outer object (OffloadManager) stay valid.
    if (impl_) impl_->outer_ = this;
}

UnifiedMemoryPool& UnifiedMemoryPool::operator=(UnifiedMemoryPool&& other) noexcept {
    if (this != &other) {
        // unique_ptr assignment destroys our current Impl (its destructor
        // performs the device-resource cleanup the old hand-rolled move did).
        impl_ = std::move(other.impl_);
        if (impl_) impl_->outer_ = this;
    }
    return *this;
}

UnifiedMemoryPool::~UnifiedMemoryPool() = default;

UnifiedMemoryPoolImpl::~UnifiedMemoryPoolImpl() {
    // Lock so a concurrent allocate/deallocate on another thread cannot race
    // teardown (blocks_/dedicatedAllocations_ are mutated under mutex_).
    std::lock_guard<std::mutex> lock(mutex_);
    if (device_) {
        // Normal at process exit (static containers outlive pools); a real
        // mid-run ownership bug is only interesting on demand, so the warn
        // is opt-in rather than noise in every test/server shutdown log.
        if (getenv("VVM_WARN_LIVE_POOL")) {
            VVM_LOG_WARN("~UnifiedMemoryPool destroying LIVE pool (device={}, blocks={}, "
                         "dedicated={}) - if this is not process exit, a move/ownership bug "
                         "is freeing a stored pool",
                         (void*)device_, blocks_.size(), dedicatedAllocations_.size());
        }
        // Clean up dedicated allocations (exportable/imported)
        for (auto& alloc : dedicatedAllocations_) {
            if (alloc.buffer) {
                backend_->destroy_buffer(reinterpret_cast<uint64_t>(alloc.buffer));
            }
            if (alloc.memory) {
                if (alloc.hostPtr) {
                    backend_->unmap(reinterpret_cast<uint64_t>(alloc.memory));
                }
                backend_->free(reinterpret_cast<uint64_t>(alloc.memory));
            }
        }
        dedicatedAllocations_.clear();

        for (auto& block : blocks_) {
            if (block.memory) {
                if (block.hostPtr) {
                    backend_->unmap(reinterpret_cast<uint64_t>(block.memory));
                }
                backend_->free(reinterpret_cast<uint64_t>(block.memory));
            }
        }
        // Retired-but-uncollected items: destroy their buffers; free memory
        // only for dedicated items (sub-allocated ones share their block's
        // memory, freed by the blocks loop below - freeing here would
        // double-free). The GPU work they waited on dies with the device.
        for (auto& item : retired_) {
            if (item.allocation.buffer) {
                backend_->destroy_buffer(reinterpret_cast<uint64_t>(item.allocation.buffer));
            }
            if (item.allocation.blockIndex == UINT32_MAX && item.allocation.memory) {
                if (item.allocation.hostPtr) {
                    backend_->unmap(reinterpret_cast<uint64_t>(item.allocation.memory));
                }
                backend_->free(reinterpret_cast<uint64_t>(item.allocation.memory));
            }
        }
        retired_.clear();
        if (retireTimeline_) {
            vkDestroySemaphore(device_, retireTimeline_, nullptr);
            retireTimeline_ = VK_NULL_HANDLE;
        }
        if (transferCmdPool_) {
            vkDestroyCommandPool(device_, transferCmdPool_, nullptr);
        }
    }
}

bool UnifiedMemoryPoolImpl::initialize(const DeviceConfig& device, const PoolConfig& config) {
    deviceConfig_ = device;
    config_ = config;
    device_ = device.device;

    // Vendor seam: dispatch on what the DeviceConfig carries (see
    // isVulkanBackend for the predicate; Level0/Cuda require explicit kind).
    const bool isVulkan = isVulkanBackend();
    const MemBackendKind explicitKind =
        static_cast<MemBackendKind>(deviceConfig_.memBackendKind);
    const MemBackendKind kind = isVulkan ? MemBackendKind::Vulkan
        : (explicitKind == MemBackendKind::Level0 ? MemBackendKind::Level0
        : (explicitKind == MemBackendKind::Cuda  ? MemBackendKind::Cuda
                                                 : MemBackendKind::Hip));
    backend_ = create_memory_backend(kind, deviceConfig_);
    if (!backend_) {
        VVM_LOG_ERROR("failed to create the {} memory backend",
                      isVulkan ? "Vulkan"
                               : (kind == MemBackendKind::Cuda ? "CUDA"
                                                              : (kind == MemBackendKind::Level0 ? "L0" : "HIP")));
        return false;
    }

    // Verify the device was created with the features/extensions the pool needs.
    if (isVulkan && !validateDeviceCapabilities()) {
        return false;
    }
    if (!validateConfig()) {
        return false;
    }
    if (!selectMemoryTypes()) {
        return false;
    }
    if (isVulkan && !initTransferPool()) {
        return false;
    }
    if (!bootstrapFirstBlock()) {
        return false;
    }

    // Create OffloadManager if offload is enabled (Vulkan only in v1; the
    // discovery stage above already warned for non-Vulkan configs).
    if (isVulkan && config_.enableHostVisible) {
        OffloadConfig offloadConfig;
        VkDeviceSize shadow = static_cast<VkDeviceSize>(config_.blockSize * config_.hostShadowMultiplier);
        if (config_.maxHostShadowBytes > 0) {
            shadow = std::min(shadow, config_.maxHostShadowBytes);
        }
        offloadConfig.hostShadowSize = shadow;
        offloadConfig.transferQueue = deviceConfig_.transferQueue;
        offloadConfig.transferQueueFamily = deviceConfig_.transferQueueFamily != UINT32_MAX
            ? deviceConfig_.transferQueueFamily
            : deviceConfig_.graphicsQueueFamily;

        offloadManager_ = std::unique_ptr<OffloadManager, OffloadManagerDeleter>(
            new OffloadManager(outer_, offloadConfig));
        VVM_LOG_INFO("OffloadManager created with host shadow size: {} MB",
                     offloadConfig.hostShadowSize / (1024*1024));
    }

    VVM_LOG_INFO("UnifiedMemoryPool initialized successfully (blockSize={} MB, alignment={} KB)",
                 config_.blockSize / (1024*1024), config_.minAlignment / 1024);
    // One-line env summary (audit QoL): every VVM_* knob that changes pool
    // behavior, printed once per pool so bug reports carry their config.
    {
        const char* keys[] = {"VVM_WARN_LIVE_POOL", "VVM_SKIP_CMDPOOL",
                              "VVM_SKIP_INITBLOCK", "VVM_DEVICE_INDEX",
                              "VVM_STAGED_CHUNK_MB", "VVM_P2P_POLICY",
                              "VVM_ALLOW_CROSSVENDOR_ZC"};
        std::string summary;
        for (const char* k : keys) {
            if (const char* v = std::getenv(k)) {
                if (!summary.empty()) summary += " ";
                summary += k;
                summary += "=";
                summary += v;
            }
        }
        VVM_LOG_INFO("VVM env: {}", summary.empty() ? "(defaults)" : summary.c_str());
    }
    if (isVulkan && deviceConfig_.physicalDevice != VK_NULL_HANDLE) {
        const VkDeviceSize cap = driverMaxSingleAlloc(deviceConfig_.physicalDevice);
        if (cap == UINT64_MAX) {
            VVM_LOG_INFO("driver maxMemoryAllocationSize: unknown (pool blocks unclamped)");
        } else {
            VVM_LOG_INFO("driver maxMemoryAllocationSize: {} MB (pool blocks never exceed it)",
                         cap / (1024*1024));
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// initialize() stages
// ---------------------------------------------------------------------------

bool UnifiedMemoryPoolImpl::isVulkanBackend() const {
    const MemBackendKind explicitKind =
        static_cast<MemBackendKind>(deviceConfig_.memBackendKind);
    return (explicitKind == MemBackendKind::Vulkan) ||
           (explicitKind == MemBackendKind::Auto &&
            deviceConfig_.physicalDevice != VK_NULL_HANDLE);
}

bool UnifiedMemoryPoolImpl::validateConfig() const {
    // Block geometry boundary: the buddy core divides by minAlignment and
    // assumes power-of-two blocks, so enforce it once here instead of
    // corrupting silently on a misconfigured pool.
    if (config_.minAlignment == 0 || !isPowerOfTwoU64(config_.blockSize) ||
        config_.blockSize < config_.minAlignment) {
        VVM_LOG_ERROR("Invalid blockSize={} / minAlignment={} "
                      "(block must be a power of two >= minAlignment, minAlignment > 0)",
                      config_.blockSize, config_.minAlignment);
        return false;
    }
    // Chonk Chunks validation: both knobs must be set together, sizes must be
    // powers of two, and a chunk must hold at least one min-aligned allocation.
    auto pow2 = [](VkDeviceSize v) { return v != 0 && (v & (v - 1)) == 0; };
    if (config_.smallAllocThreshold > 0 || config_.chunkBlockSize > 0) {
        if (config_.smallAllocThreshold == 0 || config_.chunkBlockSize == 0 ||
            !pow2(config_.chunkBlockSize) ||
            config_.chunkBlockSize < config_.minAlignment ||
            config_.chunkBlockSize < config_.smallAllocThreshold) {
            VVM_LOG_ERROR("Invalid Chonk Chunks config: smallAllocThreshold={} chunkBlockSize={} "
                          "(both required, chunk must be a power of two >= threshold and >= minAlignment)",
                          config_.smallAllocThreshold, config_.chunkBlockSize);
            return false;
        }
    }
    // Multi-tier ladder: thresholds strictly ascending, each block a power of
    // two >= its threshold and >= minAlignment.
    {
        VkDeviceSize prevThreshold = 0;
        for (size_t i = 0; i < config_.chunkTiers.size(); ++i) {
            const auto& [th, bs] = config_.chunkTiers[i];
            if (th == 0 || bs == 0 || th <= prevThreshold || !pow2(bs) ||
                bs < config_.minAlignment || bs < th) {
                VVM_LOG_ERROR("Invalid chunkTiers[{}]: (threshold={}, block={}) "
                              "(thresholds strictly ascending; blocks power-of-two >= threshold and >= minAlignment)",
                              i, th, bs);
                return false;
            }
            prevThreshold = th;
        }
    }
    if (config_.allocationAlignment != 0 &&
        (!pow2(config_.allocationAlignment) || config_.allocationAlignment < config_.minAlignment)) {
        VVM_LOG_ERROR("Invalid allocationAlignment {} (must be power of two >= minAlignment {})",
                      config_.allocationAlignment, config_.minAlignment);
        return false;
    }
    return true;
}

bool UnifiedMemoryPoolImpl::selectMemoryTypes() {
    // Discovery dispatch: the Vulkan path is byte-identical to the historical
    // behavior; the else-branch drives discovery through the backend seam
    // (HIP today).
    if (isVulkanBackend()) {
    // Use MemoryTypeSelector for optimal memory type selection.
    // NOTE: type selection is decoupled from capacity here (minHeapBudget=0).
    // Demanding a full blockSize of *free budget* at type-selection time would
    // refuse to create pools on heaps that are merely nearly full (e.g. a
    // context-init pool created after layer-split weight placement committed
    // ~97% of the heap) - even though the best-fit first-block ladder below
    // can bootstrap from a small block. Capacity is enforced per-allocation
    // by wouldExceedBudget + vkAllocateMemory, where it belongs.
    MemoryTypeSelector selector(deviceConfig_.physicalDevice);
    auto devLocalResult = selector.select(
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        config_.preferredFlags,
        0);

    if (devLocalResult.memoryTypeIndex == UINT32_MAX) {
        VVM_LOG_ERROR("Failed to find DEVICE_LOCAL memory type with sufficient budget");
        return false;
    }
    deviceLocalMemoryType_ = devLocalResult.memoryTypeIndex;

    // Experiment knob: some drivers place/map ReBAR-mapped VRAM (types with
    // DEVICE_LOCAL|HOST_VISIBLE) differently from pure DEVICE_LOCAL. When
    // requested, swap to the pure DEVICE_LOCAL type on the LARGEST heap
    // (first-match is wrong on NVIDIA: early pure types can sit on small
    // heaps - seen 256 MB OOM on a 1080 Ti).
    if (config_.preferPureDeviceLocal) {
        auto memPropsEarly = getDeviceMemoryInfo().memProps;
        VkMemoryPropertyFlags selFlags{};
        getMemoryTypeProperties(deviceLocalMemoryType_, selFlags, memPropsEarly);
        if (selFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            auto pure = findLargestHeapPureDeviceLocal(memPropsEarly);
            if (pure.has_value()) {
                deviceLocalMemoryType_ = *pure;
                const uint32_t h = memPropsEarly.memoryTypes[*pure].heapIndex;
                const uint64_t hs = h < memPropsEarly.memoryHeapCount
                    ? memPropsEarly.memoryHeaps[h].size : 0;
                VVM_LOG_INFO("preferPureDeviceLocal: switched to memory type {} ({} MB heap)",
                             *pure, hs / (1024*1024));
            }
        }
    }

    VVM_LOG_INFO("Selected DEVICE_LOCAL memory type {} (heap budget: {} MB, utilization: {} percent)",
                 deviceLocalMemoryType_,
                 devLocalResult.heapBudget / (1024*1024),
                 devLocalResult.heapUtilization * 100.0f);

    // Remember which heap the device-local type lives on (for budget checks).
    {
        auto memProps = getDeviceMemoryInfo().memProps;
        if (deviceLocalMemoryType_ < memProps.memoryTypeCount) {
            deviceLocalHeapIndex_ = memProps.memoryTypes[deviceLocalMemoryType_].heapIndex;
        }
    }

    // Detect VK_EXT_memory_budget for live budget checks.
    memoryBudgetAvailable_ = checkDeviceExtensionSupport(
        deviceConfig_.physicalDevice, {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME});
    if (memoryBudgetAvailable_) {
        VVM_LOG_INFO("VK_EXT_memory_budget available; budget-aware pool growth enabled");
    }

    // VK_EXT_debug_utils: name buffers/memory for RenderDoc/validation.
    debugUtilsEnabled_ = checkDeviceExtensionSupport(
        deviceConfig_.physicalDevice, {VK_EXT_DEBUG_UTILS_EXTENSION_NAME});
    if (debugUtilsEnabled_) {
        fnSetDebugName_ = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetDeviceProcAddr(
            device_, "vkSetDebugUtilsObjectNameEXT");
        if (!fnSetDebugName_) debugUtilsEnabled_ = false;
    }

    // Host-visible for shadow/offload
    if (config_.enableHostVisible) {
        auto hostVisibleResult = selector.select(
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            0,
            256 * 1024 * 1024  // 256MB minimum
        );
        if (hostVisibleResult.memoryTypeIndex != UINT32_MAX) {
            hostVisibleMemoryType_ = hostVisibleResult.memoryTypeIndex;
VVM_LOG_INFO("Selected HOST_VISIBLE memory type {} (heap budget: {} MB)",
                 hostVisibleMemoryType_,
                         hostVisibleResult.heapBudget / (1024*1024));
        } else {
            VVM_LOG_WARN("No suitable HOST_VISIBLE memory type found");
        }
    }
    } else {
        // Backend-driven discovery (HIP today): the backend's synthetic
        // topology replaces the Vulkan queries; policy stays pool-side.
        const auto types = backend_->memoryTypes();
        uint32_t typeIdx = UINT32_MAX;
        for (size_t i = 0; i < types.size(); ++i) {
            if (types[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                typeIdx = static_cast<uint32_t>(i);
                break;
            }
        }
        if (typeIdx != UINT32_MAX && config_.preferPureDeviceLocal &&
            (types[typeIdx].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            for (size_t i = 0; i < types.size(); ++i) {
                const uint32_t f = types[i].propertyFlags;
                if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                    !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                    typeIdx = static_cast<uint32_t>(i);
                    break;
                }
            }
        }
        if (typeIdx == UINT32_MAX) {
            VVM_LOG_ERROR("backend reports no DEVICE_LOCAL memory type");
            return false;
        }
        deviceLocalMemoryType_ = typeIdx;
        deviceLocalHeapIndex_ = types[typeIdx].heapIndex;

        const BackendBudget budget = backend_->heapBudget(deviceLocalHeapIndex_);
        memoryBudgetAvailable_ = budget.valid;
        if (budget.valid) {
            VVM_LOG_INFO("live device budget available ({} MB total, {} MB used) - "
                         "budget-aware pool growth enabled",
                         static_cast<uint64_t>(budget.budgetBytes) / (1024 * 1024),
                         static_cast<uint64_t>(budget.usedBytes) / (1024 * 1024));
        }
        VVM_LOG_INFO("Selected DEVICE_LOCAL memory type {} via {} backend (heap budget: {} MB, utilization: {} percent)",
                     deviceLocalMemoryType_, backend_->name(),
                     static_cast<uint64_t>(budget.budgetBytes) / (1024 * 1024),
                     budget.valid && budget.budgetBytes
                         ? (budget.usedBytes * 100.0f / budget.budgetBytes) : 0.0f);

        if (config_.enableHostVisible) {
            VVM_LOG_WARN("enableHostVisible is not supported by non-Vulkan backends yet - ignored");
        }
    }
    return true;
}

bool UnifiedMemoryPoolImpl::initTransferPool() {
    // Vulkan only: the HIP memory plane has no Vulkan queuing; the
    // copyBuffer path is a Vulkan-direct v1 API.
    // Bisect knobs (GGML_VVM_NO_CMDPOOL / GGML_VVM_NO_INITBLOCK via env are
    // handled by the integration layer flipping these config-adjacent statics;
    // here we only honor the internal skip flags).
    if (getenv("VVM_SKIP_CMDPOOL")) {
        return true;
    }
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = deviceConfig_.transferQueueFamily != UINT32_MAX
        ? deviceConfig_.transferQueueFamily
        : deviceConfig_.graphicsQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device_, &poolInfo, nullptr, &transferCmdPool_) != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to create transfer command pool");
        return false;
    }
    return true;
}

bool UnifiedMemoryPoolImpl::bootstrapFirstBlock() {
    // First block: best-fit, not config-size. On a device whose heap is already
    // mostly committed (context-init pools created after layer-split weight
    // placement), a blind 1 GiB initial block can OOM where a smaller first
    // block leaves enough budget for the pool to bootstrap and grow on demand.
    // Order: device-resident config ladder (smallest >= minAllocation),
    // then fall back to the configured block size, then to 256 MiB chunks.
    if (getenv("VVM_SKIP_INITBLOCK")) {
        return true;
    }
    VkDeviceSize firstBlock = 0;
    {
        VkDeviceSize smallestLadder = 0;
        for (VkDeviceSize bs : config_.blockSizes) {
            if (bs >= config_.minAlignment && (smallestLadder == 0 || bs < smallestLadder))
                smallestLadder = bs;
        }
        const VkDeviceSize candidates[4] = {
            smallestLadder ? smallestLadder : 0,
            config_.chunkBlockSize,
            config_.blockSize,
            256ull * 1024ull * 1024ull,
        };
        for (VkDeviceSize c : candidates) {
            if (c == 0) continue;
            if (!wouldExceedBudget(c)) { firstBlock = c; break; }
        }
        if (firstBlock == 0) {
            // Even 256 MiB exceeds the configured cap - the budget check is
            // advisory here (growth checks still apply); try 256 MiB once and
            // let vkAllocateMemory decide.
            firstBlock = 256ull * 1024ull * 1024ull;
        }
    }
    if (!allocateBlock(firstBlock, deviceLocalMemoryType_).has_value()) {
        VVM_LOG_ERROR("Failed to allocate initial memory block ({} MB)", firstBlock / (1024 * 1024));
        return false;
    }
    return true;
}
// Adaptive block sizing: size new buddy blocks to the observed request
// pattern instead of the static pick. Two rules, both cheap and local:
//   1. Pair-packing: if recent requests cluster around the current size,
//      size the block to hold TWO plus slack (pow2) instead of stranding a
//      tail the next sibling cannot reuse.
//   2. Small-heap cap: never commit more than a quarter of the device-local
//      heap in one block, so older/smaller GPUs get many small blocks rather
//      than a few huge ones (with a 256 MiB floor so the block stays useful).
// The result is snapped up to a power of two (buddy requirement) and never
// below the request itself. Static pick wins when adaptive is disabled.
VkDeviceSize UnifiedMemoryPoolImpl::adaptiveBlockSizeFor(VkDeviceSize requestSize,
                                                     VkDeviceSize staticPick) {
    // File-local pow2 helpers (buddy's own are private to the audited core).
    auto ceilPow2 = [](VkDeviceSize v) -> VkDeviceSize {
        if (v <= 1) return 1;
        --v;
        v |= v >> 1;  v |= v >> 2;  v |= v >> 4;
        v |= v >> 8;  v |= v >> 16; v |= v >> 32;
        return v + 1;
    };
    auto floorPow2 = [](VkDeviceSize v) -> VkDeviceSize {
        if (v == 0) return 0;
        VkDeviceSize r = 1;
        while ((r << 1) != 0 && (r << 1) <= v) r <<= 1;
        return r;
    };

    VkDeviceSize pick = staticPick;

    // Rule 1: cluster check over the recent window (sizes within ~[0.75x, 1.33x]).
    uint32_t cluster = 0;
    const uint32_t n = recentCount_ < kSizeHistory ? recentCount_ : kSizeHistory;
    for (uint32_t i = 0; i < n; ++i) {
        const VkDeviceSize s = recentSizes_[i];
        if (s >= (requestSize * 3) / 4 && s <= (requestSize * 4) / 3) ++cluster;
    }
    if (cluster >= 3) {
        const VkDeviceSize pairTarget = ceilPow2(2 * requestSize + config_.minAlignment);
        if (pairTarget > pick) pick = pairTarget;
    }

    // Rule 2: small-heap cap (a quarter of the device-local heap, floored).
    {
        uint64_t heapSize = 0;
        if (deviceConfig_.physicalDevice != VK_NULL_HANDLE) {
            const auto info = getDeviceMemoryInfo();
            if (deviceLocalHeapIndex_ < info.heapSizes.size()) {
                heapSize = info.heapSizes[deviceLocalHeapIndex_];
            }
        } else if (backend_) {
            const auto heaps = backend_->heaps();
            if (deviceLocalHeapIndex_ < heaps.size()) {
                heapSize = heaps[deviceLocalHeapIndex_].size;
            }
        }
        if (heapSize > 0) {
            VkDeviceSize cap = heapSize / 4;
            const VkDeviceSize floor = 256ull * 1024ull * 1024ull;
            if (cap < floor) cap = floor;
            if (pick > cap) {
                // Shrink to the largest pow2 <= cap that still fits the request.
                VkDeviceSize shrunk = floorPow2(cap);
                const VkDeviceSize need = ceilPow2(
                    alignUp(requestSize, config_.minAlignment));
                pick = (shrunk >= need) ? shrunk : need;
            }
        }
    }

    return pick;
}

std::optional<uint32_t> UnifiedMemoryPoolImpl::findMemoryType(
    VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) {
    
    return findMemoryTypeIndex(getDeviceMemoryInfo().memProps, required, preferred);
}

VkMemoryPropertyFlags UnifiedMemoryPoolImpl::usageToFlags(MemoryUsage usage) const {
    switch (usage) {
        case MemoryUsage::GpuOnly:
            return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        case MemoryUsage::CpuToGpu:   // staging / upload
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        case MemoryUsage::GpuToCpu:   // readback
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        case MemoryUsage::CpuCopy:    // HOST_VISIBLE staging
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        case MemoryUsage::Auto:
        default:
            return 0;  // pool default (device-local)
    }
}

bool UnifiedMemoryPoolImpl::reserve(VkDeviceSize bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes == 0) return true;
    // Sanity ceiling independent of fraction settings: a hold larger than
    // the entire heap can never be satisfied.
    const VkDeviceSize heap = heapSizeBytes();
    if (heap > 0 && bytes > heap) {
        VVM_LOG_WARN("reserve: {} MB exceeds heap size {} MB - refused",
                     bytes / (1024 * 1024), heap / (1024 * 1024));
        return false;
    }
    if (wouldExceedBudget(bytes)) {
        VVM_LOG_WARN("reserve: {} MB does not fit remaining budget",
                     bytes / (1024 * 1024));
        return false;
    }
    reservedBytes_ += bytes;
    VVM_LOG_INFO("reserve: holding {} MB (total reserved {} MB)",
                 bytes / (1024 * 1024), reservedBytes_ / (1024 * 1024));
    return true;
}

void UnifiedMemoryPoolImpl::unreserve(VkDeviceSize bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    const VkDeviceSize before = reservedBytes_;
    reservedBytes_ = (bytes >= reservedBytes_) ? 0 : reservedBytes_ - bytes;
    VVM_LOG_INFO("unreserve: released {} MB (reserved {} -> {} MB)",
                 (before - reservedBytes_) / (1024 * 1024),
                 before / (1024 * 1024), reservedBytes_ / (1024 * 1024));
}

VkDeviceSize UnifiedMemoryPoolImpl::reservedBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reservedBytes_;
}

// Static heap size on either path (Vulkan physical device or standalone
// backend). Caller must hold mutex_. Returns 0 when unknown.
VkDeviceSize UnifiedMemoryPoolImpl::heapSizeBytes() const {
    if (deviceConfig_.physicalDevice != VK_NULL_HANDLE) {
        const auto info = getDeviceMemoryInfo();
        if (deviceLocalHeapIndex_ < info.heapSizes.size()) {
            return info.heapSizes[deviceLocalHeapIndex_];
        }
        return 0;
    }
    if (backend_) {
        const auto heaps = backend_->heaps();
        if (deviceLocalHeapIndex_ < heaps.size()) {
            return heaps[deviceLocalHeapIndex_].size;
        }
    }
    return 0;
}

bool UnifiedMemoryPoolImpl::wouldExceedBudget(VkDeviceSize additionalBytes) const {
    // Note: caller must hold mutex_ if called from within another locked method
    VkDeviceSize currentPool = reservedBytes_;
    for (const auto& block : blocks_) currentPool = satAddU64(currentPool, block.size);
    for (const auto& alloc : dedicatedAllocations_) currentPool = satAddU64(currentPool, alloc.size);

    // Hard byte cap.
    if (config_.maxPoolBytes > 0 && satAddU64(currentPool, additionalBytes) > config_.maxPoolBytes) {
        VVM_LOG_WARN("budget: pool ({} MB) + {} MB would exceed maxPoolBytes ({} MB)",
                     currentPool / (1024 * 1024), additionalBytes / (1024 * 1024),
                     config_.maxPoolBytes / (1024 * 1024));
        return true;
    }

    // Heap-fraction cap (VK_EXT_memory_budget / hipMemGetInfo when available).
    if (config_.maxHeapFraction > 0.0f) {
        VkDeviceSize heapBudget = 0;
        VkDeviceSize heapUsed = 0;
        VkDeviceSize heapSize = 0;
        if (deviceConfig_.physicalDevice != VK_NULL_HANDLE) {
            VkPhysicalDeviceMemoryProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
            budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
            if (memoryBudgetAvailable_) {
                props2.pNext = &budget;
            }
            vkGetPhysicalDeviceMemoryProperties2(deviceConfig_.physicalDevice, &props2);
            const auto& memProps = props2.memoryProperties;
            if (deviceLocalHeapIndex_ < memProps.memoryHeapCount) {
                if (memoryBudgetAvailable_ && deviceLocalHeapIndex_ < VK_MAX_MEMORY_HEAPS) {
                    heapBudget = budget.heapBudget[deviceLocalHeapIndex_];
                    heapUsed = budget.heapUsage[deviceLocalHeapIndex_];
                }
                if (heapBudget == 0) {
                    heapBudget = memProps.memoryHeaps[deviceLocalHeapIndex_].size;
                }
                heapSize = memProps.memoryHeaps[deviceLocalHeapIndex_].size;
            }
        } else if (backend_) {
            const BackendBudget b = backend_->heapBudget(deviceLocalHeapIndex_);
            const auto heaps = backend_->heaps();
            heapBudget = b.valid ? b.budgetBytes
                                 : (deviceLocalHeapIndex_ < heaps.size()
                                        ? heaps[deviceLocalHeapIndex_].size : 0);
            heapUsed = b.valid ? b.usedBytes : 0;
            heapSize = deviceLocalHeapIndex_ < heaps.size()
                           ? heaps[deviceLocalHeapIndex_].size : heapBudget;
        }
        // Cap against the STATIC heap size, not the dynamic budget: VK_EXT
        // budget fluctuates with desktop/other-process VRAM usage, so a
        // budget-based cap fails unpredictably (measured: 256K q8 workload
        // at ~92% VRAM failed at fraction 0.95/0.98). Spill occurs when
        // system-wide usage crosses the heap size - that is the ceiling.
        // reservedBytes_ counts as committed here: the driver-reported usage
        // cannot see memory we intend to allocate but haven't yet.
        const VkDeviceSize cap = static_cast<VkDeviceSize>(heapSize * config_.maxHeapFraction);
        if (satAddU64(satAddU64(heapUsed, reservedBytes_), additionalBytes) > cap) {
            VVM_LOG_WARN("budget: heap usage {} MB + {} MB would exceed {} MB ({}% of {} MB cap); allocate() failing soft instead of stealing VRAM",
                         heapUsed / (1024 * 1024), additionalBytes / (1024 * 1024),
                         cap / (1024 * 1024), static_cast<int>(config_.maxHeapFraction * 100.0f),
                         heapBudget / (1024 * 1024));
            return true;
        }
    }
    return false;
}

void UnifiedMemoryPoolImpl::setDebugName(VkObjectType objectType, uint64_t objectHandle,
                                     const char* name) const {
    if (!debugUtilsEnabled_ || !fnSetDebugName_ || !name || objectHandle == 0) return;
    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = objectType;
    info.objectHandle = objectHandle;
    info.pObjectName = name;
    fnSetDebugName_(device_, &info);
}

bool UnifiedMemoryPoolImpl::validateDeviceCapabilities() const {
    bool ok = true;

    // Query 1.2 features (bufferDeviceAddress + timelineSemaphore live here,
    // not in VkPhysicalDeviceFeatures).
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features12;
    vkGetPhysicalDeviceFeatures2(deviceConfig_.physicalDevice, &features2);

    // Device address requires the bufferDeviceAddress feature at device creation.
    if (config_.enableDeviceAddress) {
        if (!features12.bufferDeviceAddress) {
            VVM_LOG_ERROR("PoolConfig.enableDeviceAddress is true but the device was NOT created "
                          "with the bufferDeviceAddress feature enabled");
            ok = false;
        }
    }

    // External memory export/import requires the KHR external memory extensions.
    if (config_.enableExternal) {
        std::vector<const char*> required = {
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        };
        #ifdef VVM_PLATFORM_LINUX
        required.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        #elif defined(VVM_PLATFORM_WINDOWS)
        required.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
        #endif
        if (!checkDeviceExtensionSupport(deviceConfig_.physicalDevice, required)) {
            VVM_LOG_ERROR("PoolConfig.enableExternal is true but the device is missing required "
                          "external memory extensions (VK_KHR_external_memory + fd/win32)");
            ok = false;
        }
    }

    // Soft checks: warn but do not fail (graceful fallbacks exist).
    if (!features12.timelineSemaphore) {
        VVM_LOG_WARN("timelineSemaphore feature not enabled; migration sync falls back to fences");
    }
    if (!checkDeviceExtensionSupport(deviceConfig_.physicalDevice,
                                     {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME})) {
        VVM_LOG_WARN("VK_EXT_memory_budget not available; budget-aware selection disabled");
    }

    if (!ok) {
        VVM_LOG_ERROR("Pool creation failed: device was not created with the required Vulkan "
                      "features/extensions. Recreate the VkDevice with them enabled (see "
                      "validateDeviceCapabilities in unified_memory_pool.cpp).");
    }
    return ok;
}

std::optional<VkDeviceMemory> UnifiedMemoryPoolImpl::allocateBlock(
    VkDeviceSize size, uint32_t memoryTypeIndex, bool isChunk, uint32_t chunkTier) {

    // Pool blocks are strictly NON-exportable. Cross-GPU sharing must use
    // allocateDedicatedExportable() which gives each exported allocation its
    // own dedicated VkDeviceMemory (required for reliable external import).
    //
    // NOTE: Do NOT chain a dedicated-allocation hint here for sub-allocated
    // blocks. A dedicated allocation is bound to a SINGLE resource.
    // Sub-allocating multiple buffers from one "dedicated" memory violates
    // the spec. Exportable allocations should use allocateDedicatedExportable()
    // instead.

    int err = 0;
    BackendAllocRequest req{
        static_cast<uint64_t>(size), memoryTypeIndex,
        /*exportable=*/false,
        /*deviceAddress=*/config_.enableDeviceAddress,
        /*priority=*/config_.memoryPriority,
        /*dedicatedFor=*/0};
    BackendMemory mem = backend_->allocate(req, &err);
    if (mem == 0 && req.priority > 0.0f) {
        // Driver-quirk retry: priority is a pure scheduling hint, but some
        // Windows drivers are finicky about the priority pNext chain on
        // large allocations. deviceAddress is load-bearing (ggml adds
        // SHADER_DEVICE_ADDRESS usage) and stays; retry once without
        // priority before giving up.
        VVM_LOG_INFO("backend allocate failed at priority {}; retrying without priority pNext "
                     "(size={}, type={})", req.priority, size, memoryTypeIndex);
        req.priority = 0.0f;
        mem = backend_->allocate(req, &err);
    }
    if (mem == 0) {
        int native = 0;
        const bool known = decode_backend_error(err, &native);
        VVM_LOG_ERROR("backend allocate failed: {} (size={}, type={})",
                      known ? vkResultToString(static_cast<VkResult>(native)).c_str()
                            : "unknown",
                      size, memoryTypeIndex);
        return std::nullopt;
    }
    VkDeviceMemory memory = reinterpret_cast<VkDeviceMemory>(mem);

    // VRAM high-water warning (VRAM_OVERFLOW_FINDINGS.md): committing past
    // ~90% of the device-local heap lets the driver spill to shared memory,
    // which silently halves decode throughput. Warn once per pool.
    if (config_.maxHeapFraction > 0.0f && !warnedHighWater_) {
        uint64_t budget = 0, used = 0;
        if (deviceConfig_.physicalDevice != VK_NULL_HANDLE) {
            auto info = getDeviceMemoryInfo();
            if (deviceLocalHeapIndex_ < VK_MAX_MEMORY_HEAPS) {
                budget = info.budget.heapBudget[deviceLocalHeapIndex_];
                used = info.budget.heapUsage[deviceLocalHeapIndex_];
            }
        } else {
            const BackendBudget b = backend_->heapBudget(deviceLocalHeapIndex_);
            if (b.valid) { budget = b.budgetBytes; used = b.usedBytes; }
        }
        if (budget > 0 && used > (uint64_t)(budget * 0.90)) {
            VVM_LOG_WARN("heap {} is {}% committed - the driver may spill to "
                         "shared memory and ~2x decode throughput. Reduce "
                         "context/model or rebalance across GPUs.",
                         deviceLocalHeapIndex_,
                         (int)(used * 100 / budget));
            warnedHighWater_ = true;
        }
    }
    
    // Map if host-visible
    void* hostPtr = nullptr;
    VkMemoryPropertyFlags memFlags = 0;
    if (deviceConfig_.physicalDevice != VK_NULL_HANDLE) {
        getMemoryTypeProperties(memoryTypeIndex, memFlags, getDeviceMemoryInfo().memProps);
    } else {
        const auto types = backend_->memoryTypes();
        if (memoryTypeIndex < types.size()) {
            memFlags = types[memoryTypeIndex].propertyFlags;
        }
    }
    
    bool isHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    bool isCoherent = (memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    
    if (isHostVisible) {
        bool mapOk = false;
        hostPtr = backend_->map(reinterpret_cast<uint64_t>(memory), &mapOk);
        if (!mapOk || !hostPtr) {
            VVM_LOG_WARN("backend map failed for host-visible block");
            hostPtr = nullptr;
            isHostVisible = false;
        }
    }
    
    BlockInfo block;
    block.memory = memory;
    block.size = size;
    block.used = 0;
    block.hostPtr = hostPtr;
    block.memoryFlags = memFlags;
    block.isHostVisible = isHostVisible;
    block.isCoherent = isCoherent;
    block.isChunk = isChunk;
    block.chunkTier = isChunk ? chunkTier : UINT32_MAX;
    
    // Create buddy allocator for this block
    block.buddy = std::make_unique<BuddyAllocator>(size, config_.minAlignment);
    
    blocks_.push_back(std::move(block));
    return memory;
}

// ---------------------------------------------------------------------------
// Vendor-aware export handle type selection.
// Some drivers reject bitmask union of handle types that they don't
// actually support, even though the spec says they should be fine.
// The candidate mask below is vendor-heuristic; filterExportableBits() then
// keeps only the bits the driver actually reports EXPORTABLE for this usage.
static VkExternalMemoryHandleTypeFlags getExportHandleTypes(VkPhysicalDevice phys) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    uint32_t vendor = props.vendorID;

    #ifdef VVM_PLATFORM_WINDOWS
    if (vendor == 0x10DE)
        return VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
    return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    #else
    return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
           VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    #endif
}

// Capability-driven selection (external audit P0): never advertise a handle
// type the driver does not report EXPORTABLE for this buffer usage. Each
// candidate bit is queried individually via
// vkGetPhysicalDeviceExternalBufferProperties.
static VkExternalMemoryHandleTypeFlags filterExportableBits(
    VkPhysicalDevice phys, VkBufferUsageFlags usage,
    VkExternalMemoryHandleTypeFlags candidates) {
    VkExternalMemoryHandleTypeFlags out = 0;
    uint32_t bit = 0;
    while (candidates >> bit) {
        auto ht = static_cast<VkExternalMemoryHandleTypeFlagBits>(1u << bit);
        if (candidates & (1u << bit)) {
            VkPhysicalDeviceExternalBufferInfo info{};
            info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
            info.usage = usage;
            info.handleType = ht;
            VkExternalBufferProperties props{};
            props.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
            vkGetPhysicalDeviceExternalBufferProperties(phys, &info, &props);
            if (props.externalMemoryProperties.externalMemoryFeatures &
                VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) {
                out |= (1u << bit);
            }
        }
        ++bit;
        if (bit >= 31) break;
    }
    return out;
}

// Allocate a dedicated VkDeviceMemory for a single exportable buffer.
        // ---------------------------------------------------------------------------
// Dedicated / exportable / import allocation paths.
//
// v1 BACKEND SCOPE NOTE: these paths remain Vulkan-direct. Cross-GPU
// export/import needs vendor queuing + handle-type semantics beyond the
// memory plane (see mem_backend.hpp v1 scope); route them through
// IDeviceMemoryBackend when the HIP/L0 adapters land.
// ---------------------------------------------------------------------------

std::optional<Allocation> UnifiedMemoryPoolImpl::allocateDedicatedExportable(
        VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) {
        
        VVM_LOG_INFO("allocateDedicatedExportable: size={}, usage={}, flags={}", size, usage, flags);
        
        std::lock_guard<std::mutex> lock(mutex_);
        VVM_LOG_INFO("allocateDedicatedExportable: lock acquired");
        
        size = alignUp(size, config_.minAlignment);
        VVM_LOG_INFO("allocateDedicatedExportable: aligned size={}", size);

// Step 1: Create buffer with VkExternalMemoryBufferCreateInfo
        VkExternalMemoryBufferCreateInfo extBufferInfo{};
        extBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        VkExternalMemoryHandleTypeFlags ht = filterExportableBits(
            deviceConfig_.physicalDevice, usage,
            getExportHandleTypes(deviceConfig_.physicalDevice));
        if (!ht) {
            VVM_LOG_ERROR("allocateDedicatedExportable: no EXPORTABLE external "
                          "handle type for usage={} on this device",
                          static_cast<unsigned>(usage));
            return std::nullopt;
        }
        extBufferInfo.handleTypes = ht;
        
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        if (config_.enableDeviceAddress) {
            bufferInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        }
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferInfo.pNext = &extBufferInfo;

        VkBuffer buffer;
        VkResult result = vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer);
        if (result != VK_SUCCESS) {
            VVM_LOG_ERROR("vkCreateBuffer failed for dedicated exportable: {}", vkResultToString(result).c_str());
            return std::nullopt;
        }
        
        // Step 2: Get memory requirements
        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device_, buffer, &memReq);
        VVM_LOG_INFO("allocateDedicatedExportable: memReq.size={}, memReq.memoryTypeBits={}", memReq.size, memReq.memoryTypeBits);
        
        // Budget check: fail soft instead of stealing VRAM past the configured cap.
        if (wouldExceedBudget(memReq.size)) {
            VVM_LOG_ERROR("allocateDedicatedExportable: would exceed budget");
            vkDestroyBuffer(device_, buffer, nullptr);
            return std::nullopt;
        }
VVM_LOG_INFO("allocateDedicatedExportable: budget check passed");

        // Re-configure memory type from memoryTypeBits (which accounts for
        // external handle type support) if the initial pre-selection doesn't match.
        uint32_t memType = deviceLocalMemoryType_;
        if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            memType = (hostVisibleMemoryType_ != UINT32_MAX) ? hostVisibleMemoryType_ : deviceLocalMemoryType_;
        }
        if ((memType >= 32) || ((memReq.memoryTypeBits & (1u << memType)) == 0)) {
            VkMemoryPropertyFlags requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
                requiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            }
            const auto& mp = getDeviceMemoryInfo().memProps;
            for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
                if ((memReq.memoryTypeBits & (1u << i)) &&
                    (mp.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags) {
                    memType = i;
                    break;
                }
            }
        }
        VVM_LOG_INFO("allocateDedicatedExportable: memType={} (memoryTypeBits={})", 
                     memType, memReq.memoryTypeBits);

VkMemoryDedicatedAllocateInfo dedicatedInfo{};
        dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicatedInfo.buffer = buffer;
        dedicatedInfo.image = VK_NULL_HANDLE;

        VkExportMemoryAllocateInfo exportInfo{};
        exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exportInfo.handleTypes = ht;
        exportInfo.pNext = &dedicatedInfo;
        
        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flagsInfo.pNext = &exportInfo;
        
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = memType;
        allocInfo.pNext = &flagsInfo;
        
        VkDeviceMemory memory;
        VVM_LOG_INFO("allocateDedicatedExportable: calling vkAllocateMemory (memType={}, size={})", memType, memReq.size);
        result = vkAllocateMemory(device_, &allocInfo, nullptr, &memory);
        VVM_LOG_INFO("allocateDedicatedExportable: vkAllocateMemory result={}", result);
        if (result != VK_SUCCESS) {
            VVM_LOG_ERROR("vkAllocateMemory failed for dedicated exportable: {}", vkResultToString(result).c_str());
            vkDestroyBuffer(device_, buffer, nullptr);
            return std::nullopt;
        }
        VVM_LOG_INFO("allocateDedicatedExportable: vkAllocateMemory succeeded");
        
        // Step 4: Bind buffer to memory
        VkBindBufferMemoryInfo bindInfo{};
        bindInfo.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
        bindInfo.buffer = buffer;
        bindInfo.memory = memory;
        bindInfo.memoryOffset = 0;
        
        result = vkBindBufferMemory2(device_, 1, &bindInfo);
        VVM_LOG_INFO("allocateDedicatedExportable: vkBindBufferMemory2 result={}", result);
        if (result != VK_SUCCESS) {
            VVM_LOG_ERROR("vkBindBufferMemory2 failed for dedicated exportable: {}", vkResultToString(result).c_str());
            vkFreeMemory(device_, memory, nullptr);
            vkDestroyBuffer(device_, buffer, nullptr);
            return std::nullopt;
        }
VVM_LOG_INFO("allocateDedicatedExportable: bind succeeded");
     
    // Step 5: Map if host-visible
    VVM_LOG_INFO("allocateDedicatedExportable: checking host visibility (memType={})", memType);
    VVM_LOG_INFO("allocateDedicatedExportable: calling getMemoryTypeProperties");
    void* hostPtr = nullptr;
    VkMemoryPropertyFlags memFlags;
    getMemoryTypeProperties(memType, memFlags, getDeviceMemoryInfo().memProps);
    VVM_LOG_INFO("allocateDedicatedExportable: getMemoryTypeProperties returned, memFlags={}", memFlags);
    
    bool isHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    bool isCoherent = (memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    VVM_LOG_INFO("allocateDedicatedExportable: isHostVisible={}, isCoherent={}", isHostVisible, isCoherent);
    
    if (isHostVisible) {
        VVM_LOG_INFO("allocateDedicatedExportable: calling vkMapMemory");
        result = vkMapMemory(device_, memory, 0, VK_WHOLE_SIZE, 0, &hostPtr);
        VVM_LOG_INFO("allocateDedicatedExportable: vkMapMemory result={}", result);
        if (result != VK_SUCCESS) {
            VVM_LOG_ERROR("vkMapMemory failed for dedicated exportable: {} (will not be mappable)", 
                          vkResultToString(result).c_str());
            hostPtr = nullptr;
            // Don't clear isHostVisible - the memory type IS host-visible, just mapping failed
        }
        VVM_LOG_INFO("allocateDedicatedExportable: hostPtr={}", hostPtr ? "non-null" : "null");
    } else {
        VVM_LOG_INFO("allocateDedicatedExportable: memory NOT host-visible, skipping vkMapMemory");
    }
    
    // Step 6: Get device address
    VkDeviceAddress deviceAddress = 0;
    if (config_.enableDeviceAddress) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = buffer;
        deviceAddress = vkGetBufferDeviceAddress(device_, &addrInfo);
    }
    
    // Step 7: Create Allocation (blockIndex = UINT32_MAX indicates dedicated)
    Allocation alloc;
    alloc.buffer = buffer;
    alloc.memory = memory;
    alloc.offset = 0;
    alloc.size = size;
    alloc.blockIndex = UINT32_MAX;  // Special marker for dedicated allocation
    alloc.isHostVisible = isHostVisible;
    alloc.isMapped = isHostVisible;
    alloc.isCoherent = isCoherent;
    alloc.isExternal = true;
    alloc.memoryFlags = memFlags;
    alloc.hostPtr = hostPtr;
    alloc.deviceAddress = deviceAddress;
    // Generation counter for handle validation
    alloc.generation = nextGeneration();
    
    VVM_LOG_INFO("allocateDedicatedExportable: pushing to dedicatedAllocations_");
    // Track dedicated allocation for cleanup in destructor
    dedicatedAllocations_.push_back(alloc);
    
    VVM_LOG_INFO("allocateDedicatedExportable: returning allocation (hostPtr={})", hostPtr ? "non-null" : "null");
    return alloc;
}

std::optional<Allocation> UnifiedMemoryPoolImpl::allocateDedicatedBackend(
    VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) {
    // Standalone backend allocation for non-Vulkan pools (HIP today):
    // the backend's dedicated equivalent is a plain hipMalloc - there is no
    // buffer/memory split, no pNext chain, no import/export. Tracked in
    // dedicatedAllocations_ so subDeallocate/dtor free it correctly.
    (void)usage;
    (void)flags;   // device-local type as selected at init

    if (wouldExceedBudget(size)) {
        VVM_LOG_WARN("allocateDedicatedBackend: would exceed budget ({} MB)",
                     size / (1024 * 1024));
        return std::nullopt;
    }

    int err = 0;
    const BackendMemory mem = backend_->allocate(
        {static_cast<uint64_t>(size), deviceLocalMemoryType_,
         /*exportable=*/false, /*deviceAddress=*/false,
         /*priority=*/0.0f, /*dedicatedFor=*/0},
        &err);
    if (mem == 0) {
        int native = 0;
        const bool known = decode_backend_error(err, &native);
        VVM_LOG_ERROR("allocateDedicatedBackend: backend allocate failed (err {})",
                      known ? native : err);
        return std::nullopt;
    }
    VVM_LOG_INFO("allocateDedicatedBackend: {} MB via backend (offset-0 standalone)",
                 static_cast<unsigned long long>(size / (1024 * 1024)));
    const BackendBuffer buf = backend_->create_buffer(
        mem, 0, static_cast<uint64_t>(size),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        /*exportable=*/false, &err);
    if (buf == 0) {
        VVM_LOG_ERROR("allocateDedicatedBackend: backend create_buffer failed");
        backend_->free(mem);
        return std::nullopt;
    }

    Allocation alloc;
    alloc.buffer = reinterpret_cast<VkBuffer>(buf);
    alloc.memory = reinterpret_cast<VkDeviceMemory>(mem);
    alloc.offset = 0;
    alloc.size = size;
    alloc.blockIndex = UINT32_MAX;
    alloc.isHostVisible = false;
    alloc.isCoherent = false;
    alloc.isMapped = false;
    alloc.memoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    alloc.hostPtr = nullptr;
    alloc.deviceAddress = backend_->buffer_device_address(buf);
    alloc.generation = nextGeneration();
    dedicatedAllocations_.push_back(alloc);
    return alloc;
}

std::optional<Allocation> UnifiedMemoryPoolImpl::allocateDedicated(
    VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) {
    // NOTE: caller (allocate) holds mutex_ -- do NOT lock here or we deadlock.

    size = alignUp(size, config_.minAlignment);

    // Step 1: Create the buffer (no external handle types -- plain dedicated).
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    if (config_.enableDeviceAddress) {
        bufferInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VkResult result = vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        VVM_LOG_ERROR("allocateDedicated: vkCreateBuffer failed: {}", vkResultToString(result).c_str());
        return std::nullopt;
    }

    // Step 2: Get memory requirements
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, buffer, &memReq);

    // Budget check: fail soft instead of stealing VRAM past the configured cap.
    if (wouldExceedBudget(memReq.size)) {
        VVM_LOG_ERROR("allocateDedicated: would exceed budget");
        vkDestroyBuffer(device_, buffer, nullptr);
        return std::nullopt;
    }

    // Step 3: Pick memory type honoring the caller's host-visible preference.
    // HOST_CACHED requests resolve the full flag set (cached GART over
    // write-combined ReBAR VRAM for CPU staging); otherwise the init-time
    // host-visible type applies.
    uint32_t memType = deviceLocalMemoryType_;
    if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        memType = (hostVisibleMemoryType_ != UINT32_MAX) ? hostVisibleMemoryType_ : deviceLocalMemoryType_;
        if ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) &&
            findMemoryType(flags, 0).has_value()) {
            memType = *findMemoryType(flags, 0);
        }
    }
    if ((memType >= 32) || ((memReq.memoryTypeBits & (1u << memType)) == 0)) {
        VkMemoryPropertyFlags requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            requiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        const auto& mp = getDeviceMemoryInfo().memProps;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((memReq.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags) {
                memType = i;
                break;
            }
        }
    }

    VkMemoryDedicatedAllocateInfo dedicatedInfo{};
    dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedInfo.buffer = buffer;
    dedicatedInfo.image = VK_NULL_HANDLE;
    // Optional hint: some drivers place dedicated-hinted allocations
    // differently. PoolConfig::dedicatedAllocateInfo = false allocates
    // generically (matching callers that never use the hint).
    const bool useDedicatedHint = config_.dedicatedAllocateInfo;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memType;
    // Both pNext structs MUST outlive the vkAllocateMemory call below. Declaring
    // VkMemoryAllocateFlagsInfo inside a nested block made allocInfo.pNext dangle
    // once the block ended, which crashed RADV on gfx1151. Chain order: dedicated
    // info first, then the device-address flags (spec VUID-VkMemoryAllocateInfo-pNext-00638).
    VkMemoryAllocateFlagsInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryPriorityAllocateInfoEXT priorityInfo{};
    if (config_.memoryPriority > 0.0f) {
        priorityInfo.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT;
        priorityInfo.priority = config_.memoryPriority;
    }
    {
        VkMemoryPriorityAllocateInfoEXT* tail = (config_.memoryPriority > 0.0f) ? &priorityInfo : nullptr;
        if (config_.enableDeviceAddress) {
            flagsInfo.pNext = tail;
            dedicatedInfo.pNext = &flagsInfo;
            allocInfo.pNext = useDedicatedHint ? static_cast<void*>(&dedicatedInfo) : static_cast<void*>(&flagsInfo);
        } else {
            dedicatedInfo.pNext = tail;
            allocInfo.pNext = useDedicatedHint ? static_cast<void*>(&dedicatedInfo) : static_cast<void*>(tail);
        }
    }

    VkDeviceMemory memory;
    result = vkAllocateMemory(device_, &allocInfo, nullptr, &memory);
    if (result != VK_SUCCESS) {
        VVM_LOG_ERROR("allocateDedicated: vkAllocateMemory failed: {}", vkResultToString(result).c_str());
        vkDestroyBuffer(device_, buffer, nullptr);
        return std::nullopt;
    }

    // Step 4: Bind buffer to memory
    VkBindBufferMemoryInfo bindInfo{};
    bindInfo.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
    bindInfo.buffer = buffer;
    bindInfo.memory = memory;
    bindInfo.memoryOffset = 0;

    result = vkBindBufferMemory2(device_, 1, &bindInfo);
    if (result != VK_SUCCESS) {
        VVM_LOG_ERROR("allocateDedicated: vkBindBufferMemory2 failed: {}", vkResultToString(result).c_str());
        vkFreeMemory(device_, memory, nullptr);
        vkDestroyBuffer(device_, buffer, nullptr);
        return std::nullopt;
    }

    // Step 5: Map if host-visible
    VkMemoryPropertyFlags memFlags;
    getMemoryTypeProperties(memType, memFlags, getDeviceMemoryInfo().memProps);
    bool isHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    bool isCoherent = (memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    void* hostPtr = nullptr;
    if (isHostVisible) {
        result = vkMapMemory(device_, memory, 0, VK_WHOLE_SIZE, 0, &hostPtr);
        if (result != VK_SUCCESS) {
            VVM_LOG_ERROR("allocateDedicated: vkMapMemory failed: {}", vkResultToString(result).c_str());
            hostPtr = nullptr;
        }
    }

    // Step 6: Get device address
    VkDeviceAddress deviceAddress = 0;
    if (config_.enableDeviceAddress) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = buffer;
        deviceAddress = vkGetBufferDeviceAddress(device_, &addrInfo);
    }

    // Step 7: Create Allocation (blockIndex = UINT32_MAX indicates dedicated)
    Allocation alloc;
    alloc.buffer = buffer;
    alloc.memory = memory;
    alloc.offset = 0;
    alloc.size = size;
    alloc.blockIndex = UINT32_MAX;  // Special marker for dedicated allocation
    alloc.isHostVisible = isHostVisible;
    alloc.isMapped = isHostVisible;
    alloc.isCoherent = isCoherent;
    alloc.isExternal = false;
    alloc.memoryFlags = memFlags;
    alloc.hostPtr = hostPtr;
    alloc.deviceAddress = deviceAddress;
    alloc.generation = nextGeneration();

    dedicatedAllocations_.push_back(alloc);
    return alloc;
}

std::optional<Allocation> UnifiedMemoryPoolImpl::allocate(VkDeviceSize size,
                                                        VkBufferUsageFlags usage,
                                                        VkMemoryPropertyFlags flags) {
    std::lock_guard<std::mutex> lock(mutex_);

    
    // Align size
    size = alignUp(size, config_.minAlignment);

    // Chonk Chunks: route small requests to tiered chunk blocks.
    // Tier resolution: first tier with size <= threshold wins; when chunkTiers
    // is empty the legacy single-tier pair acts as one implicit tier.
    // tier == UINT32_MAX means "not a chunk request" (buddy path).
    uint32_t tier = UINT32_MAX;
    VkDeviceSize tierBlock = 0;
    if (!config_.chunkTiers.empty()) {
        for (uint32_t i = 0; i < config_.chunkTiers.size(); ++i) {
            if (size <= config_.chunkTiers[i].first) {
                tier = i;
                tierBlock = config_.chunkTiers[i].second;
                break;
            }
        }
    } else if (config_.smallAllocThreshold > 0 && config_.chunkBlockSize > 0 &&
               size <= config_.smallAllocThreshold) {
        tier = 0;
        tierBlock = config_.chunkBlockSize;
    }
    const bool wantChunk = (tier != UINT32_MAX);

    // Record buddy-path request sizes for adaptive sizing (cheap ring store
    // under the already-held lock; chunk-path sizes have their own tiers).
    if (!wantChunk && config_.adaptiveBlockSize) {
        recentSizes_[recentIdx_ % kSizeHistory] = size;
        recentIdx_++;
        if (recentCount_ < kSizeHistory) recentCount_++;
    }

    // Best-fit block selection within the size class: pick the block with the
    // SMALLEST largest-free that still fits. First-fit scattered allocations
    // across blocks; best-fit packs them tightly and avoids spawning extra
    // partially-filled blocks. Cached requests (CPU staging) only match
    // cached blocks: falling back to a write-combined ReBAR block costs
    // ~100x on CPU reads (measured 41 MiB/s vs GB/s).
    const bool wantHostVisible = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    const bool wantHostCached = (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
    int bestIdx = -1;
    VkDeviceSize bestFree = UINT64_MAX;
    for (uint32_t i = 0; i < blocks_.size(); ++i) {
        const auto& block = blocks_[i];
        if (!block.buddy) continue;
        if (block.isHostVisible != wantHostVisible) continue;
        if (wantHostCached && !(block.memoryFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) continue;
        if (block.isChunk != wantChunk) continue;
        if (wantChunk && block.chunkTier != tier) continue; // per-tier packing
        const VkDeviceSize lf = block.buddy->getLargestFree();
        if (lf >= size && lf < bestFree) {
            bestFree = lf;
            bestIdx = static_cast<int>(i);
        }
    }
    if (bestIdx >= 0) {
        auto sub = subAllocate(size, config_.minAlignment, bestIdx, usage);
        if (sub.has_value()) {
            return sub;
        }
        // Aligned grant didn't fit the chosen block (e.g. base-alignment
        // padding overflows a full block) - fall through and create a fresh
        // block instead of failing the allocation.
    }

    // Need new block
    // maxBlocks == 0 means unlimited (documented contract in PoolConfig).
    if (config_.maxBlocks > 0 && blocks_.size() >= config_.maxBlocks) {
        // Oversized allocation: can't create another pool block. Fall back to
        // a dedicated allocation if the request still fits the budget.
        if (size > config_.blockSize) {
            return deviceConfig_.physicalDevice != VK_NULL_HANDLE
                       ? allocateDedicated(size, usage, flags)
                       : allocateDedicatedBackend(size, usage, flags);
        }
        return std::nullopt;
    }

    // Pick the block size for this request: the request's chunk tier, or the
    // regular configured ladder for buddy-path allocations, refined by the
    // adaptive sizer (pair-packing + small-heap cap) when enabled.
    VkDeviceSize blockSize = wantChunk ? tierBlock : config_.blockSize;
    if (!wantChunk && !config_.blockSizes.empty()) {
        // Pick the smallest block size >= request size (aligned)
        VkDeviceSize alignedSize = alignUp(size, config_.minAlignment);
        for (VkDeviceSize bs : config_.blockSizes) {
            if (bs >= alignedSize) {
                blockSize = bs;
                break;
            }
        }
        // If no blockSize in blockSizes fits, fall back to largest available
        if (blockSize == config_.blockSize && !config_.blockSizes.empty()) {
            blockSize = *std::max_element(config_.blockSizes.begin(), config_.blockSizes.end());
        }
    }
    if (!wantChunk && config_.adaptiveBlockSize) {
        blockSize = adaptiveBlockSizeFor(size, blockSize);
    }

    // Driver single-allocation cap (maxMemoryAllocationSize): Windows AMD /
    // Intel drivers refuse single vkAllocateMemory at/above ~4 GiB, and a
    // pool block above the cap can never be created (one block == one
    // backend allocation; VkBuffers cannot span blocks). A request the
    // driver cannot serve as one allocation cannot live in the pool either:
    // decline early so the caller (ggml fail-soft) routes it to native
    // memory instead of tripping a driver ERROR. Clamp adaptive overshoot
    // (pair-packing can exceed the request) to the cap. HIP/L0 backends
    // have no VkPhysicalDevice: unclamped.
    if (!wantChunk && deviceConfig_.physicalDevice != VK_NULL_HANDLE) {
        const VkDeviceSize cap = driverMaxSingleAlloc(deviceConfig_.physicalDevice);
        if (size > cap) {
            VVM_LOG_WARN("pool: request {} MB exceeds driver single-allocation max {} MB; "
                         "declining to pool (caller fail-soft applies)",
                         size / (1024*1024), cap / (1024*1024));
            return std::nullopt;
        }
        if (blockSize > cap) {
            blockSize = cap;
        }
    }
    
    // Budget check: fail soft instead of stealing VRAM past the configured cap.
    if (wouldExceedBudget(blockSize)) {
        return std::nullopt;
    }
    
    // Request larger than the picked block: allocate dedicated memory instead
    // of growing the pool by a block that still couldn't hold the request.
    if (size > blockSize) {
        return deviceConfig_.physicalDevice != VK_NULL_HANDLE
                   ? allocateDedicated(size, usage, flags)
                   : allocateDedicatedBackend(size, usage, flags);
    }
    
    uint32_t memType = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        ? (hostVisibleMemoryType_ != UINT32_MAX ? hostVisibleMemoryType_ : deviceLocalMemoryType_)
        : deviceLocalMemoryType_;
    // HOST_CACHED requests (CPU staging) resolve the full flag set: the
    // init-time host-visible type is ReBAR VRAM (write-combined, ~40 MB/s
    // CPU reads) while cached GART RAM memcpys at GB/s. Falls back to the
    // init type when no cached type exists.
    if ((flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) ==
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
        auto cached = findMemoryType(flags, 0);
        if (cached.has_value()) {
            memType = *cached;
        }
    }

    if (!allocateBlock(blockSize, memType, wantChunk, wantChunk ? tier : 0).has_value()) {
        return std::nullopt;
    }
    
    auto sub = subAllocate(size, config_.minAlignment, blocks_.size() - 1, usage);
    if (sub.has_value()) {
        return sub;
    }
    // Last resort: dedicated memory (e.g. aligned grant could not fit).
    return allocateDedicated(size, usage, flags);
}

std::optional<Allocation> UnifiedMemoryPoolImpl::allocate(const AllocDesc& desc) {
    if (desc.exportable) {
        auto alloc = allocateDedicatedExportable(desc.size, desc.usage,
                                                 usageToFlags(desc.memoryUsage));
        if (alloc && !desc.name.empty()) {
            setDebugName(VK_OBJECT_TYPE_BUFFER, (uint64_t)alloc->buffer, desc.name.c_str());
            setDebugName(VK_OBJECT_TYPE_DEVICE_MEMORY, (uint64_t)alloc->memory, desc.name.c_str());
        }
        return alloc;
    }
    auto alloc = allocate(desc.size, desc.usage, usageToFlags(desc.memoryUsage));
    if (alloc && !desc.name.empty()) {
        setDebugName(VK_OBJECT_TYPE_BUFFER, (uint64_t)alloc->buffer, desc.name.c_str());
    }
    return alloc;
}

std::optional<Allocation> UnifiedMemoryPoolImpl::allocateTensor(VkDeviceSize size,
                                                             VkBufferUsageFlags usage) {
    return allocate(size, usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
}

void UnifiedMemoryPoolImpl::deallocate(Allocation&& alloc) {
    std::lock_guard<std::mutex> lock(mutex_);
    deallocateLocked(std::move(alloc));
}

void UnifiedMemoryPoolImpl::deallocateLocked(Allocation&& alloc) {
    // Validate generation counter to prevent stale handle use
    if (!isValidGeneration(alloc.generation)) {
        VVM_LOG_WARN("deallocate: stale allocation handle (generation {}) rejected", alloc.generation);
        // Zero the stale copy so it can never be re-freed (a second pass with
        // dead handles surfaces later as vkUnmapMemory/vkFreeMemory parameter
        // errors or double-free corruption).
        alloc.buffer = VK_NULL_HANDLE;
        alloc.memory = VK_NULL_HANDLE;
        alloc.hostPtr = nullptr;
        alloc.deviceAddress = 0;
        return;
    }

    // Retire the generation BEFORE freeing. subDeallocate must not re-validate:
    // the generation has already been checked here exactly once.
    retireGeneration(alloc.generation);
    alloc.generation = 0;

    if (alloc.blockIndex == UINT32_MAX) {
        // Dedicated allocation - destroy directly
        subDeallocate(std::move(alloc));
    } else if (alloc.blockIndex < blocks_.size()) {
        subDeallocate(std::move(alloc));
    }
}

void UnifiedMemoryPoolImpl::deallocate(UniqueAllocation&& alloc) {
    if (!alloc) return;
    
    // Extract the raw allocation and deallocate
    Allocation rawAlloc = alloc.release();
    deallocate(std::move(rawAlloc));
}

// ---------------------------------------------------------------------------
// Retirement queue: GPU-lifetime-safe reclamation.
// ---------------------------------------------------------------------------

bool UnifiedMemoryPoolImpl::ensureRetireTimeline() {
    if (retireTimeline_) return true;
    if (!isVulkanBackend() || device_ == VK_NULL_HANDLE) return false;
    VkSemaphoreTypeCreateInfo typeInfo{};
    typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    info.pNext = &typeInfo;
    if (vkCreateSemaphore(device_, &info, nullptr, &retireTimeline_) != VK_SUCCESS) {
        // Timeline semaphores need the feature enabled at device creation;
        // without it there is no GPU-completion tracking on this pool.
        retireTimeline_ = VK_NULL_HANDLE;
        return false;
    }
    retireNextValue_ = 1;  // 0 is the born-signaled value; first real value is 1
    return true;
}

std::pair<VkSemaphore, uint64_t> UnifiedMemoryPoolImpl::retireTicket() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensureRetireTimeline()) return {VK_NULL_HANDLE, 0};
    return {retireTimeline_, retireNextValue_++};
}

bool UnifiedMemoryPoolImpl::retire(Allocation&& alloc, VkSemaphore timeline,
                                   uint64_t value, VkCommandBuffer cmd,
                                   VkCommandPool cmdPool) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!isVulkanBackend() || device_ == VK_NULL_HANDLE || !timeline) return false;
    // cmd without a pool (or vice versa) cannot be torn down at reclaim.
    if ((cmd == VK_NULL_HANDLE) != (cmdPool == VK_NULL_HANDLE)) return false;
    if (!isValidGeneration(alloc.generation)) return false;
    // Generation stays live across the retire window: the registry copy in
    // dedicatedAllocations_ must not be freed by anyone meanwhile, and the
    // double-free guard must see exactly one deallocate at collect().
    // (trim()/defragment() never touch dedicateds or live buddy ranges.)
    retired_.push_back(RetirementItem{std::move(alloc), cmd, cmdPool, timeline, value});
    return true;
}

uint32_t UnifiedMemoryPoolImpl::collect() {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t reclaimed = 0;
    for (auto it = retired_.begin(); it != retired_.end();) {
        uint64_t cur = 0;
        // Query failure (device lost) keeps the item queued: freeing GPU
        // memory against a lost device is unsafe, leaking is safer.
        if (it->timeline &&
            vkGetSemaphoreCounterValue(device_, it->timeline, &cur) == VK_SUCCESS &&
            cur >= it->value) {
            if (it->cmd) {
                const VkCommandPool pool = it->cmdPool ? it->cmdPool : transferCmdPool_;
                if (pool) vkFreeCommandBuffers(device_, pool, 1, &it->cmd);
                if (it->cmdPool) vkDestroyCommandPool(device_, it->cmdPool, nullptr);
            }
            deallocateLocked(std::move(it->allocation));
            it = retired_.erase(it);
            ++reclaimed;
        } else {
            ++it;
        }
    }
    return reclaimed;
}

std::optional<Allocation> UnifiedMemoryPoolImpl::importMemoryHostPointer(
    void* hostPtr, VkDeviceSize size, VkBufferUsageFlags usage) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!hostPtr || size == 0 || device_ == VK_NULL_HANDLE) {
        return std::nullopt;
    }
    // Resolved dynamically (loader import libs may lack extension exports;
    // device proc addr dispatches device-level extension commands on 1.1+
    // loaders regardless).
    auto getProps = (PFN_vkGetMemoryHostPointerPropertiesEXT)
        vkGetDeviceProcAddr(device_, "vkGetMemoryHostPointerPropertiesEXT");
    if (!getProps) {
        VVM_LOG_ERROR("importHostPointer: loader lacks vkGetMemoryHostPointerPropertiesEXT");
        return std::nullopt;
    }
    VkMemoryHostPointerPropertiesEXT mpp{};
    mpp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    if (getProps(device_, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                 hostPtr, &mpp) != VK_SUCCESS || mpp.memoryTypeBits == 0) {
        VVM_LOG_ERROR("importHostPointer: host pointer properties refused "
                      "(VK_EXT_external_memory_host enabled on the device?)");
        return std::nullopt;
    }
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < 32; ++i) {
        if (mpp.memoryTypeBits & (1u << i)) { mt = i; break; }
    }
    if (mt == UINT32_MAX) {
        VVM_LOG_ERROR("importHostPointer: no importable memory type in pointer properties");
        return std::nullopt;
    }

    VkImportMemoryHostPointerInfoEXT imp{};
    imp.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = hostPtr;
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = &imp;
    ai.allocationSize = size;
    ai.memoryTypeIndex = mt;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(device_, &ai, nullptr, &mem) != VK_SUCCESS) {
        VVM_LOG_ERROR("importHostPointer: vkAllocateMemory(import) failed");
        return std::nullopt;
    }
    VkBuffer buffer = VK_NULL_HANDLE;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &bi, nullptr, &buffer) != VK_SUCCESS) {
        vkFreeMemory(device_, mem, nullptr);
        VVM_LOG_ERROR("importHostPointer: vkCreateBuffer failed");
        return std::nullopt;
    }
    if (vkBindBufferMemory(device_, buffer, mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(device_, buffer, nullptr);
        vkFreeMemory(device_, mem, nullptr);
        VVM_LOG_ERROR("importHostPointer: vkBindBufferMemory failed");
        return std::nullopt;
    }

    Allocation alloc{};
    alloc.buffer = buffer;
    alloc.memory = mem;
    alloc.offset = 0;
    alloc.size = size;
    alloc.deviceAddress = 0;
    // hostPtr stays NULL: the caller owns the arena pointer; a null keeps
    // dealloc/dtor from vkUnmapMemory on memory the pool never mapped.
    alloc.hostPtr = nullptr;
    alloc.blockIndex = UINT32_MAX;   // dedicated-style tracking
    alloc.isHostVisible = true;
    alloc.isExternal = true;
    {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(deviceConfig_.physicalDevice, &mp);
        if (mt < mp.memoryTypeCount) {
            alloc.memoryFlags = mp.memoryTypes[mt].propertyFlags;
            alloc.isCoherent = (alloc.memoryFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        }
    }
    alloc.generation = nextGeneration();
    dedicatedAllocations_.push_back(alloc);
    VVM_LOG_INFO("importHostPointer: imported {} MB (type {}, buffer bound)",
                 size / (1024*1024), mt);
    return alloc;
}

std::optional<ExternalMemoryInfo> UnifiedMemoryPoolImpl::exportMemory(
    const Allocation& alloc, ExternalHandleType type) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Only support dedicated allocations for cross-GPU export.
    // Sub-allocated blocks are NOT supported for cross-GPU export because
    // Vulkan external memory import requires dedicated allocations for
    // reliable cross-device import. Sub-allocating from a shared block
    // and then exporting the whole block (or a sub-range) is fragile
    // across vendors and violates the Vulkan spec for reliable import.
    if (alloc.blockIndex != UINT32_MAX) {
        VVM_LOG_ERROR("exportMemory: only dedicated allocations (blockIndex == UINT32_MAX) are supported for cross-GPU export. Sub-allocated blocks are not supported.");
        return std::nullopt;
    }
    
    // Dedicated allocation - export directly from alloc.memory
    ExternalMemoryInfo info;
    info.type = type;
    info.size = alloc.size;
    info.memoryTypeIndex = UINT32_MAX;  // Will be re-selected by importer
    info.dedicatedAllocation = true;
    
    #if defined(VVM_PLATFORM_ANDROID)
    if (type == ExternalHandleType::AndroidHardwareBuffer) {
        VkMemoryGetAndroidHardwareBufferInfoANDROID ahbInfo{};
        ahbInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
        ahbInfo.memory = alloc.memory;
        
        PFN_vkGetMemoryAndroidHardwareBufferANDROID vkGetMemoryAndroidHardwareBufferANDROID = 
            (PFN_vkGetMemoryAndroidHardwareBufferANDROID)vkGetDeviceProcAddr(device_, "vkGetMemoryAndroidHardwareBufferANDROID");
        if (!vkGetMemoryAndroidHardwareBufferANDROID) return std::nullopt;
        
        AHardwareBuffer* ahb = nullptr;
        if (vkGetMemoryAndroidHardwareBufferANDROID(device_, &ahbInfo, &ahb) != VK_SUCCESS) return std::nullopt;
        
        info.handle = ExternalHandle(ahb);  // RAII wrapper takes ownership
        return info;
    }
    #elif defined(VVM_PLATFORM_LINUX)
    if (type == ExternalHandleType::OpaqueFd || type == ExternalHandleType::DmaBuf) {
        VkMemoryGetFdInfoKHR fdInfo{};
        fdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fdInfo.memory = alloc.memory;
        
        switch (type) {
            case ExternalHandleType::OpaqueFd:
                fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
                break;
            case ExternalHandleType::DmaBuf:
                fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
                break;
            default:
                return std::nullopt;
        }
        
        PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR =
            (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
        if (!vkGetMemoryFdKHR) {
            VVM_LOG_ERROR("exportMemory: vkGetMemoryFdKHR unavailable - was VK_KHR_external_memory_fd "
                          "enabled at device creation? For DmaBuf also VK_EXT_external_memory_dma_buf.");
            return std::nullopt;
        }

        int fd = -1;
        if (vkGetMemoryFdKHR(device_, &fdInfo, &fd) != VK_SUCCESS) {
            VVM_LOG_ERROR("exportMemory: vkGetMemoryFdKHR failed (handleType={}) - verify the allocation "
                          "was created with VkExportMemoryAllocateInfo and, for DMA-BUF, that "
                          "VK_EXT_external_memory_dma_buf was enabled at device creation",
                          fdInfo.handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT ? "DMA_BUF" : "OPAQUE_FD");
            return std::nullopt;
        }
        info.handle = ExternalHandle(fd);  // RAII wrapper takes ownership
        return info;
    }
    #elif defined(VVM_PLATFORM_WINDOWS)
    if (type == ExternalHandleType::OpaqueWin32 || type == ExternalHandleType::D3D12Heap ||
        type == ExternalHandleType::D3D12Resource) {
        VkMemoryGetWin32HandleInfoKHR handleInfo{};
        handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handleInfo.memory = alloc.memory;
        
        switch (type) {
            case ExternalHandleType::OpaqueWin32:
                handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
                break;
            case ExternalHandleType::D3D12Heap:
                handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
                break;
            case ExternalHandleType::D3D12Resource:
                handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
                break;
            default:
                return std::nullopt;
        }
        
        PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = 
            (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR");
        if (!vkGetMemoryWin32HandleKHR) return std::nullopt;
        
        HANDLE handle = nullptr;
        if (vkGetMemoryWin32HandleKHR(device_, &handleInfo, &handle) != VK_SUCCESS) return std::nullopt;
        info.handle = ExternalHandle(handle);  // RAII wrapper takes ownership
        return info;
    }
    #endif
    
    return std::nullopt;
}

std::optional<Allocation> UnifiedMemoryPoolImpl::importMemory(
    ExternalMemoryInfo&& info, VkBufferUsageFlags usage) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Step 1: Determine the VkExternalMemoryHandleTypeFlagBits for import
    VkExternalMemoryHandleTypeFlagBits importHandleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(0);
    #ifdef VVM_PLATFORM_ANDROID
    if (info.type == ExternalHandleType::AndroidHardwareBuffer) {
        importHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
    } else
    #endif
    #ifdef VVM_PLATFORM_LINUX
    if (info.type == ExternalHandleType::OpaqueFd) {
        importHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    } else if (info.type == ExternalHandleType::DmaBuf) {
        importHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    } else
    #endif
    #ifdef VVM_PLATFORM_WINDOWS
    if (info.type == ExternalHandleType::OpaqueWin32) {
        importHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    } else if (info.type == ExternalHandleType::D3D12Heap) {
        importHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
    } else if (info.type == ExternalHandleType::D3D12Resource) {
        importHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    } else
    #endif
    {
        VVM_LOG_ERROR("Unsupported external handle type for import");
        return std::nullopt;
    }
    
    // Step 2: Re-select memory type on THIS (destination) device
    // NEVER trust the source device's memoryTypeIndex - it's not portable across devices
    VkMemoryPropertyFlags requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT || usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) {
        // For staging buffers, we might need host-visible
    }
    
    auto memTypeOpt = findImportMemoryTypeIndex(
        deviceConfig_.physicalDevice,
        info.memoryTypeIndex,  // Source device's index (hint only)
        requiredFlags,
        importHandleType
    );
    
    if (!memTypeOpt) {
        VVM_LOG_ERROR("Failed to find compatible memory type for import on destination device");
        return std::nullopt;
    }
    uint32_t memoryTypeIndex = *memTypeOpt;
    VVM_LOG_INFO("Import: re-selected memory type index {} on destination device", memoryTypeIndex);
    
    // Step 3: Create buffer FIRST with VkExternalMemoryBufferCreateInfo
    // This is required for imported external memory
    VkExternalMemoryBufferCreateInfo extBufferInfo{};
    extBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extBufferInfo.handleTypes = importHandleType;
    
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = info.size;
    bufferInfo.usage = usage;
    if (config_.enableDeviceAddress) {
        // Step 6 queries vkGetBufferDeviceAddress on this buffer; without this
        // usage bit that query is invalid (VUID-VkBufferDeviceAddressInfo-buffer-02601).
        bufferInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo.pNext = &extBufferInfo;
    
    VkBuffer buffer;
    VkResult result = vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to create buffer for imported memory: {}", vkResultToString(result).c_str());
        return std::nullopt;
    }
    
    // Step 4: Build import chain with the ACTUAL buffer in VkMemoryDedicatedAllocateInfo
    VkImportMemoryFdInfoKHR importFdInfo{};
#ifdef VVM_PLATFORM_WINDOWS
    VkImportMemoryWin32HandleInfoKHR importWin32Info{};
#endif
#ifdef VVM_PLATFORM_ANDROID
    VkImportAndroidHardwareBufferInfoANDROID importAhbInfo{};
#endif
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    // The exporter reports the logical size (info.size), but a dedicated
    // import must satisfy THIS device's memory requirements for the buffer
    // we just created. Alignment requirements differ between devices, so
    // allocate max(exported size, local requirement).
    VkMemoryRequirements importMemReq;
    vkGetBufferMemoryRequirements(device_, buffer, &importMemReq);
    allocInfo.allocationSize = std::max<VkDeviceSize>(info.size, importMemReq.size);
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    
    void* pNext = nullptr;
    
#if defined(VVM_PLATFORM_ANDROID)
    if (info.type == ExternalHandleType::AndroidHardwareBuffer && info.handle) {
        importAhbInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
        importAhbInfo.buffer = info.handle.get();
        importAhbInfo.pNext = pNext;
        pNext = &importAhbInfo;
    }
#elif defined(VVM_PLATFORM_LINUX)
    if (info.type == ExternalHandleType::OpaqueFd && info.handle) {
        importFdInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        importFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        importFdInfo.fd = info.handle.get();  // Use RAII wrapper's get()
        importFdInfo.pNext = pNext;
        pNext = &importFdInfo;
    } else if (info.type == ExternalHandleType::DmaBuf && info.handle) {
        importFdInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        importFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        importFdInfo.fd = info.handle.get();  // Use RAII wrapper's get()
        importFdInfo.pNext = pNext;
        pNext = &importFdInfo;
    }
#elif defined(VVM_PLATFORM_WINDOWS)
    if (info.type == ExternalHandleType::OpaqueWin32 && info.handle) {
        importWin32Info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        importWin32Info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        importWin32Info.handle = info.handle.get();  // Use RAII wrapper's get()
        importWin32Info.pNext = pNext;
        pNext = &importWin32Info;
    } else if (info.type == ExternalHandleType::D3D12Heap && info.handle) {
        importWin32Info.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
        importWin32Info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
        importWin32Info.handle = info.handle.get();  // Use RAII wrapper's get()
        importWin32Info.pNext = pNext;
        pNext = &importWin32Info;
    }
#endif
    
    // Dedicated allocation for imported memory - NOW with the actual buffer!
    VkMemoryDedicatedAllocateInfo dedicatedInfo{};
    dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicatedInfo.buffer = buffer;
    dedicatedInfo.image = VK_NULL_HANDLE;
    dedicatedInfo.pNext = pNext;
    pNext = &dedicatedInfo;
    
    // Device address support
    VkMemoryAllocateFlagsInfo flagsInfo{};
    if (config_.enableDeviceAddress) {
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flagsInfo.pNext = pNext;
        pNext = &flagsInfo;
    }
    
    allocInfo.pNext = pNext;

    VkDeviceMemory memory;
    VkResult allocResult = vkAllocateMemory(device_, &allocInfo, nullptr, &memory);
    if (allocResult != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to allocate memory for import: {} (VkResult={}) [memTypeIndex={} size={} handleType={}]",
                      vkResultToString(allocResult).c_str(), static_cast<int>(allocResult),
                      memoryTypeIndex, static_cast<unsigned long long>(info.size),
                      static_cast<unsigned>(importHandleType));
        vkDestroyBuffer(device_, buffer, nullptr);
        // Handle NOT consumed: info keeps ownership and its destructor closes it.
        return std::nullopt;
    }
    
    // On success the driver owns the OS handle (FD/HANDLE). Release it from
    // the RAII wrapper so its destructor does NOT double-close it.
    info.handle.release();

    // Step 5: Bind buffer to memory
    VkBindBufferMemoryInfo bindInfo{};
    bindInfo.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
    bindInfo.buffer = buffer;
    bindInfo.memory = memory;
    bindInfo.memoryOffset = 0;

    if (vkBindBufferMemory2(device_, 1, &bindInfo) != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to bind imported memory");
        vkDestroyBuffer(device_, buffer, nullptr);
        vkFreeMemory(device_, memory, nullptr);
        return std::nullopt;
    }
    
    // Step 6: Get device address
    VkDeviceAddress deviceAddress = 0;
    if (config_.enableDeviceAddress) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = buffer;
        deviceAddress = vkGetBufferDeviceAddress(device_, &addrInfo);
    }
    
    // Step 7: Create Allocation (blockIndex = UINT32_MAX for dedicated import)
    Allocation alloc;
    alloc.buffer = buffer;
    alloc.memory = memory;
    alloc.offset = 0;
    alloc.size = info.size;
    alloc.blockIndex = UINT32_MAX;  // Dedicated import
    alloc.isHostVisible = false;
    alloc.isMapped = false;
    alloc.isCoherent = false;
    alloc.isExternal = true;
    alloc.memoryFlags = 0;
    alloc.hostPtr = nullptr;
    alloc.deviceAddress = deviceAddress;

    // Generation REQUIRED: deallocate() validates against liveGenerations_;
    // without this, every imported allocation is rejected as "stale" and
    // leaks its VkDeviceMemory + VkBuffer until teardown.
    alloc.generation = nextGeneration();

    // Track dedicated allocation for cleanup in destructor
    dedicatedAllocations_.push_back(alloc);

    return alloc;
}

bool UnifiedMemoryPoolImpl::copyBuffer(const Allocation& src, const Allocation& dst,
                                   VkDeviceSize srcOffset, VkDeviceSize dstOffset,
                                   VkDeviceSize size, VkFence fence) {
    if (!device_ || src.buffer == VK_NULL_HANDLE || dst.buffer == VK_NULL_HANDLE ||
        size == 0) return false;

    VkQueue queue = deviceConfig_.transferQueue != VK_NULL_HANDLE
                        ? deviceConfig_.transferQueue
                        : deviceConfig_.graphicsQueue;
    uint32_t family = deviceConfig_.transferQueueFamily != UINT32_MAX
                          ? deviceConfig_.transferQueueFamily
                          : deviceConfig_.graphicsQueueFamily;
    if (queue == VK_NULL_HANDLE) return false;

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cp.queueFamilyIndex = family;
    if (vkCreateCommandPool(device_, &cp, nullptr, &pool) != VK_SUCCESS) return false;

    VkCommandBufferAllocateInfo cba{};
    cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = pool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &cba, &cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(device_, pool, nullptr);
        return false;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkBufferCopy region{};
    region.srcOffset = srcOffset;
    region.dstOffset = dstOffset;
    region.size = size;
    VkResult rc = VK_SUCCESS;
    if ((rc = vkBeginCommandBuffer(cmd, &bi)) != VK_SUCCESS ||
        (vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &region),
         (rc = vkEndCommandBuffer(cmd)) != VK_SUCCESS)) {
        vkDestroyCommandPool(device_, pool, nullptr);
        return false;
    }

    bool waitInternal = (fence == VK_NULL_HANDLE);
    VkFence done = fence;
    VkFence internalFence = VK_NULL_HANDLE;
    if (waitInternal) {
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device_, &fi, nullptr, &internalFence) != VK_SUCCESS) {
            vkDestroyCommandPool(device_, pool, nullptr);
            return false;
        }
        done = internalFence;
    }

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    rc = vkQueueSubmit(queue, 1, &si, done);

    if (waitInternal && rc == VK_SUCCESS) {
        rc = vkWaitForFences(device_, 1, &internalFence, VK_TRUE, UINT64_MAX);
    }

    if (internalFence) vkDestroyFence(device_, internalFence, nullptr);
    vkDestroyCommandPool(device_, pool, nullptr);
    return rc == VK_SUCCESS;
}

std::optional<MigrationOperation> UnifiedMemoryPoolImpl::offloadToHost(Allocation& alloc) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!offloadManager_) {
        VVM_LOG_WARN("offloadToHost called but OffloadManager not initialized");
        return std::nullopt;
    }
    return offloadManager_->offload(alloc);
}

std::optional<MigrationOperation> UnifiedMemoryPoolImpl::reloadToDevice(Allocation& alloc) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!offloadManager_) {
        VVM_LOG_WARN("reloadToDevice called but OffloadManager not initialized");
        return std::nullopt;
    }
    return offloadManager_->reload(alloc);
}

void UnifiedMemoryPoolImpl::waitMigration(const MigrationOperation& op) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (op.completionFence) {
        vkWaitForFences(device_, 1, &op.completionFence, VK_TRUE, UINT64_MAX);
    }
}

PoolStats UnifiedMemoryPoolImpl::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    PoolStats stats;
    stats.dedicatedCount = static_cast<uint32_t>(dedicatedAllocations_.size());
    stats.totalCapacity = config_.maxPoolBytes;
    for (const auto& block : blocks_) {
        stats.totalAllocated += block.size;
        stats.totalUsed += block.used;
        stats.totalFree += (block.size - block.used);
        stats.blockCount++;
        stats.totalCapacity += block.size;
        
        if (block.buddy) {
            stats.largestFreeBlock = std::max(stats.largestFreeBlock, block.buddy->getLargestFree());
            stats.allocationCount += static_cast<uint32_t>(block.buddy->getAllocationCount());
        }
    }
    // Dedicated allocations each own their full VkDeviceMemory. Count them so
    // getStats() reflects all live memory, not just sub-allocated blocks.
    // (No free space inside a dedicated allocation -- it is fully committed.)
    for (const auto& alloc : dedicatedAllocations_) {
        stats.totalAllocated += alloc.size;
        stats.totalUsed += alloc.size;
        stats.allocationCount++;
    }
    stats.reservedBytes = reservedBytes_;
    
    if (stats.totalAllocated > 0) {
        stats.fragmentationRatio = 1.0f - 
            static_cast<float>(stats.largestFreeBlock) / 
            static_cast<float>(stats.totalFree > 0 ? stats.totalFree : 1);
    }
    
    return stats;
}

DeviceMemoryInfo UnifiedMemoryPoolImpl::getDeviceMemoryInfo() const {
    // Note: caller must hold mutex_ if called from within a locked method
    DeviceMemoryInfo info;
    vkGetPhysicalDeviceMemoryProperties(deviceConfig_.physicalDevice, &info.memProps);
    
    // Query budget if extension available
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    
    VkPhysicalDeviceMemoryProperties2 memProps2{};
    memProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memProps2.pNext = &budget;
    
    vkGetPhysicalDeviceMemoryProperties2(deviceConfig_.physicalDevice, &memProps2);
    info.budget = budget;
    
    info.heapSizes.resize(info.memProps.memoryHeapCount);
    info.heapUsed.resize(info.memProps.memoryHeapCount);
    for (uint32_t i = 0; i < info.memProps.memoryHeapCount; ++i) {
        info.heapSizes[i] = info.memProps.memoryHeaps[i].size;
        info.heapUsed[i] = budget.heapUsage[i];
    }
    
    return info;
}

void UnifiedMemoryPoolImpl::defragment() {
    std::lock_guard<std::mutex> lock(mutex_);
    // The buddy allocator already coalesces adjacent free ranges on every
    // deallocate, and the pool has no registry of user-owned VkBuffer handles
    // for sub-allocated memory, so in-place data migration is not possible
    // here (moving a live sub-allocation would require rebinding user buffers).
    // What we CAN do safely: release any fully-idle blocks back to the driver.
    for (int i = static_cast<int>(blocks_.size()) - 1; i >= 0; --i) {
        if (blocks_.size() <= 1) break;
        auto& block = blocks_[static_cast<size_t>(i)];
        if (block.used != 0) continue;
        if (block.memory) {
            if (block.hostPtr) {
                backend_->unmap(reinterpret_cast<uint64_t>(block.memory));
            }
            backend_->free(reinterpret_cast<uint64_t>(block.memory));
        }
        blocks_.erase(blocks_.begin() + i);
    }
    VVM_LOG_INFO("defragment: released idle blocks; {} block(s) remain (sub-allocation compaction requires user-side rebinding and is intentionally not performed)",
                 blocks_.size());
}

void UnifiedMemoryPoolImpl::trim() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Release empty blocks back to the driver, keeping at least one block alive.
    for (int i = static_cast<int>(blocks_.size()) - 1; i >= 0; --i) {
        if (blocks_.size() <= 1) break;
        auto& block = blocks_[static_cast<size_t>(i)];
        if (block.used != 0) continue;
        if (block.memory) {
            if (block.hostPtr) {
                backend_->unmap(reinterpret_cast<uint64_t>(block.memory));
            }
            backend_->free(reinterpret_cast<uint64_t>(block.memory));
        }
        blocks_.erase(blocks_.begin() + i);
    }
}

// ============================================================================
// Private Helpers
// ============================================================================

std::optional<Allocation> UnifiedMemoryPoolImpl::subAllocate(VkDeviceSize size,
                                                          VkDeviceSize alignment,
                                                          uint32_t blockIndex,
                                                          VkBufferUsageFlags usage) {
    auto& block = blocks_[blockIndex];
    if (!block.buddy) {
        VVM_LOG_ERROR("Block {} has no buddy allocator", blockIndex);
        return std::nullopt;
    }
    
    // Align size
    size = alignUp(size, alignment);
    
    // Allocate from buddy allocator. Honor the configured base alignment
    // (e.g. 2 MB) so buffer starts sit on driver memory-page boundaries;
    // slack is returned to the free lists by the buddy. Requests whose
    // alignment padding would overflow the block fall back to unaligned.
    VkDeviceSize baseAlign = config_.allocationAlignment;
    if (baseAlign < alignment) baseAlign = alignment;
    auto offsetOpt = block.buddy->allocateAligned(size, baseAlign);
    if (!offsetOpt) {
        offsetOpt = block.buddy->allocate(size);
    }
    if (!offsetOpt) {
        return std::nullopt;
    }
    
    VkDeviceSize offset = *offsetOpt;
    
    // Create buffer with the caller's usage flags (device address is added when
    // enabled at pool creation; buffers in a shared block are NOT exportable),
    // bound into the block at the buddy-assigned offset.
    uint64_t usageBits = static_cast<uint64_t>(usage);
    if (config_.enableDeviceAddress) {
        usageBits |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }

    int err = 0;
    const BackendBuffer bufHandle = backend_->create_buffer(
        reinterpret_cast<uint64_t>(block.memory), static_cast<uint64_t>(offset),
        static_cast<uint64_t>(size), usageBits, /*exportable=*/false, &err);
    if (bufHandle == 0) {
        int native = 0;
        const bool known = decode_backend_error(err, &native);
        VVM_LOG_ERROR("backend create_buffer failed: {}",
                      known ? vkResultToString(static_cast<VkResult>(native)).c_str()
                            : "unknown");
        block.buddy->deallocate(offset, size);
        return std::nullopt;
    }
    VkBuffer buffer = reinterpret_cast<VkBuffer>(bufHandle);

    block.used += size;

    Allocation alloc;
    alloc.buffer = buffer;
    alloc.memory = block.memory;
    alloc.offset = offset;
    alloc.size = size;
    alloc.blockIndex = blockIndex;
    alloc.isHostVisible = block.isHostVisible;
    alloc.isMapped = (block.hostPtr != nullptr);
    alloc.isCoherent = block.isCoherent;
    alloc.memoryFlags = block.memoryFlags;
    alloc.hostPtr = alloc.isHostVisible
        ? static_cast<char*>(block.hostPtr) + offset
        : nullptr;

    if (config_.enableDeviceAddress) {
        alloc.deviceAddress = backend_->buffer_device_address(bufHandle);
    }
    
    // Generation counter for handle validation
    alloc.generation = nextGeneration();
    
    return alloc;
}

void UnifiedMemoryPoolImpl::subDeallocate(Allocation&& alloc) {
    // Dedicated allocations (blockIndex == UINT32_MAX) have their own VkDeviceMemory
    // and VkBuffer - destroy both directly and remove from tracking
    if (alloc.blockIndex == UINT32_MAX) {
        // Only free if WE are still tracking it. An untracked dedicated handle
        // was already freed (double-deallocate of a stale copy) or came from a
        // foreign pool/device; destroying it here produces loader parameter
        // errors ("vkUnmapMemory: Invalid device") and worse.
        const bool tracked = std::any_of(
            dedicatedAllocations_.begin(), dedicatedAllocations_.end(),
            [&alloc](const Allocation& a) {
                return a.buffer == alloc.buffer && a.memory == alloc.memory;
            });
        if (!tracked) {
            // VVM_LOG supports only literal {} - pre-format hex.
            char buf[20];
            std::snprintf(buf, sizeof(buf), "%p", (void*)alloc.buffer);
            VVM_LOG_WARN("subDeallocate: untracked dedicated allocation "
                         "(buffer={}) - already freed or foreign, skipping", buf);
            alloc.buffer = VK_NULL_HANDLE;
            alloc.memory = VK_NULL_HANDLE;
            alloc.hostPtr = nullptr;
            return;
        }
        if (alloc.buffer) backend_->destroy_buffer(reinterpret_cast<uint64_t>(alloc.buffer));
        if (alloc.memory) {
            if (alloc.hostPtr) backend_->unmap(reinterpret_cast<uint64_t>(alloc.memory));
            backend_->free(reinterpret_cast<uint64_t>(alloc.memory));
        }
        // Remove from dedicatedAllocations_ to prevent double-free in destructor
        dedicatedAllocations_.erase(
            std::remove_if(dedicatedAllocations_.begin(), dedicatedAllocations_.end(),
                [&alloc](const Allocation& a) {
                    return a.buffer == alloc.buffer && a.memory == alloc.memory;
                }),
            dedicatedAllocations_.end());
        alloc.buffer = VK_NULL_HANDLE;
        alloc.memory = VK_NULL_HANDLE;
        alloc.hostPtr = nullptr;
        return;
    }

    if (alloc.blockIndex >= blocks_.size()) return;

    backend_->destroy_buffer(reinterpret_cast<uint64_t>(alloc.buffer));
    alloc.buffer = VK_NULL_HANDLE;

    auto& block = blocks_[alloc.blockIndex];
    if (block.buddy) {
        // Pass size=0: the buddy recorded the rounded (order) size at allocate
        // time and deallocates using its own record. Passing the original
        // requested size here would log a benign size-mismatch warning.
        block.buddy->deallocate(alloc.offset, 0);
    }
    block.used -= alloc.size;
    alloc.memory = VK_NULL_HANDLE;
    alloc.hostPtr = nullptr;
}

VkDeviceSize UnifiedMemoryPoolImpl::alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    // Validate power-of-2 alignment (required for bitwise rounding)
    if (alignment == 0) return value;
    if ((alignment & (alignment - 1)) != 0) {
        VVM_LOG_ERROR("alignUp: non-power-of-2 alignment {}; rounding up to next pow2", alignment);
        // Round alignment up to next power-of-2 to maintain correctness
        alignment--;
        alignment |= alignment >> 1;
        alignment |= alignment >> 2;
        alignment |= alignment >> 4;
        alignment |= alignment >> 8;
        alignment |= alignment >> 16;
        alignment |= alignment >> 32;
        alignment++;
    }
    return satAddU64(value, alignment - 1) & ~(alignment - 1);
}

// RAII factory: the only way to construct a UniqueAllocation. The private
// constructor pairs the allocation with uniqueAllocationDeleter so reset()
// routes back through pool->deallocate().
UniqueAllocation UniqueAllocation::make(UnifiedMemoryPool* pool, Allocation&& alloc) {
    return UniqueAllocation(pool, std::move(alloc));
}

} // namespace vvm
