// Path A: pure-Vulkan SSD -> Chonk Buffer streamer for MoE experts.
// Portable today (Windows + Linux). Win32 uses overlapped-friendly sequential
// reads into a host-visible staging Allocation, then pool->copyBuffer into a
// device-local expert Allocation. Throughput note: keep blobs >=1MB; 64KB
// random reads cap at tens of MB/s on the same drive that does GB/s batched.

#include "vulkan_vm/storage_stream.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace vvm::storage {

uint64_t alignUp(uint64_t v, uint64_t align) {
    if (align == 0) return v;
    return (v + align - 1) & ~(align - 1);
}

bool writePackFile(const std::string& path,
                   const std::vector<std::pair<uint32_t, std::vector<uint8_t>>>& experts,
                   std::vector<ExpertSpec>* outSpecs) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;

    PackHeader hdr;
    hdr.expertCount = static_cast<uint32_t>(experts.size());
    const uint64_t tableBytes = sizeof(PackHeader) + sizeof(ExpertEntry) * experts.size();
    uint64_t cursor = alignUp(tableBytes, kPackAlign);

    std::vector<ExpertEntry> table;
    table.reserve(experts.size());
    for (const auto& [id, blob] : experts) {
        ExpertEntry e;
        e.expertId = id;
        e.fileOffset = cursor;
        e.bytes = blob.size();
        table.push_back(e);
        cursor = alignUp(cursor + blob.size(), kPackAlign);
    }

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(table.data()),
            static_cast<std::streamsize>(sizeof(ExpertEntry) * table.size()));

    // Pad to first blob. NOTE: `cursor` after the table loop holds
    // END-OF-ALL-BLOBS (it advanced past every entry during offset
    // assignment); the pad must reach table[0].fileOffset, not cursor.
    const uint64_t afterTable = sizeof(hdr) + sizeof(ExpertEntry) * table.size();
    const uint64_t firstBlob =
        table.empty() ? afterTable : table.front().fileOffset;
    std::vector<char> zeros;
    if (firstBlob > afterTable) {
        zeros.assign(static_cast<size_t>(firstBlob - afterTable), 0);
        f.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    }

    for (size_t i = 0; i < experts.size(); ++i) {
        const auto& blob = experts[i].second;
        if (!blob.empty())
            f.write(reinterpret_cast<const char*>(blob.data()),
                    static_cast<std::streamsize>(blob.size()));
        const uint64_t end = table[i].fileOffset + table[i].bytes;
        const uint64_t aligned = alignUp(end, kPackAlign);
        if (aligned > end) {
            zeros.assign(static_cast<size_t>(aligned - end), 0);
            f.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
        }
    }

    if (outSpecs) {
        outSpecs->clear();
        for (const auto& e : table)
            outSpecs->push_back({e.expertId, e.fileOffset, e.bytes});
    }
    return static_cast<bool>(f);
}

std::optional<std::vector<ExpertSpec>> readPackTable(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;
    PackHeader hdr;
    f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!f || hdr.magic != kPackMagic || hdr.version != kPackVersion) return std::nullopt;
    std::vector<ExpertEntry> table(hdr.expertCount);
    f.read(reinterpret_cast<char*>(table.data()),
           static_cast<std::streamsize>(sizeof(ExpertEntry) * table.size()));
    if (!f) return std::nullopt;
    std::vector<ExpertSpec> out;
    out.reserve(table.size());
    for (const auto& e : table) out.push_back({e.expertId, e.fileOffset, e.bytes});
    return out;
}

std::optional<std::vector<uint8_t>> readPackBlob(const std::string& path, const ExpertSpec& spec) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;
    std::vector<uint8_t> out(static_cast<size_t>(spec.bytes));
    f.seekg(static_cast<std::streamoff>(spec.fileOffset));
    if (!f) return std::nullopt;
    if (!out.empty()) {
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
        if (!f) return std::nullopt;
    }
    return out;
}

// ---------------------------------------------------------------- ExpertRegistry

ExpertRegistry::ExpertRegistry(uint32_t maxResident) : maxResident_(maxResident == 0 ? 1 : maxResident) {}

void ExpertRegistry::registerPack(std::vector<ExpertSpec> specs) {
    specs_.clear();
    lru_.clear();
    pos_.clear();
    for (auto& s : specs) specs_[s.expertId] = s;
}

std::optional<ExpertSpec> ExpertRegistry::specFor(uint32_t expertId) const {
    auto it = specs_.find(expertId);
    if (it == specs_.end()) return std::nullopt;
    return it->second;
}

std::optional<uint32_t> ExpertRegistry::touch(uint32_t expertId) {
    auto pit = pos_.find(expertId);
    if (pit != pos_.end()) {
        lru_.erase(pit->second);
        lru_.push_front(expertId);
        pit->second = lru_.begin();
        return std::nullopt;
    }
    lru_.push_front(expertId);
    pos_[expertId] = lru_.begin();
    if (lru_.size() > maxResident_) {
        uint32_t victim = lru_.back();
        lru_.pop_back();
        pos_.erase(victim);
        return victim;
    }
    return std::nullopt;
}

bool ExpertRegistry::isResident(uint32_t expertId) const {
    return pos_.find(expertId) != pos_.end();
}

void ExpertRegistry::evict(uint32_t expertId) {
    auto it = pos_.find(expertId);
    if (it == pos_.end()) return;
    lru_.erase(it->second);
    pos_.erase(it);
}

std::vector<uint32_t> ExpertRegistry::planPrefetch(const std::vector<uint32_t>& want) const {
    // Skip resident, largest-first for NVMe throughput.
    std::vector<std::pair<uint64_t, uint32_t>> scored;
    for (uint32_t id : want) {
        if (isResident(id)) continue;
        auto s = specFor(id);
        if (!s) continue;
        scored.emplace_back(s->bytes, id);
    }
    std::sort(scored.begin(), scored.end(), std::greater<>());
    std::vector<uint32_t> out;
    out.reserve(scored.size());
    for (auto& [b, id] : scored) out.push_back(id);
    return out;
}

size_t ExpertRegistry::residentCount() const { return lru_.size(); }

std::vector<uint32_t> ExpertRegistry::residentIds() const {
    return {lru_.begin(), lru_.end()};
}

// ---------------------------------------------------------------- Caps

StreamCaps queryStreamCaps(VkPhysicalDevice physicalDevice) {
    StreamCaps c;
    c.pureAvailable = true;
    c.directStorageAvailable = dstorageRuntimeAvailable();
    if (physicalDevice != VK_NULL_HANDLE) {
        // Probe VK_NV/EXT_memory_decompression for future GDeflate path.
        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> exts(n);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &n, exts.data());
        for (auto& e : exts) {
            if (!std::strcmp(e.extensionName, "VK_NV_memory_decompression") ||
                !std::strcmp(e.extensionName, "VK_EXT_memory_decompression")) {
                c.gdeflateAvailable = true;
                break;
            }
        }
    }
    c.notes = "Pure path always available. DirectStorage needs Win11+NVMe+SDK (VVM_HAS_DSTORAGE).";
    return c;
}

// ---------------------------------------------------------------- ExpertStreamer (Path A)

ExpertStreamer::ExpertStreamer(UnifiedMemoryPool* pool, StorageStreamConfig config)
    : pool_(pool), config_(std::move(config)), registry_(config_.maxResidentExperts) {}

ExpertStreamer::~ExpertStreamer() { close(); }

Result ExpertStreamer::open() {
    if (!pool_) return Result::error(ErrorCode::InvalidConfig, "null pool (R2: embedder owns device+pool)");
    std::lock_guard<std::mutex> lock(mutex_);
    if (open_) return Result::success();

    auto table = readPackTable(config_.packPath);
    if (!table) return Result::error(ErrorCode::InvalidConfig, "bad pack: " + config_.packPath);
    specs_ = *table;
    registry_.registerPack(specs_);

    // Staging: host-visible ring reused for every expert load.
    AllocDesc sd;
    sd.size = config_.stagingBytes == 0 ? 64ull * 1024 * 1024 : config_.stagingBytes;
    sd.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    sd.memoryUsage = MemoryUsage::CpuToGpu;
    sd.name = "moe-staging";
    auto st = pool_->allocate(sd);
    if (!st) return Result::error(ErrorCode::AllocationFailed, "staging allocate failed");
    staging_ = std::move(*st);
    stagingCursor_ = 0;

    activeBackend_ = StreamBackend::PureVulkan;
    if (config_.backend == StreamBackend::DirectStorage && config_.allowDirectStorage &&
        dstorageRuntimeAvailable()) {
        activeBackend_ = StreamBackend::DirectStorage;
    }
    open_ = true;
    return Result::success();
}

void ExpertStreamer::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!open_) return;
    for (auto& [id, alloc] : resident_) {
        if (alloc.buffer != VK_NULL_HANDLE) pool_->deallocate(std::move(alloc));
    }
    resident_.clear();
    if (staging_ && staging_->buffer != VK_NULL_HANDLE) pool_->deallocate(std::move(*staging_));
    staging_.reset();
    open_ = false;
}

bool ExpertStreamer::isOpen() const { return open_; }

Result ExpertStreamer::prefetch(const std::vector<uint32_t>& expertIds) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!open_) return Result::error(ErrorCode::InvalidConfig, "not open");
    for (uint32_t id : registry_.planPrefetch(expertIds)) {
        auto s = registry_.specFor(id);
        if (!s) continue;
        if (resident_.find(id) != resident_.end()) {
            stats_.prefetchHits++;
            continue;
        }
        stats_.prefetchMisses++;
        if (auto r = loadLocked(*s); !r) return r;
    }
    return Result::success();
}

ResultT<Allocation*> ExpertStreamer::ensureResident(uint32_t expertId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!open_) return ResultT<Allocation*>::error(ErrorCode::InvalidConfig, "not open");
    auto it = resident_.find(expertId);
    if (it != resident_.end()) {
        registry_.touch(expertId);
        stats_.prefetchHits++;
        return ResultT<Allocation*>::success(&it->second);
    }
    auto s = registry_.specFor(expertId);
    if (!s) return ResultT<Allocation*>::error(ErrorCode::InvalidConfig, "unknown expert");
    stats_.prefetchMisses++;
    if (auto r = loadLocked(*s); !r)
        return ResultT<Allocation*>::error(r.code, r.message, r.vkResult);
    auto it2 = resident_.find(expertId);
    if (it2 == resident_.end())
        return ResultT<Allocation*>::error(ErrorCode::AllocationFailed, "load raced");
    return ResultT<Allocation*>::success(&it2->second);
}

void ExpertStreamer::evict(uint32_t expertId) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = resident_.find(expertId);
    if (it == resident_.end()) return;
    if (it->second.buffer != VK_NULL_HANDLE) pool_->deallocate(std::move(it->second));
    resident_.erase(it);
    registry_.evict(expertId);
    stats_.evictions++;
}

void ExpertStreamer::evictAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [id, alloc] : resident_) {
        if (alloc.buffer != VK_NULL_HANDLE) pool_->deallocate(std::move(alloc));
        registry_.evict(id);
        stats_.evictions++;
    }
    resident_.clear();
}

ExpertStreamer::Stats ExpertStreamer::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

Result ExpertStreamer::loadLocked(const ExpertSpec& spec) {
    // LRU victim first so VRAM stays bounded (cache/actives untouched).
    if (auto victim = registry_.touch(spec.expertId)) {
        auto it = resident_.find(*victim);
        if (it != resident_.end()) {
            if (it->second.buffer != VK_NULL_HANDLE) pool_->deallocate(std::move(it->second));
            resident_.erase(it);
            stats_.evictions++;
        }
    }

    // 1) file -> staging hostPtr (chunked; staging ring reuse).
    if (!staging_ || !staging_->hostPtr)
        return Result::error(ErrorCode::MigrationFailed, "staging not host-visible");
    if (spec.bytes > staging_->size)
        return Result::error(ErrorCode::OutOfMemory, "expert larger than staging; bump stagingBytes");

    std::ifstream f(config_.packPath, std::ios::binary);
    if (!f) return Result::error(ErrorCode::InvalidConfig, "pack reopen failed");
    f.seekg(static_cast<std::streamoff>(spec.fileOffset));
    if (!f) return Result::error(ErrorCode::MigrationFailed, "seek failed");

    // Wrap cursor if the expert would overrun the ring.
    if (stagingCursor_ + spec.bytes > staging_->size) stagingCursor_ = 0;
    char* dst = static_cast<char*>(staging_->hostPtr) + stagingCursor_;
    uint64_t left = spec.bytes;
    // Read in <=8MB slices to keep page-cache behavior smooth.
    constexpr uint64_t kSlice = 8ull * 1024 * 1024;
    while (left > 0) {
        uint64_t n = left < kSlice ? left : kSlice;
        f.read(dst, static_cast<std::streamsize>(n));
        if (!f) return Result::error(ErrorCode::MigrationFailed, "pack read short");
        dst += n;
        left -= n;
    }
    VkDeviceSize stagingOff = stagingCursor_;
    stagingCursor_ += spec.bytes;

    // 2) staging -> device-local expert Allocation (Chonk pool owns it, R1).
    auto expert = pool_->allocateTensor(spec.bytes == 0 ? 4096 : spec.bytes);
    if (!expert) return Result::error(ErrorCode::AllocationFailed, "expert allocate failed");
    if (!pool_->copyBuffer(*staging_, *expert, stagingOff, 0, spec.bytes)) {
        pool_->deallocate(std::move(*expert));
        return Result::error(ErrorCode::MigrationFailed, "staging->device copy failed");
    }

    resident_[spec.expertId] = std::move(*expert);
    stats_.bytesStreamed += spec.bytes;
    return Result::success();
}

} // namespace vvm::storage
