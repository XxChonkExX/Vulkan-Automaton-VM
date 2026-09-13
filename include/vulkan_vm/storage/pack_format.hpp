#pragma once

// L0 — sharded pack format v2 (SSD <-> GPU streaming on-disk layout).
//
// Design philosophy (matches the transport layer): "best tool for the job,
// as many tools as needed." Nothing is locked to 64-bit. The header is
// self-describing: it records the exact integer width of each shard-table
// field in `flags`, so a reader validates against its compiled traits and a
// mismatch is a clean error, never silent corruption.
//
//   [PackHeader (fixed, 32 bytes)][ShardEntry x N][pad][blob_0]...[blob_N-1]
//
// Blobs are 4 KiB-aligned (NVMe LBA family). `shardLog2` is the *nominal*
// cache-line size used by L1; it does not force file alignment.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

// DLL export macro (canonical definition, guarded so it coexists with core.hpp)
#ifndef VVM_API
#if defined(VVM_BUILD_SHARED) && defined(VVM_EXPORT)
#  if defined(_MSC_VER)
#    define VVM_API __declspec(dllexport)
#  else
#    define VVM_API __attribute__((visibility("default")))
#  endif
#elif defined(VVM_BUILD_SHARED) && defined(_MSC_VER)
#  define VVM_API __declspec(dllimport)
#else
#  define VVM_API
#endif
#endif

namespace vvm {
namespace storage {
namespace pack {

constexpr uint32_t kPackMagic   = 0x32454D56u; // 'VME2' little-endian
constexpr uint32_t kPackVersion = 2;
constexpr uint64_t kBlobAlign   = 4096;

// Integer width of a table field (encoded in PackHeader::flags).
enum class FieldWidth : uint8_t { W0 = 0, W16 = 1, W32 = 2, W64 = 3 };

// Swappable checksum tool (none / 32-bit FNV-1a / 32-bit CRC / 64-bit FNV-1a).
enum class HashId : uint32_t { None = 0, Fnv1a32 = 1, Crc32 = 2, Fnv1a64 = 3 };

// ---- width <-> type mapping -----------------------------------------------
template <typename T> struct WidthOf;
template <> struct WidthOf<uint16_t> { static constexpr FieldWidth value = FieldWidth::W16; };
template <> struct WidthOf<uint32_t> { static constexpr FieldWidth value = FieldWidth::W32; };
template <> struct WidthOf<uint64_t> { static constexpr FieldWidth value = FieldWidth::W64; };

constexpr size_t bytesFor(FieldWidth w) {
    return w == FieldWidth::W16 ? 2 : (w == FieldWidth::W32 ? 4 : (w == FieldWidth::W64 ? 8 : 0));
}

// Pack the four field widths into the flags word (2 bits each, low->high:
// offset, size, id, hash).
constexpr uint32_t packFlags(FieldWidth off, FieldWidth sz, FieldWidth id, FieldWidth h) {
    return (static_cast<uint32_t>(off))
         | (static_cast<uint32_t>(sz) << 2)
         | (static_cast<uint32_t>(id) << 4)
         | (static_cast<uint32_t>(h)  << 6);
}
constexpr FieldWidth offsetWidth(uint32_t flags) { return static_cast<FieldWidth>(flags & 0x3); }
constexpr FieldWidth sizeWidth(uint32_t flags)   { return static_cast<FieldWidth>((flags >> 2) & 0x3); }
constexpr FieldWidth idWidth(uint32_t flags)     { return static_cast<FieldWidth>((flags >> 4) & 0x3); }
constexpr FieldWidth hashWidth(uint32_t flags)   { return static_cast<FieldWidth>((flags >> 6) & 0x3); }

// ---- checksums (swappable; FNV-1a64 is default) ---------------------------
VVM_API uint64_t fnv1a64(const uint8_t* p, size_t n, uint64_t seed = 1469598103934665603ull);
VVM_API uint32_t fnv1a32(const uint8_t* p, size_t n, uint32_t seed = 2166136261u);
VVM_API uint32_t crc32(const uint8_t* p, size_t n);

// ---- fixed, always-64-bit-count header ------------------------------------
// Counts/offsets here are byte counts and absolute positions, which genuinely
// exceed 32 bits on multi-TB model packs — that is their best fit. The
// *per-shard* table (below) is where narrow widths pay off.
struct PackHeader {
    uint32_t magic = kPackMagic;
    uint32_t version = kPackVersion;
    uint32_t flags = 0;        // width selection (see packFlags)
    uint32_t shardLog2 = 20;   // nominal cache-line size = 1 << shardLog2
    uint64_t shardCount = 0;   // number of ShardEntry rows
    uint64_t dataOffset = 0;   // absolute offset of first blob
};
static_assert(sizeof(PackHeader) == 32, "PackHeader must be 32 bytes");

// ---- reader/writer traits (the "tools"; swap widths to fit the job) --------
struct PackTraits {
    using OffsetT = uint64_t;   // file offsets — best fit is 64-bit (large packs)
    using SizeT   = uint32_t;   // per-shard byte size — shards stay < 4 GiB
    using IdT     = uint32_t;   // shard/expert id — millions is enough
    using HashT   = uint64_t;   // 64-bit checksum
    static constexpr HashId hashId = HashId::Fnv1a64;
    static constexpr uint32_t flags() {
        return packFlags(WidthOf<OffsetT>::value, WidthOf<SizeT>::value,
                         WidthOf<IdT>::value, WidthOf<HashT>::value);
    }
};

struct PackTraits32 {
    using OffsetT = uint32_t;   // small packs (< 4 GiB): tighter tables
    using SizeT   = uint32_t;
    using IdT     = uint32_t;
    using HashT   = uint64_t;
    static constexpr HashId hashId = HashId::Fnv1a64;
    static constexpr uint32_t flags() {
        return packFlags(WidthOf<OffsetT>::value, WidthOf<SizeT>::value,
                         WidthOf<IdT>::value, WidthOf<HashT>::value);
    }
};

// ---- per-shard table row ---------------------------------------------------
template <class T = PackTraits>
struct ShardEntry {
    typename T::IdT     id = 0;
    typename T::OffsetT offset = 0; // absolute file offset of the blob
    typename T::SizeT   size = 0;   // blob byte size
    typename T::HashT   hash = 0;   // checksum over the blob bytes

    static constexpr size_t serialized() {
        return sizeof(typename T::IdT) + sizeof(typename T::OffsetT)
             + sizeof(typename T::SizeT) + sizeof(typename T::HashT);
    }
};

// ---- little-endian field (de)serialization helpers -------------------------
template <class U>
void putField(std::vector<uint8_t>& b, U v) {
    for (size_t i = 0; i < sizeof(U); ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
}
template <class U>
bool getField(const uint8_t*& p, const uint8_t* end, U& out) {
    if (static_cast<size_t>(end - p) < sizeof(U)) return false;
    out = 0;
    for (size_t i = 0; i < sizeof(U); ++i) out |= static_cast<U>(p[i]) << (8 * i);
    p += sizeof(U);
    return true;
}

inline uint64_t alignUp(uint64_t v, uint64_t a) { return a ? (v + a - 1) & ~(a - 1) : v; }

// ---- reader ----------------------------------------------------------------
template <class T = PackTraits>
class PackReader {
public:
    // Opens and validates magic, version, and that the file's field widths
    // match T. Returns false on any mismatch / truncation.
    bool open(const std::string& path) {
        path_ = path;
        table_.clear();

        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        PackHeader h;
        f.read(reinterpret_cast<char*>(&h), sizeof(h));
        if (!f || h.magic != kPackMagic || h.version != kPackVersion) return false;
        if (!validateFlags(h.flags)) return false;
        hdr_ = h;

        const size_t tableBytes = static_cast<size_t>(h.shardCount) * ShardEntry<T>::serialized();
        std::vector<uint8_t> raw(tableBytes);
        if (tableBytes && !(bool)f.read(reinterpret_cast<char*>(raw.data()), tableBytes)) return false;

        const uint8_t* p = raw.data();
        const uint8_t* end = raw.data() + raw.size();
        table_.reserve(static_cast<size_t>(h.shardCount));
        for (uint64_t i = 0; i < h.shardCount; ++i) {
            ShardEntry<T> e;
            if (!getField(p, end, e.id) || !getField(p, end, e.offset)
                || !getField(p, end, e.size) || !readHash(p, end, e.hash)) return false;
            table_.push_back(e);
        }
        return true;
    }

    const PackHeader& header() const { return hdr_; }
    const std::vector<ShardEntry<T>>& table() const { return table_; }

    // Read a blob into `out` (resized). Returns false on short read.
    bool readShard(const ShardEntry<T>& e, std::vector<uint8_t>& out) const {
        out.resize(static_cast<size_t>(e.size));
        if (out.empty()) return true;
        std::ifstream f(path_, std::ios::binary);
        if (!f) return false;
        f.seekg(static_cast<std::streamoff>(e.offset));
        if (!f) return false;
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
        return static_cast<bool>(f);
    }

    // Re-hash a blob and compare to the stored checksum.
    bool verifyShard(const ShardEntry<T>& e) const {
        std::vector<uint8_t> buf;
        if (!readShard(e, buf)) return false;
        return hash(buf.data(), buf.size()) == e.hash;
    }

    static typename T::HashT hash(const uint8_t* p, size_t n) {
        switch (T::hashId) {
            case HashId::None:      return static_cast<typename T::HashT>(0);
            case HashId::Fnv1a32:   return static_cast<typename T::HashT>(fnv1a32(p, n));
            case HashId::Crc32:     return static_cast<typename T::HashT>(crc32(p, n));
            case HashId::Fnv1a64:   return static_cast<typename T::HashT>(fnv1a64(p, n));
        }
        return 0;
    }

private:
    static bool validateFlags(uint32_t flags) {
        if (offsetWidth(flags) != WidthOf<typename T::OffsetT>::value) return false;
        if (sizeWidth(flags)   != WidthOf<typename T::SizeT>::value)   return false;
        if (idWidth(flags)     != WidthOf<typename T::IdT>::value)     return false;
        const FieldWidth hw = hashWidth(flags);
        const FieldWidth want = (T::hashId == HashId::None) ? FieldWidth::W0
                              : (T::hashId == HashId::Fnv1a64) ? FieldWidth::W64
                              : FieldWidth::W32;
        return hw == want;
    }
    bool readHash(const uint8_t*& p, const uint8_t* end, typename T::HashT& h) {
        h = 0;
        return hashWidth(hdr_.flags) == FieldWidth::W0 || getField(p, end, h);
    }

    PackHeader hdr_{};
    std::vector<ShardEntry<T>> table_;
    std::string path_;
};

// ---- writer ----------------------------------------------------------------
template <class T = PackTraits>
class PackWriter {
public:
    explicit PackWriter(uint32_t shardLog2 = 20) : shardLog2_(shardLog2) {}

    void addShard(typename T::IdT id, const uint8_t* data, size_t n) {
        blobs_.emplace_back(data, data + n);
        blobsOffsets_.push_back(0); // finalized in write()
        (void)id;
        ids_.push_back(id);
    }

    // Serialize header + table + aligned blobs. Returns false on IO failure.
    bool write(const std::string& path) {
        const uint64_t tableBytes = static_cast<uint64_t>(ids_.size()) * ShardEntry<T>::serialized();
        const uint64_t dataOffset = alignUp(sizeof(PackHeader) + tableBytes, kBlobAlign);

        PackHeader h;
        h.flags = T::flags();
        h.shardLog2 = shardLog2_;
        h.shardCount = static_cast<uint64_t>(ids_.size());
        h.dataOffset = dataOffset;

        // layout blobs
        uint64_t cursor = dataOffset;
        for (size_t i = 0; i < blobs_.size(); ++i) {
            blobsOffsets_[i] = cursor;
            cursor = alignUp(cursor + blobs_[i].size(), kBlobAlign);
        }

        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(&h), sizeof(h));

        // shard table (packed LE; field order id, offset, size, hash)
        for (size_t i = 0; i < ids_.size(); ++i) {
            ShardEntry<T> e;
            e.id = ids_[i];
            e.offset = static_cast<typename T::OffsetT>(blobsOffsets_[i]);
            e.size = static_cast<typename T::SizeT>(blobs_[i].size());
            e.hash = PackReader<T>::hash(blobs_[i].data(), blobs_[i].size());

            std::vector<uint8_t> row;
            putField(row, e.id);
            putField(row, e.offset);
            putField(row, e.size);
            if (T::hashId != HashId::None) putField(row, e.hash);
            f.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
        }

        // pad to first blob
        const uint64_t afterTable = sizeof(h) + tableBytes;
        if (dataOffset > afterTable) {
            std::vector<uint8_t> pad(static_cast<size_t>(dataOffset - afterTable), 0);
            f.write(reinterpret_cast<const char*>(pad.data()), static_cast<std::streamsize>(pad.size()));
        }

        // blobs + inter-blob alignment
        for (size_t i = 0; i < blobs_.size(); ++i) {
            f.write(reinterpret_cast<const char*>(blobs_[i].data()),
                    static_cast<std::streamsize>(blobs_[i].size()));
            const uint64_t endc = blobsOffsets_[i] + blobs_[i].size();
            const uint64_t aligned = alignUp(endc, kBlobAlign);
            if (aligned > endc) {
                std::vector<uint8_t> pad(static_cast<size_t>(aligned - endc), 0);
                f.write(reinterpret_cast<const char*>(pad.data()), static_cast<std::streamsize>(pad.size()));
            }
        }
        return static_cast<bool>(f);
    }

private:
    std::vector<std::vector<uint8_t>> blobs_;
    std::vector<uint64_t> blobsOffsets_;
    std::vector<typename T::IdT> ids_;
    uint32_t shardLog2_;
};

} // namespace pack
} // namespace storage
} // namespace vvm