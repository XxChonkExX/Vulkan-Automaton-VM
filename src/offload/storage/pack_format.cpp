#include "vulkan_vm/storage/pack_format.hpp"

namespace vvm {
namespace storage {
namespace pack {

uint64_t fnv1a64(const uint8_t* p, size_t n, uint64_t seed) {
    uint64_t h = seed;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull; // FNV-1a 64-bit prime
    }
    return h;
}

uint32_t fnv1a32(const uint8_t* p, size_t n, uint32_t seed) {
    uint32_t h = seed;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 16777619u; // FNV-1a 32-bit prime
    }
    return h;
}

// Tableless CRC-32 (correct but slow — used only as a swappable alternative;
// default pack checksum is FNV-1a64).
uint32_t crc32(const uint8_t* p, size_t n) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        crc ^= p[i];
        for (int b = 0; b < 8; ++b)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

} // namespace pack
} // namespace storage
} // namespace vvm