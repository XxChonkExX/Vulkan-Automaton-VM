#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vvm {
namespace network {
namespace wire {

// ============================================================================
// Canonical binary wire format for the VVM TCP protocol.
// All multi-byte integers are BIG-ENDIAN (network order), lengths are u32
// prefixes, strings are u32 length + raw bytes. Every get* helper is bounds
// checked against `end` and advances the cursor only on success.
// ============================================================================

inline void putU8(std::vector<uint8_t>& v, uint8_t x) {
    v.push_back(x);
}

inline void putU32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((x >> 24) & 0xff);
    v.push_back((x >> 16) & 0xff);
    v.push_back((x >> 8) & 0xff);
    v.push_back(x & 0xff);
}

inline void putU64(std::vector<uint8_t>& v, uint64_t x) {
    for (int i = 7; i >= 0; --i) v.push_back((x >> (8 * i)) & 0xff);
}

inline void putBytes(std::vector<uint8_t>& v, const std::vector<uint8_t>& bytes) {
    putU32(v, static_cast<uint32_t>(bytes.size()));
    v.insert(v.end(), bytes.begin(), bytes.end());
}

inline void putStr(std::vector<uint8_t>& v, const std::string& s) {
    putU32(v, static_cast<uint32_t>(s.size()));
    v.insert(v.end(), s.begin(), s.end());
}

inline bool getU8(const uint8_t*& p, const uint8_t* end, uint8_t& out) {
    if (p == end) return false;
    out = *p;
    ++p;
    return true;
}

inline bool getU32(const uint8_t*& p, const uint8_t* end, uint32_t& out) {
    if (p + 4 > end) return false;
    out = (static_cast<uint32_t>(p[0]) << 24) |
          (static_cast<uint32_t>(p[1]) << 16) |
          (static_cast<uint32_t>(p[2]) << 8) |
          static_cast<uint32_t>(p[3]);
    p += 4;
    return true;
}

inline bool getU64(const uint8_t*& p, const uint8_t* end, uint64_t& out) {
    if (p + 8 > end) return false;
    out = 0;
    for (int i = 0; i < 8; ++i) out = (out << 8) | p[i];
    p += 8;
    return true;
}

inline bool getBytes(const uint8_t*& p, const uint8_t* end, std::vector<uint8_t>& out) {
    uint32_t len = 0;
    if (!getU32(p, end, len)) return false;
    if (p + len > end) return false;
    out.assign(p, p + len);
    p += len;
    return true;
}

inline bool getStr(const uint8_t*& p, const uint8_t* end, std::string& out) {
    uint32_t len = 0;
    if (!getU32(p, end, len)) return false;
    if (p + len > end) return false;
    out.assign(reinterpret_cast<const char*>(p), len);
    p += len;
    return true;
}

} // namespace wire
} // namespace network
} // namespace vvm