#pragma once

// Public-domain SHA-256 (no dependencies). FIPS 180-4.

#include <cstdint>
#include <string>

namespace vvm {
namespace network {

class Sha256 {
public:
    Sha256() { reset(); }

    void reset() {
        total_ = 0;
        bufused_ = 0;
        for (int i = 0; i < 8; ++i) H_[i] = iv[i];
    }

    void update(const void* data, size_t len) {
        auto* p = static_cast<const uint8_t*>(data);
        total_ += len;
        while (len) {
            size_t n = 64 - bufused_;
            if (len < n) n = len;
            for (size_t i = 0; i < n; ++i) buf_[bufused_ + i] = p[i];
            bufused_ += (unsigned)n;
            p += n;
            len -= n;
            if (bufused_ == 64) { transform(buf_); bufused_ = 0; }
        }
    }

    void finalize(uint8_t out[32]) {
        uint64_t bits = total_ * 8;
        buf_[bufused_++] = 0x80;
        if (bufused_ > 56) {
            while (bufused_ < 64) buf_[bufused_++] = 0;
            transform(buf_);
            bufused_ = 0;
        }
        while (bufused_ < 56) buf_[bufused_++] = 0;
        for (int i = 0; i < 8; ++i)
            buf_[56 + i] = (uint8_t)(bits >> (56 - 8 * i));
        transform(buf_);
        for (int i = 0; i < 8; ++i) {
            out[i * 4 + 0] = (uint8_t)(H_[i] >> 24);
            out[i * 4 + 1] = (uint8_t)(H_[i] >> 16);
            out[i * 4 + 2] = (uint8_t)(H_[i] >> 8);
            out[i * 4 + 3] = (uint8_t) H_[i];
        }
        reset();
    }

    static std::string hex(const uint8_t d[32]) {
        static const char hx[] = "0123456789abcdef";
        std::string s(64, '0');
        for (int i = 0; i < 32; ++i) {
            s[2 * i] = hx[d[i] >> 4];
            s[2 * i + 1] = hx[d[i] & 0xf];
        }
        return s;
    }

private:
    static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

    void transform(const uint8_t block[64]) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
                   ((uint32_t)block[i * 4 + 2] << 8) | block[i * 4 + 3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = H_[0], b = H_[1], c = H_[2], d = H_[3],
                 e = H_[4], f = H_[5], g = H_[6], h = H_[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = h + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        H_[0] += a;
        H_[1] += b;
        H_[2] += c;
        H_[3] += d;
        H_[4] += e;
        H_[5] += f;
        H_[6] += g;
        H_[7] += h;
    }

    uint32_t H_[8] = {};
    uint8_t buf_[64] = {};
    unsigned bufused_ = 0;
    uint64_t total_ = 0;

    static constexpr uint32_t iv[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    static constexpr uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
};

}  // namespace network
}  // namespace vvm