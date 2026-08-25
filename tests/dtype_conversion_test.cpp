// Exhaustive tests for dtype conversion primitives (FP8 E4M3/E5M2, INT4).
// Catches edge cases in readElem/writeElem/bytesForElements.
// Pure CPU - no Vulkan device required.

#include "vulkan_vm/tensor_transport.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace vvm::tensor;

static int failures = 0;
static int checks = 0;

static void expect(bool ok, const char* what) {
    checks++;
    if (!ok) {
        failures++;
        std::printf("FAIL: %s\n", what);
    }
}

// FP8 E4M3: 1 sign, 4 exp (bias=3), 3 mantissa
// Max finite = 2^(14-3) * (1 + 7/8) = 2048 * 1.875 = 3840.0 (0x7D = 0_1110_111)
// 0xFE = 0_1111_110 = +inf, 0xFF = 0_1111_111 = NaN
static void testFp8E4M3Exhaustive() {
    for (uint16_t raw = 0; raw < 256; ++raw) {
        uint8_t buf[1] = {static_cast<uint8_t>(raw)};
        double val = readElem(DataType::Float8_E4M3, buf, 0);

        // Verify special values
        if (raw == 0x00) {
            expect(val == 0.0, "FP8_E4M3 +0.0");
        } else if (raw == 0x80) {
            expect(val == -0.0, "FP8_E4M3 -0.0");
        } else if (raw == 0x77) {
            // exp=14, man=7: 2^(14-3) * (1+7/8) = 2048 * 1.875 = 3840
            expect(val == 3840.0, "FP8_E4M3 max finite = 3840.0");
        } else if (raw == 0x78) {
            // exp=15, man=0: +inf
            expect(std::isinf(val) && val > 0, "FP8_E4M3 +inf");
        } else if (raw == 0x79) {
            // exp=15, man=1: NaN
            expect(std::isnan(val), "FP8_E4M3 NaN");
        }

        // Verify subnormal values (exp=0, man!=0)
        if ((raw & 0x78) == 0 && (raw & 0x07) != 0) {
            bool negative = (raw & 0x80) != 0;
            double expected = static_cast<double>(raw & 0x07) * 0.03125;
            if (negative) expected = -expected;
            if (std::fabs(val - expected) >= 1e-10) {
                std::printf("  E4M3 subnormal raw=0x%02x: got %g, expected %g\n", raw, val, expected);
            }
            expect(std::fabs(val - expected) < 1e-10, "FP8_E4M3 subnormal value");
        }
    }
}

// FP8 E5M2: 1 sign, 5 exp (bias=15), 2 mantissa
// Max finite = 2^(30-15) * (1 + 3/4) = 32768 * 1.75 = 57344.0 (0x7B = 0_11110_11)
// 0xFF = 1_11111_11 = NaN
static void testFp8E5M2Exhaustive() {
    for (uint16_t raw = 0; raw < 256; ++raw) {
        uint8_t buf[1] = {static_cast<uint8_t>(raw)};
        double val = readElem(DataType::Float8_E5M2, buf, 0);

        if (raw == 0x00) {
            expect(val == 0.0, "FP8_E5M2 +0.0");
        } else if (raw == 0x80) {
            expect(val == -0.0, "FP8_E5M2 -0.0");
        } else if (raw == 0x7B) {
            // exp=30, man=3: 2^(30-15) * (1+3/4) = 32768 * 1.75 = 57344
            expect(val == 57344.0, "FP8_E5M2 max finite = 57344.0");
        } else if (raw == 0x7C) {
            // exp=31, man=0: +inf
            expect(std::isinf(val) && val > 0, "FP8_E5M2 +inf");
        } else if (raw == 0x7D) {
            // exp=31, man=1: NaN
            expect(std::isnan(val), "FP8_E5M2 NaN");
        }

        // Verify subnormal values (exp=0, man!=0)
        if ((raw & 0x7C) == 0 && (raw & 0x03) != 0) {
            bool negative = (raw & 0x80) != 0;
            double expected = static_cast<double>(raw & 0x03) * 0.00006103515625;
            if (negative) expected = -expected;
            if (std::fabs(val - expected) >= 1e-15) {
                std::printf("  E5M2 subnormal raw=0x%02x: got %.18g, expected %.18g\n", raw, val, expected);
            }
            expect(std::fabs(val - expected) < 1e-15, "FP8_E5M2 subnormal value");
        }
    }
}

// INT4: packed nibbles, values -8 to 7
static void testInt4Packed() {
    // Test each nibble independently to avoid cross-contamination
    uint8_t buf[1] = {0};

    // Test low nibble (index 0)
    writeElem(DataType::Int4, buf, 0, 3.0);
    expect(readElem(DataType::Int4, buf, 0) == 3.0, "INT4 low nibble = 3");

    // Test high nibble (index 1)
    buf[0] = 0;
    writeElem(DataType::Int4, buf, 1, 5.0);
    expect(readElem(DataType::Int4, buf, 1) == 5.0, "INT4 high nibble = 5");

    // Test negative values in low nibble
    buf[0] = 0;
    writeElem(DataType::Int4, buf, 0, -1.0);
    expect(readElem(DataType::Int4, buf, 0) == -1.0, "INT4 low nibble -1");

    // Test -8 in high nibble
    buf[0] = 0;
    writeElem(DataType::Int4, buf, 1, -8.0);
    expect(readElem(DataType::Int4, buf, 1) == -8.0, "INT4 high nibble -8");

    // Test boundary: max = 7
    buf[0] = 0;
    writeElem(DataType::Int4, buf, 0, 7.0);
    expect(readElem(DataType::Int4, buf, 0) == 7.0, "INT4 max = 7");

    // Test boundary: min = -8
    buf[0] = 0;
    writeElem(DataType::Int4, buf, 0, -8.0);
    expect(readElem(DataType::Int4, buf, 0) == -8.0, "INT4 min = -8");

    // Test clamping: 10 should clamp to 7
    buf[0] = 0;
    writeElem(DataType::Int4, buf, 0, 10.0);
    expect(readElem(DataType::Int4, buf, 0) == 7.0, "INT4 clamp 10 -> 7");

    // Test clamping: -10 should clamp to -8
    buf[0] = 0;
    writeElem(DataType::Int4, buf, 0, -10.0);
    expect(readElem(DataType::Int4, buf, 0) == -8.0, "INT4 clamp -10 -> -8");

    // Test bytesForElements for packed format
    expect(bytesForElements(DataType::Int4, 0) == 0, "INT4 0 elements = 0 bytes");
    expect(bytesForElements(DataType::Int4, 1) == 1, "INT4 1 element = 1 byte");
    expect(bytesForElements(DataType::Int4, 2) == 1, "INT4 2 elements = 1 byte");
    expect(bytesForElements(DataType::Int4, 3) == 2, "INT4 3 elements = 2 bytes");
    expect(bytesForElements(DataType::Int4, 4) == 2, "INT4 4 elements = 2 bytes");
    expect(bytesForElements(DataType::Int4, 5) == 3, "INT4 5 elements = 3 bytes");
}

// Test bytesForElements for all types
static void testBytesForElements() {
    expect(bytesForElements(DataType::Float32, 10) == 40, "Float32 10 elements = 40 bytes");
    expect(bytesForElements(DataType::Float16, 10) == 20, "Float16 10 elements = 20 bytes");
    expect(bytesForElements(DataType::Int8, 10) == 10, "Int8 10 elements = 10 bytes");
    expect(bytesForElements(DataType::Int32, 10) == 40, "Int32 10 elements = 40 bytes");
    expect(bytesForElements(DataType::Int64, 10) == 80, "Int64 10 elements = 80 bytes");
    expect(bytesForElements(DataType::Float8_E4M3, 10) == 10, "FP8_E4M3 10 elements = 10 bytes");
    expect(bytesForElements(DataType::Float8_E5M2, 10) == 10, "FP8_E5M2 10 elements = 10 bytes");
    expect(bytesForElements(DataType::Bool, 10) == 10, "Bool 10 elements = 10 bytes");
}

// Test TensorShape overflow safety
static void testShapeOverflow() {
    // Valid shape
    TensorShape s1 = TensorShape::makeContiguous({2, 3, 4});
    expect(s1.numel() == 24, "2x3x4 = 24 elements");
    expect(s1.isValid(), "2x3x4 is valid");

    // Empty shape (scalar)
    TensorShape s2 = TensorShape::makeContiguous({});
    expect(s2.numel() == 1, "scalar numel = 1");
    expect(s2.isValid(), "scalar is valid");

    // Negative dimension
    TensorShape s3 = TensorShape::makeContiguous({-1, 3, 4});
    expect(s3.numel() == 0, "negative dim gives 0 numel");
    expect(!s3.isValid(), "negative dim is invalid");

    // Zero dimension
    TensorShape s4 = TensorShape::makeContiguous({2, 0, 4});
    expect(s4.numel() == 0, "zero dim gives 0 numel");
    expect(!s4.isValid(), "zero dim is invalid");
}

// Test makeChannelsLast for rank-4
static void testChannelsLast() {
    TensorShape s = TensorShape::makeChannelsLast({2, 3, 4, 5});
    expect(s.isChannelsLast(), "rank-4 channels-last is valid");
    expect(s.numel() == 120, "2x3x4x5 = 120");

    // Non-4D should not be channels-last
    TensorShape s2 = TensorShape::makeChannelsLast({2, 3, 4});
    expect(!s2.isChannelsLast(), "rank-3 is not channels-last");
}

// Test makeBlocked
static void testBlocked() {
    TensorShape s = TensorShape::makeBlocked({10, 20}, 8);
    expect(s.numel() > 0, "blocked shape has elements");
    expect(s.strides.size() == 2, "blocked shape has strides");
    // With blockSize=8: dim0 padded to 16, dim1 padded to 24
    // strides should be [padded_dim1, 1] = [24, 1]
    expect(s.strides[1] == 1, "blocked inner stride = 1");
}

int main() {
    std::printf("=== Dtype Conversion Tests ===\n");

    testFp8E4M3Exhaustive();
    testFp8E5M2Exhaustive();
    testInt4Packed();
    testBytesForElements();
    testShapeOverflow();
    testChannelsLast();
    testBlocked();

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures > 0 ? 1 : 0;
}

