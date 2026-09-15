// ============================================================================
// GGUF tensor inventory: names + exact byte sizes for every tensor in a
// (possibly split) GGUF file, feeding the auto-placement planner.
// Sizes come from section offset diffs - no quant tables needed.
// ============================================================================

#include "vulkan_vm/device_registry.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace vvm {

namespace {

uint32_t rd32(FILE* f) { uint32_t v = 0; (void)fread(&v, 4, 1, f); return v; }
uint64_t rd64(FILE* f) { uint64_t v = 0; (void)fread(&v, 8, 1, f); return v; }
std::string rdStr(FILE* f) {
    const uint64_t n = rd64(f);
    std::string s;
    if (n > 0 && n < (1ull << 40)) {
        s.resize((size_t)n, 0);
        if (fread(s.data(), 1, (size_t)n, f) != (size_t)n) s.clear();
    }
    return s;
}
#ifdef _WIN32
int64_t fpos(FILE* f) { return _ftelli64(f); }
void fseekTo(FILE* f, uint64_t off) { _fseeki64(f, (int64_t)off, SEEK_SET); }
#else
int64_t fpos(FILE* f) { return (int64_t)ftello(f); }
void fseekTo(FILE* f, uint64_t off) { fseeko(f, (off_t)off, SEEK_SET); }
#endif

// Skip one GGUF metadata value; captures general.alignment when seen.
void skipVal(FILE* f, uint32_t t, uint32_t& alignOut) {
    static const int SZ[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};  // 10 is u64
    if (t == 8) {
        const uint64_t n = rd64(f);
        if (n && n < (1ull << 40)) fseek(f, (long)n, SEEK_CUR);
        return;
    }
    if (t == 9) {
        const uint32_t et = rd32(f);
        const uint64_t cnt = rd64(f);
        if (et <= 12 && et != 8 && et != 9 && cnt < (1ull << 40)) {
            fseek(f, (long)(SZ[et] * cnt), SEEK_CUR);
            return;
        }
        for (uint64_t i = 0; i < cnt && i < (1ull << 30); ++i) skipVal(f, et, alignOut);
        return;
    }
    if (t <= 12) { fseek(f, SZ[t], SEEK_CUR); return; }
    fseek(f, 4, SEEK_CUR);
}

struct RawTensor { std::string name; uint64_t offset; };

bool parsePart(const char* path, std::vector<std::pair<std::string, uint64_t>>& out) {
    FILE* f = nullptr;
#ifdef _WIN32
    if (fopen_s(&f, path, "rb") != 0 || !f) return false;
#else
    f = fopen(path, "rb");
    if (!f) return false;
#endif
    char magic[4] = {};
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) { fclose(f); return false; }
    (void)rd32(f);
    const uint64_t nT = rd64(f), nKV = rd64(f);
    if (nT > (1ull << 24) || nKV > (1ull << 24)) { fclose(f); return false; }
    uint32_t align = 32;
    for (uint64_t i = 0; i < nKV; ++i) {
        std::string key = rdStr(f);
        if (key.empty()) { fclose(f); return false; }
        const uint32_t t = rd32(f);
        if (key == "general.alignment" && (t == 4 || t == 5)) {
            const uint32_t v = rd32(f);
            if (v > 0) align = v;
        } else {
            skipVal(f, t, align);
        }
    }
    std::vector<RawTensor> infos;
    infos.reserve((size_t)(nT > 100000 ? 100000 : nT));
    for (uint64_t i = 0; i < nT; ++i) {
        RawTensor t;
        t.name = rdStr(f);
        if (t.name.empty()) { fclose(f); return false; }
        const uint32_t nd = rd32(f);
        if (nd > 8) { fclose(f); return false; }
        fseek(f, 8L * nd, SEEK_CUR);
        (void)rd32(f);
        t.offset = rd64(f);
        infos.push_back(std::move(t));
    }
    const int64_t pos = fpos(f);
    const uint64_t dataStart = (uint64_t)((pos + (int64_t)align - 1) & ~((int64_t)align - 1));
    fseekTo(f, 0);
    fseek(f, 0, SEEK_END);
    const uint64_t fileSize = (uint64_t)fpos(f);
    fclose(f);

    std::sort(infos.begin(), infos.end(),
              [](const RawTensor& a, const RawTensor& b) { return a.offset < b.offset; });
    for (size_t i = 0; i < infos.size(); ++i) {
        const uint64_t next = (i + 1 < infos.size())
            ? infos[i + 1].offset
            : (fileSize > dataStart ? fileSize - dataStart : 0);
        if (next <= infos[i].offset) continue;
        out.emplace_back(infos[i].name, next - infos[i].offset);
    }
    return true;
}

// Split sibling paths: replace the -NNNNN-of- shard marker, else append.
std::string siblingPath(const std::string& first, int index /*1-based*/) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "-%05d-of-", index);
    std::string out = first;
    std::size_t pos = out.find("-00001-of-");
    if (pos == std::string::npos) {
        // Non-split naming: only the given file exists.
        return index == 1 ? first : std::string();
    }
    out.replace(pos, 10, buf);
    return out;
}

bool fileExists(const std::string& path) {
    FILE* f = nullptr;
#ifdef _WIN32
    if (fopen_s(&f, path.c_str(), "rb") != 0) return false;
#else
    f = fopen(path.c_str(), "rb");
    if (!f) return false;
#endif
    fclose(f);
    return true;
}

} // namespace

std::vector<TensorSpec> read_gguf_inventory(const std::string& firstPartPath) {
    std::vector<std::pair<std::string, uint64_t>> raw;
    for (int index = 1; index <= 128; ++index) {
        const std::string path = siblingPath(firstPartPath, index);
        if (path.empty() || !fileExists(path)) {
            if (index == 1) break;
            // Missing part: if we already have tensors, later parts may
            // simply not exist (fewer shards than the marker suggests).
            if (!raw.empty()) break;
            return {};
        }
        std::vector<std::pair<std::string, uint64_t>> part;
        if (!parsePart(path.c_str(), part)) break;
        raw.insert(raw.end(), part.begin(), part.end());
        if (index == 1 && raw.empty()) {
            // Single-file model whose only part has no tensors? Keep going -
            // the loop naturally ends when the next sibling is missing.
        }
        // Non-split files: siblingPath returns the same file only for
        // index 1; stop after it.
        if (siblingPath(firstPartPath, 2).empty()) break;
    }

    std::vector<TensorSpec> specs;
    specs.reserve(raw.size());
    for (const auto& [name, size] : raw) {
        TensorSpec spec{};
        size_t n = 0;
        while (n + 1 < sizeof(spec.name) && n < name.size()) {
            spec.name[n] = name[n];
            ++n;
        }
        spec.name[n] = 0;
        spec.size = size;
        int32_t layer = -1;
        spec.cls = classify_tensor(spec.name, &layer);
        spec.layer = layer;
        specs.push_back(spec);
    }
    return specs;
}

} // namespace vvm
