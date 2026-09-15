// Auto-placement validation: read the real Qwen3.8 GGUF inventory (all 3
// parts, exact byte sizes via offset diffs), run the registry + planner,
// and assert the output rediscovers the measured champion from first
// principles (dense on HIP, PLE on CPU, ~14 expert layers on GPU).
#include "vulkan_vm/device_registry.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace vvm;

namespace {

uint32_t rd32(FILE* f) { uint32_t v = 0; (void)fread(&v, 4, 1, f); return v; }
uint64_t rd64(FILE* f) { uint64_t v = 0; (void)fread(&v, 8, 1, f); return v; }
std::string rdStr(FILE* f) {
    const uint64_t n = rd64(f);
    std::string s(n, 0);
    if (n) (void)fread(s.data(), 1, (size_t)n, f);
    return s;
}
// Skip one GGUF metadata value of the given type; capture u32 alignment.
void skipVal(FILE* f, uint32_t t, uint32_t& alignOut) {
    static const int SZ[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (t == 8) { const uint64_t n = rd64(f); if (n) fseek(f, (long)n, SEEK_CUR); return; }
    if (t == 9) {
        const uint32_t et = rd32(f);
        const uint64_t cnt = rd64(f);
        if (et <= 12 && et != 8 && et != 9) { fseek(f, (long)(SZ[et] * cnt), SEEK_CUR); return; }
        for (uint64_t i = 0; i < cnt; ++i) skipVal(f, et, alignOut);
        return;
    }
    if (t <= 12) { fseek(f, SZ[t], SEEK_CUR); return; }
    fseek(f, 4, SEEK_CUR);
}

struct RawTensor { std::string name; uint64_t offset; };

// Parse ONE split part: tensor names + section-relative offsets, the data
// section file start, and the part's file size + alignment.
bool parsePart(const char* path, std::vector<RawTensor>& out,
               uint64_t& dataStart, uint64_t& fileSize, uint32_t& alignOut) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") != 0 || !f) return false;
    char magic[4] = {};
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) { fclose(f); return false; }
    (void)rd32(f);  // version
    const uint64_t nT = rd64(f), nKV = rd64(f);
    std::printf("diag: %s nT=%llu nKV=%llu\n", path,
                (unsigned long long)nT, (unsigned long long)nKV);
    alignOut = 32;
    for (uint64_t i = 0; i < nKV; ++i) {
        std::string key = rdStr(f);
        const uint32_t t = rd32(f);
        if (key == "general.alignment" && (t == 4 || t == 5)) {
            const uint32_t v = rd32(f);
            if (v > 0) alignOut = v;
        } else {
            skipVal(f, t, alignOut);
        }
    }
    for (uint64_t i = 0; i < nT; ++i) {
        RawTensor t;
        t.name = rdStr(f);
        const uint32_t nd = rd32(f);
        fseek(f, 8L * nd, SEEK_CUR);
        (void)rd32(f);  // type
        t.offset = rd64(f);
        out.push_back(std::move(t));
    }
    long long pos = _ftelli64(f);
    dataStart = (uint64_t)((pos + (long long)alignOut - 1) & ~((long long)alignOut - 1));
    _fseeki64(f, 0, SEEK_END);
    fileSize = (uint64_t)_ftelli64(f);
    fclose(f);
    return true;
}

} // namespace

int main() {
    int failures = 0;
    setvbuf(stdout, NULL, _IONBF, 0);   // unbuffered: crash sites stay visible
    try {
    const char* parts[3] = {
        "D:\\AI_Bundle\\qwen38next\\UD-Q3_K_XL\\Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
        "D:\\AI_Bundle\\qwen38next\\UD-Q3_K_XL\\Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf",
        "D:\\AI_Bundle\\qwen38next\\UD-Q3_K_XL\\Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00003.gguf",
    };

    std::vector<TensorSpec> specs;
    uint64_t totalBytes = 0;
    for (int p = 0; p < 3; ++p) {
        std::vector<RawTensor> infos;
        uint64_t dataStart = 0, fileSize = 0;
        uint32_t align = 32;
        if (!parsePart(parts[p], infos, dataStart, fileSize, align)) {
            std::printf("FAIL: cannot parse %s\n", parts[p]);
            return 1;
        }
        std::sort(infos.begin(), infos.end(),
                  [](const RawTensor& a, const RawTensor& b) { return a.offset < b.offset; });
        for (size_t i = 0; i < infos.size(); ++i) {
            const uint64_t next = (i + 1 < infos.size())
                ? infos[i + 1].offset
                : (fileSize > dataStart ? fileSize - dataStart : 0);
            const uint64_t size = next > infos[i].offset ? next - infos[i].offset : 0;
            if (size == 0) continue;
            TensorSpec spec{};
            size_t n = 0;
            while (n + 1 < sizeof(spec.name) && n < infos[i].name.size()) {
                spec.name[n] = infos[i].name[n];
                ++n;
            }
            spec.name[n] = 0;
            spec.size = size;
            int32_t layer = -1;
            spec.cls = classify_tensor(spec.name, &layer);
            spec.layer = layer;
            totalBytes += size;
            specs.push_back(spec);
        }
    }
    std::printf("inventory: %zu tensors, %.1f GiB total\n",
                specs.size(), totalBytes / 1073741824.0);

    // Class breakdown + expert layer count
    size_t nExpert = 0, nPle = 0;
    int maxLayer = -1;
    uint64_t expertBytes = 0, pleBytes = 0;
    for (const auto& t : specs) {
        if (t.cls == TensorClass::Expert) { ++nExpert; expertBytes += t.size; if (t.layer > maxLayer) maxLayer = t.layer; }
        if (t.cls == TensorClass::LookupTable) { ++nPle; pleBytes += t.size; }
    }
    std::printf("experts: %zu tensors, layers 0..%d, %.1f GiB; PLE: %zu tensors, %.1f GiB\n",
                nExpert, maxLayer, expertBytes / 1073741824.0,
                nPle, pleBytes / 1073741824.0);
    if (maxLayer != 47) { std::printf("FAIL: expected 48 expert layers, got %d\n", maxLayer + 1); ++failures; }

    // Plan on the live device set (KV estimate for c=4096: 512 MiB).
    const auto devices = enumerate_all_devices();
    const PlacementPlan plan = auto_place_experts(devices, specs, 512ull * 1024ull * 1024ull, 0.90f);
    std::printf("plan: %s | dense=%s pleOnCpu=%d\n", plan.summary,
                backend_kind_name(plan.denseBackend), (int)plan.pleOnCpu);

    // Champion-shape assertions (measured ground truth: dense on XTX/HIP,
    // PLE on CPU, ~14 of 48 expert layers on GPU, rest on CPU).
    int gpuLayers = 0;
    bool gpuIsHip = true, denseIsHip = (plan.denseBackend == MemBackendKind::Hip);
    for (const auto& e : plan.experts) {
        if (!e.onCpu) {
            ++gpuLayers;
            if (e.backend != MemBackendKind::Hip) gpuIsHip = false;
        }
    }
    std::printf("gpuLayers=%d denseIsHip=%d pleOnCpu=%d\n",
                gpuLayers, (int)denseIsHip, (int)plan.pleOnCpu);
    if (!plan.pleOnCpu) { std::printf("FAIL: PLE must be on CPU\n"); ++failures; }
    if (!denseIsHip) { std::printf("FAIL: dense must target HIP\n"); ++failures; }
    if (!gpuIsHip) { std::printf("FAIL: GPU experts must target HIP\n"); ++failures; }
    if (gpuLayers < 10 || gpuLayers > 18) {
        std::printf("FAIL: expected 10-18 GPU expert layers, got %d\n", gpuLayers);
        ++failures;
    }

    if (failures == 0) {
        std::printf("ALL AUTO-PLACE TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", failures);
    return 1;
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    } catch (...) {
        std::printf("EXCEPTION: unknown\n");
        return 2;
    }
}
