// Auto-placement validation: read the real Qwen3.8 GGUF inventory through
// the library reader, run the registry + planner, and assert the output
// rediscovers the measured champion from first principles (dense on HIP,
// PLE on CPU, ~14 expert layers on GPU).
#include "vulkan_vm/device_registry.hpp"

#include <cstdio>

using namespace vvm;

int main() {
    int failures = 0;
    try {
    // The library inventory reader (same code the loader will call).
    const std::vector<TensorSpec> specs = read_gguf_inventory(
        "D:\\AI_Bundle\\qwen38next\\UD-Q3_K_XL\\Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf");
    if (specs.empty()) {
        std::printf("FAIL: inventory empty\n");
        return 1;
    }
    uint64_t totalBytes = 0;
    for (const auto& t : specs) totalBytes += t.size;
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
