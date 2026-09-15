#pragma once
// ============================================================================
// Unified device registry + pool set: the unification layer.
//
// enumerate_all_devices() lists every GPU visible to the process across all
// three vendor runtimes:
//   - HIP devices via amdhip64 (dynamic loader; AMD-only, implies 0x1002)
//   - Level Zero devices via ze_loader (vendor/device/name from the driver)
//   - Vulkan physical devices via a transient VkInstance (names, PCI ids,
//     DEVICE_LOCAL heap totals)
//
// best_backend_for() picks the native path when its runtime is present and
// falls back to Vulkan otherwise (the catch-all: anything with a Vulkan
// driver runs, fast or not).
//
// UnifiedPoolSet creates one UnifiedMemoryPool per in-process-creatable
// device (HIP + Level Zero today; Vulkan pools need the application's
// VkDevice, reported as info-only entries) and serves them behind one API
// with aggregate stats. This is the object the future placement policy
// (auto tensor routing) will drive.
// ============================================================================

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vulkan_vm/core.hpp"
#include "vulkan_vm/mem_backend.hpp"

namespace vvm {

// Which runtime enumerated this device.
enum class DeviceSource : int32_t { Hip = 0, Level0 = 1, Vulkan = 2 };

struct BackendDeviceInfo {
    char     name[256] = {};
    uint64_t totalMem  = 0;          // bytes (largest DEVICE_LOCAL heap / total)
    uint32_t vendorId  = 0;          // PCI vendor (0x1002 AMD, 0x8086 Intel)
    uint32_t deviceId  = 0;          // PCI device (0 when the runtime hides it)
    bool     integrated = false;     // uma/iGPU style device
    int32_t  vendorIndex = -1;       // index within its own runtime
    DeviceSource source = DeviceSource::Vulkan;
    MemBackendKind preferred = MemBackendKind::Vulkan;  // best_backend_for()
};

VVM_API std::vector<BackendDeviceInfo> enumerate_all_devices();
VVM_API MemBackendKind best_backend_for(const BackendDeviceInfo& dev);
VVM_API const char* backend_kind_name(MemBackendKind kind);

struct PooledDevice {
    BackendDeviceInfo info;
    // Null for Vulkan entries: the application's VkDevice is required to
    // create a Vulkan pool; see the llama.cpp Vulkan hook.
    std::unique_ptr<UnifiedMemoryPool> pool;
};

class VVM_API UnifiedPoolSet {
public:
    // Creates one pool per HIP device and one per Level Zero device using
    // the caller's PoolConfig (Vulkan devices are listed, not pooled).
    static std::optional<UnifiedPoolSet> create(const PoolConfig& cfg);

    UnifiedPoolSet() = default;
    UnifiedPoolSet(const UnifiedPoolSet&) = delete;
    UnifiedPoolSet& operator=(const UnifiedPoolSet&) = delete;
    UnifiedPoolSet(UnifiedPoolSet&&) noexcept = default;
    UnifiedPoolSet& operator=(UnifiedPoolSet&&) noexcept = default;

    const std::vector<PooledDevice>& devices() const { return devices_; }
    UnifiedMemoryPool* pool(MemBackendKind kind, int vendorIndex);

    // Merged JSON array: entries carry device/vendor/kind + pool stats.
    // Buffer is reused; copy it if you need to hold it.
    const char* aggregate_stats_json();

private:
    std::vector<PooledDevice> devices_;
    std::string json_;
};

// ============================================================================
// Auto tensor placement: the --n-cpu-moe replacement.
//
// Decides, from a model tensor inventory + the enumerated devices + a KV
// estimate, where every tensor class lives. Encodes the measured rules:
//   - PLE (per_layer_token_embd) ALWAYS on CPU (55.6x decode penalty on
//     GPU, measured by lukaLLM on this exact model family).
//   - Expert fill order: HIP-discrete (fast kernels, +0.22 t/s/layer
//     measured on gfx1100) -> CPU RAM cache (page cache degrades
//     gracefully, measured working at 1.3x RAM oversubscription) ->
//     L0-discrete (unmeasured kernels: overflow only) ->
//     Vulkan-discrete (slow Q3_K kernels, -0.85/layer measured: last
//     resort only).
//   - Dense/attention/head on the biggest discrete GPU (any backend;
//     dense runs fine everywhere, even Vulkan).
//   - Fill any device only to maxFraction of its heap (0.90 default:
//     the measured spill cliff sits at 92-98%).
//   - CPU is an unbounded sink (mmap page cache - streaming works).
// ============================================================================

enum class TensorClass : int32_t {
    Expert,        // routed expert FFN weight, layer-tagged
    Attention,     // attention/QSA/DeltaNet projections
    Dense,         // shared experts, norms, head, embeddings
    LookupTable,   // per_layer_token_embd (the N-gram/PLE table)
    Other,
};

struct TensorSpec {
    char     name[256] = {};
    uint64_t size = 0;
    TensorClass cls = TensorClass::Other;
    int32_t  layer = -1;     // expert layer index for Expert, else -1
};

// Where one expert layer landed. backend==Vulkan+index<0 means CPU/mmap.
struct ExpertPlacement {
    int32_t layer = -1;
    MemBackendKind backend = MemBackendKind::Vulkan;
    int32_t deviceIndex = -1;
    bool onCpu = true;
};

struct PlacementPlan {
    std::vector<ExpertPlacement> experts;  // one per expert layer found
    MemBackendKind denseBackend = MemBackendKind::Vulkan;
    int32_t denseDeviceIndex = -1;
    bool   pleOnCpu = true;
    // Human-readable one-liner (ncmoe equivalent when CPU layers are a
    // suffix, else "mixed - use the per-layer map").
    char   summary[256] = {};
};

VVM_API TensorClass classify_tensor(const char* name, int32_t* layerOut);
// Reads names + exact byte sizes for every tensor in a (possibly split)
// GGUF file. Sizes come from section offset diffs - no quant tables needed.
// Returns an empty vector when the file cannot be parsed.
VVM_API std::vector<TensorSpec> read_gguf_inventory(const std::string& firstPartPath);
VVM_API PlacementPlan auto_place_experts(
    const std::vector<BackendDeviceInfo>& devices,
    const std::vector<TensorSpec>& tensors,
    uint64_t kvCacheBytes,
    float maxFraction = 0.90f);
// Extended form: hostCacheBytes bounds the CPU mmap sink (default
// unbounded - page cache degrades gracefully). Small-RAM boxes pass
// their usable RAM here so overflow routes to L0/Vulkan instead.
VVM_API PlacementPlan auto_place_experts_ex(
    const std::vector<BackendDeviceInfo>& devices,
    const std::vector<TensorSpec>& tensors,
    uint64_t kvCacheBytes,
    uint64_t hostCacheBytes,
    float maxFraction = 0.90f);

} // namespace vvm
