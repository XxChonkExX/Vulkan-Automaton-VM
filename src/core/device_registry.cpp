// ============================================================================
// Unified device registry: enumerate HIP + Level Zero + Vulkan devices in
// one process, pick the best backend per device, and create pools for the
// backends that own their devices (HIP, Level Zero).
// ============================================================================

#include "vulkan_vm/device_registry.hpp"
#include "vulkan_vm/hip_mem_backend.hpp"
#include "vulkan_vm/l0_mem_backend.hpp"
#include "vulkan_vm/cuda_mem_backend.hpp"
#include "vulkan_vm/utils.hpp"

#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstring>
#include <cstdio>

namespace vvm {

const char* backend_kind_name(MemBackendKind kind) {
    switch (kind) {
        case MemBackendKind::Vulkan: return "vulkan";
        case MemBackendKind::Hip:    return "hip";
        case MemBackendKind::Level0: return "level0";
        case MemBackendKind::Cuda:   return "cuda";
        case MemBackendKind::Auto:   return "auto";
    }
    return "unknown";
}

MemBackendKind best_backend_for(const BackendDeviceInfo& dev) {
    switch (dev.source) {
        case DeviceSource::Hip:
            // HIP-implied AMD discrete gets the native path; integrated AMD
            // (APU/iGPU) stays on Vulkan where the UMA memory model fits.
            return dev.integrated ? MemBackendKind::Vulkan : MemBackendKind::Hip;
        case DeviceSource::Level0:
            return dev.integrated ? MemBackendKind::Vulkan : MemBackendKind::Level0;
        case DeviceSource::Cuda:
            // NVIDIA discrete gets the native CUDA path (mature kernels);
            // Tegra-style integrated stays on Vulkan (UMA fits).
            return dev.integrated ? MemBackendKind::Vulkan : MemBackendKind::Cuda;
        case DeviceSource::Vulkan:
        default:
            return MemBackendKind::Vulkan;
    }
}

static void set_name(char (&dst)[256], const char* src) {
    if (!src) { dst[0] = 0; return; }
    size_t n = 0;
    while (n + 1 < sizeof(dst) && src[n] != 0) { dst[n] = src[n]; ++n; }
    dst[n] = 0;
}

std::vector<BackendDeviceInfo> enumerate_all_devices() {
    std::vector<BackendDeviceInfo> out;

    // ---- HIP (AMD; vendor implied 0x1002) ----
    if (hip_runtime_present()) {
        const int n = hip_enumerate_count();
        for (int i = 0; i < n; ++i) {
            char name[256] = {};
            uint64_t total = 0;
            bool integrated = false;
            if (!hip_enumerate_device(i, name, sizeof(name), &total, &integrated)) {
                continue;
            }
            BackendDeviceInfo d{};
            set_name(d.name, name);
            d.totalMem = total;
            d.vendorId = 0x1002;
            d.integrated = integrated;
            d.vendorIndex = i;
            d.source = DeviceSource::Hip;
            d.preferred = best_backend_for(d);
            out.push_back(d);
        }
    }

    // ---- Level Zero (Intel and friends; ids from the driver) ----
    if (l0_runtime_present()) {
        const int n = l0_enumerate_count();
        for (int i = 0; i < n; ++i) {
            char name[256] = {};
            uint64_t total = 0;
            uint32_t vendor = 0, device = 0;
            bool integrated = false;
            if (!l0_enumerate_device(i, name, sizeof(name), &total,
                                     &vendor, &device, &integrated)) {
                continue;
            }
            BackendDeviceInfo d{};
            set_name(d.name, name);
            d.totalMem = total;
            d.vendorId = vendor;
            d.deviceId = device;
            d.integrated = integrated;
            d.vendorIndex = i;
            d.source = DeviceSource::Level0;
            d.preferred = best_backend_for(d);
            out.push_back(d);
        }
    }

    // ---- CUDA (NVIDIA; vendor implied 0x10DE) ----
    if (cuda_runtime_present()) {
        const int n = cuda_enumerate_count();
        for (int i = 0; i < n; ++i) {
            char name[256] = {};
            uint64_t total = 0;
            bool integrated = false;
            if (!cuda_enumerate_device(i, name, sizeof(name), &total, &integrated)) {
                continue;
            }
            BackendDeviceInfo d{};
            set_name(d.name, name);
            d.totalMem = total;
            d.vendorId = 0x10DE;
            d.integrated = integrated;
            d.vendorIndex = i;
            d.source = DeviceSource::Cuda;
            d.preferred = best_backend_for(d);
            out.push_back(d);
        }
    }

    // ---- Vulkan (transient instance, discovery only) ----
    {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "vvm-device-registry";
        app.apiVersion = VK_API_VERSION_1_0;
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
        VkInstance instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&ici, nullptr, &instance) == VK_SUCCESS) {
            uint32_t physCount = 0;
            if (vkEnumeratePhysicalDevices(instance, &physCount, nullptr) == VK_SUCCESS && physCount > 0) {
                std::vector<VkPhysicalDevice> phys(physCount);
                if (vkEnumeratePhysicalDevices(instance, &physCount, phys.data()) == VK_SUCCESS) {
                    for (uint32_t i = 0; i < physCount; ++i) {
                        VkPhysicalDeviceProperties props{};
                        vkGetPhysicalDeviceProperties(phys[i], &props);
                        VkPhysicalDeviceMemoryProperties mem{};
                        vkGetPhysicalDeviceMemoryProperties(phys[i], &mem);
                        uint64_t local = 0;
                        for (uint32_t h = 0; h < mem.memoryHeapCount; ++h) {
                            if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                                local += mem.memoryHeaps[h].size;
                            }
                        }
                        BackendDeviceInfo d{};
                        set_name(d.name, props.deviceName);
                        d.totalMem = local;
                        d.vendorId = props.vendorID;
                        d.deviceId = props.deviceID;
                        d.integrated = props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
                        d.vendorIndex = static_cast<int32_t>(i);
                        d.source = DeviceSource::Vulkan;
                        d.preferred = best_backend_for(d);
                        out.push_back(d);
                    }
                }
            }
            vkDestroyInstance(instance, nullptr);
        }
    }

    return out;
}

std::optional<UnifiedPoolSet> UnifiedPoolSet::create(const PoolConfig& cfg) {
    UnifiedPoolSet set;
    int hipIdx = 0, l0Idx = 0;
    for (const BackendDeviceInfo& dev : enumerate_all_devices()) {
        PooledDevice entry;
        entry.info = dev;
        if (dev.preferred == MemBackendKind::Hip) {
            DeviceConfig dc{};
            dc.backendDeviceIndex = dev.vendorIndex;
            dc.memBackendKind = static_cast<int32_t>(MemBackendKind::Hip);
            auto pool = UnifiedMemoryPool::create(dc, cfg);
            if (!pool.has_value()) {
                VVM_LOG_WARN("UnifiedPoolSet: HIP pool create failed for '{}'", dev.name);
                continue;
            }
            entry.pool = std::make_unique<UnifiedMemoryPool>(std::move(*pool));
            VVM_LOG_INFO("UnifiedPoolSet: HIP pool on '{}' (local index {})",
                         dev.name, hipIdx++);
        } else if (dev.preferred == MemBackendKind::Level0) {
            DeviceConfig dc{};
            dc.backendDeviceIndex = dev.vendorIndex;
            dc.memBackendKind = static_cast<int32_t>(MemBackendKind::Level0);
            auto pool = UnifiedMemoryPool::create(dc, cfg);
            if (!pool.has_value()) {
                VVM_LOG_WARN("UnifiedPoolSet: L0 pool create failed for '{}'", dev.name);
                continue;
            }
            entry.pool = std::make_unique<UnifiedMemoryPool>(std::move(*pool));
            VVM_LOG_INFO("UnifiedPoolSet: L0 pool on '{}' (local index {})",
                         dev.name, l0Idx++);
        }
        // Vulkan entries stay info-only: pools need the application's VkDevice.
        set.devices_.push_back(std::move(entry));
    }
    return set;
}

UnifiedMemoryPool* UnifiedPoolSet::pool(MemBackendKind kind, int vendorIndex) {
    for (auto& entry : devices_) {
        if (entry.info.preferred == kind && entry.info.vendorIndex == vendorIndex && entry.pool) {
            return entry.pool.get();
        }
    }
    return nullptr;
}

const char* UnifiedPoolSet::aggregate_stats_json() {
    json_ = "[";
    bool first = true;
    char buf[640];
    for (auto& entry : devices_) {
        if (!first) json_ += ",";
        first = false;
        const char* stats = "\"pooled\":false";
        char poolJson[512] = {};
        if (entry.pool) {
            const PoolStats s = entry.pool->getStats();
            stats = "\"pooled\":true";
            snprintf(poolJson, sizeof(poolJson),
                     ",\"blocks\":%u,\"allocations\":%u,\"capacityBytes\":%llu,"
                     "\"usedBytes\":%llu,\"fragmentation\":%.3f",
                     s.blockCount, s.allocationCount,
                     (unsigned long long)s.totalCapacity,
                     (unsigned long long)s.totalUsed,
                     (double)s.fragmentationRatio);
        }
        snprintf(buf, sizeof(buf),
                 "{\"device\":\"%.200s\",\"vendor\":\"0x%04x\",\"source\":%d,"
                 "\"preferred\":\"%s\",%s%s}",
                 entry.info.name, entry.info.vendorId,
                 static_cast<int>(entry.info.source),
                 backend_kind_name(entry.info.preferred),
                 stats, poolJson);
        json_ += buf;
    }
    json_ += "]";
    return json_.c_str();
}

// ---------------------------------------------------------------------------
// Plan-driven routing (mirrors the llama hooks' pick_from_plan exactly;
// both must resolve identically — keep in sync by construction: same
// positional expert indexing, same CPU-safe defaults).
// ---------------------------------------------------------------------------

PoolRoute route_plan(const PlacementPlan& plan, TensorClass cls, int32_t layer) {
    PoolRoute out;  // wantCpu=true default: unknown routes to CPU/mmap
    if (cls == TensorClass::LookupTable) {
        return out;  // PLE always CPU
    }
    if (cls == TensorClass::Expert) {
        if (layer < 0 || static_cast<size_t>(layer) >= plan.experts.size()) {
            return out;  // outside the plan: safe direction is CPU
        }
        const ExpertPlacement& ep = plan.experts[static_cast<size_t>(layer)];
        out.wantCpu = ep.onCpu || ep.deviceIndex < 0;
        out.kind = ep.backend;
        out.vendorIndex = ep.deviceIndex;
        return out;
    }
    out.wantCpu = plan.denseDeviceIndex < 0;
    out.kind = plan.denseBackend;
    out.vendorIndex = plan.denseDeviceIndex;
    return out;
}

// ---------------------------------------------------------------------------
// Auto tensor placement
// ---------------------------------------------------------------------------

TensorClass classify_tensor(const char* name, int32_t* layerOut) {
    if (layerOut) *layerOut = -1;
    if (!name) return TensorClass::Other;
    const std::string n(name);
    if (n.find("per_layer_token_embd") != std::string::npos) {
        return TensorClass::LookupTable;
    }
    // Routed experts: blk.N.ffn_(gate|up|down)_exps (qwen4exp) and the
    // generic ffn_*exps / expert spellings of other MoE arches.
    std::size_t blk = n.find("blk.");
    if (blk != std::string::npos) {
        const bool isExpert = n.find("ffn_") != std::string::npos &&
                              (n.find("_exps") != std::string::npos ||
                               n.find("experts") != std::string::npos);
        if (isExpert) {
            int layer = -1;
            if (sscanf(n.c_str() + blk, "blk.%d.", &layer) == 1 && layer >= 0) {
                if (layerOut) *layerOut = layer;
                return TensorClass::Expert;
            }
        }
        if (n.find("attn_") != std::string::npos ||
            n.find("ssm_") != std::string::npos ||
            n.find("qsa_") != std::string::npos ||
            n.find("deltanet") != std::string::npos) {
            return TensorClass::Attention;
        }
        return TensorClass::Dense;
    }
    return TensorClass::Dense;
}

PlacementPlan auto_place_experts(
    const std::vector<BackendDeviceInfo>& devices,
    const std::vector<TensorSpec>& tensors,
    uint64_t kvCacheBytes,
    float maxFraction) {

    return auto_place_experts_ex(devices, tensors, kvCacheBytes,
                                 UINT64_MAX, maxFraction);
}

PlacementPlan auto_place_experts_ex(
    const std::vector<BackendDeviceInfo>& devices,
    const std::vector<TensorSpec>& tensors,
    uint64_t kvCacheBytes,
    uint64_t hostCacheBytes,
    float maxFraction) {

    PlacementPlan plan{};
    if (maxFraction <= 0.0f || maxFraction > 1.0f) maxFraction = 0.90f;

    // Inventory: per-layer expert bytes, dense bytes, PLE bytes.
    uint64_t denseBytes = 0, pleBytes = 0;
    std::vector<uint64_t> expertBytes;   // indexed by layer
    int maxLayer = -1;
    for (const auto& t : tensors) {
        if (t.cls == TensorClass::LookupTable) {
            pleBytes += t.size;
        } else if (t.cls == TensorClass::Expert && t.layer >= 0) {
            if (t.layer > maxLayer) maxLayer = t.layer;
        } else {
            denseBytes += t.size;
        }
    }
    expertBytes.assign(maxLayer >= 0 ? static_cast<size_t>(maxLayer) + 1 : 0, 0);
    for (const auto& t : tensors) {
        if (t.cls == TensorClass::Expert && t.layer >= 0 &&
            static_cast<size_t>(t.layer) < expertBytes.size()) {
            expertBytes[static_cast<size_t>(t.layer)] += t.size;
        }
    }

    // Candidate GPU sinks. Expert fill order (measured):
    //   rank 0: HIP/CUDA discrete (fast kernels: +0.22 t/s/layer on gfx1100
    //           HIP; CUDA kernels are the mature reference path)
    //   rank 1: CPU RAM cache (unbounded mmap sink; degrades gracefully -
    //           measured working at 1.3x RAM oversubscription)
    //   rank 2: L0 discrete (unmeasured kernels: overflow only, and only
    //           after the CPU sink - never ahead of it)
    //   rank 3: Vulkan discrete (slow Q3_K kernels, -0.85 t/s/layer
    //           measured: last resort only)
    struct Sink {
        MemBackendKind kind;
        int32_t vendorIndex;
        uint64_t cap;      // maxFraction * heap
        uint64_t used = 0;
        int rank;          // lower = preferred
        bool overflowOnly = false;
        bool isCpu = false;
    };
    std::vector<Sink> sinks;
    for (const auto& d : devices) {
        if ((d.preferred == MemBackendKind::Hip ||
             d.preferred == MemBackendKind::Cuda) && !d.integrated) {
            sinks.push_back({d.preferred, d.vendorIndex,
                             static_cast<uint64_t>(d.totalMem * maxFraction), 0, 0});
        } else if (d.preferred == MemBackendKind::Level0 && !d.integrated) {
            sinks.push_back({MemBackendKind::Level0, d.vendorIndex,
                             static_cast<uint64_t>(d.totalMem * maxFraction), 0, 2, true});
        }
    }
    // CPU sink: unbounded unless the caller caps the host cache.
    sinks.push_back({MemBackendKind::Vulkan, -1, hostCacheBytes, 0, 1, false, true});
    // Vulkan-discrete last resort: enumerated, not pooled (the app's VkDevice
    // would be needed; llama's tensor split handles it). Listed for the map.
    // AMD/Intel only: NVIDIA serves through the CUDA backend (rank 0), so a
    // Vulkan sink for the same card would double-count its heap.
    for (const auto& d : devices) {
        if (d.source == DeviceSource::Vulkan && !d.integrated &&
            (d.vendorId == 0x1002 || d.vendorId == 0x8086)) {
            sinks.push_back({MemBackendKind::Vulkan, d.vendorIndex,
                             static_cast<uint64_t>(d.totalMem * maxFraction), 0, 3, true});
        }
    }
    // Dense target: co-locate with experts on the biggest rank-0 (HIP/CUDA)
    // sink; without one, the biggest discrete sink; without any, CPU.
    Sink* denseSink = nullptr;
    for (auto& s : sinks) {
        if (s.isCpu || s.overflowOnly || s.rank != 0) continue;
        if (!denseSink || s.cap > denseSink->cap) denseSink = &s;
    }
    if (!denseSink) {
        for (auto& s : sinks) {
            if (s.isCpu) continue;
            if (!denseSink || s.cap > denseSink->cap) denseSink = &s;
        }
    }
    if (!denseSink) {
        for (auto& s : sinks) {
            if (s.isCpu) { denseSink = &s; break; }
        }
    }
    plan.pleOnCpu = true;   // always (the 55.6x rule)
    plan.denseBackend = denseSink ? denseSink->kind : MemBackendKind::Vulkan;
    plan.denseDeviceIndex = denseSink ? denseSink->vendorIndex : -1;
    if (denseSink) denseSink->used += denseBytes + kvCacheBytes;

    // Experts greedy in rank order.
    std::sort(sinks.begin(), sinks.end(),
              [](const Sink& a, const Sink& b) { return a.rank < b.rank; });
    plan.experts.reserve(expertBytes.size());
    int cpuLayers = 0, gpuLayers = 0;
    int firstCpu = -1, lastGpu = -1;
    for (size_t l = 0; l < expertBytes.size(); ++l) {
        ExpertPlacement p{};
        p.layer = static_cast<int32_t>(l);
        p.onCpu = true;
        for (auto& s : sinks) {
            // Rank order does the policy work: HIP fill -> CPU cache ->
            // L0 overflow -> Vulkan-discrete last resort (measured slower
            // than CPU streaming, but better than OOM).
            if (s.used + expertBytes[l] <= s.cap) {
                s.used += expertBytes[l];
                p.onCpu = s.isCpu;
                p.backend = s.kind;
                p.deviceIndex = s.vendorIndex;
                break;
            }
        }
        if (p.onCpu) {
            ++cpuLayers;
            if (firstCpu < 0) firstCpu = static_cast<int>(l);
        } else {
            ++gpuLayers;
            lastGpu = static_cast<int>(l);
        }
        plan.experts.push_back(p);
    }

    // ncmoe-style summary when CPU layers form a suffix (llama's model);
    // otherwise the per-layer map is authoritative.
    if (gpuLayers == 0) {
        std::snprintf(plan.summary, sizeof(plan.summary),
                      "all %zu expert layers on CPU", plan.experts.size());
    } else if (!plan.experts.empty() && firstCpu >= 0 &&
        (lastGpu < 0 || static_cast<size_t>(lastGpu) < plan.experts.size() - 1)) {
        // mixed layout - report both counts and the equivalent suffix
        std::snprintf(plan.summary, sizeof(plan.summary),
                      "%d/%zu expert layers on GPU, %d on CPU (mixed)",
                      gpuLayers, plan.experts.size(), cpuLayers);
    } else if (firstCpu >= 0) {
        std::snprintf(plan.summary, sizeof(plan.summary),
                      "%d/%zu expert layers on GPU, %d on CPU (~--n-cpu-moe %d)",
                      gpuLayers, plan.experts.size(), cpuLayers, firstCpu);
    } else {
        std::snprintf(plan.summary, sizeof(plan.summary),
                      "all %zu expert layers on GPU", plan.experts.size());
    }
    return plan;
}

} // namespace vvm
