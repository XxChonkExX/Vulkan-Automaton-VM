// ============================================================================
// Unified device registry: enumerate HIP + Level Zero + Vulkan devices in
// one process, pick the best backend per device, and create pools for the
// backends that own their devices (HIP, Level Zero).
// ============================================================================

#include "vulkan_vm/device_registry.hpp"
#include "vulkan_vm/hip_mem_backend.hpp"
#include "vulkan_vm/l0_mem_backend.hpp"
#include "vulkan_vm/utils.hpp"

#include <vulkan/vulkan.h>
#include <cstring>
#include <cstdio>

namespace vvm {

const char* backend_kind_name(MemBackendKind kind) {
    switch (kind) {
        case MemBackendKind::Vulkan: return "vulkan";
        case MemBackendKind::Hip:    return "hip";
        case MemBackendKind::Level0: return "level0";
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

} // namespace vvm
