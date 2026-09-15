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

} // namespace vvm
