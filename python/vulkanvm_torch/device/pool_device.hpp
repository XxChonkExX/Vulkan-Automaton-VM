// pool_device.hpp - Vulkan instance/device selection and Chonk Buffer pool
// lifecycle for the PyTorch integration. Owns the global pool instance.

#pragma once

#include <vulkan_vm/vulkan_vm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace vvm_torch {

// Device selection policy (audit: machine-specific workarounds belong in a
// policy, not silently in library semantics). Select via
// CHONK_DEVICE_PREFERENCE=best|prefer_amd|prefer_discrete.
enum class DevicePreference {
    BestScore,     // pure score ranking
    PreferAmd,     // prefer AMD vendors outright (llvmpipe workaround)
    PreferDiscrete // prefer discrete GPUs
};

DevicePreference devicePreferenceFromEnv();

struct DeviceEntryInfo {
    std::string name;
    uint32_t vendor = 0;
    int score = 0;
    uint64_t heapBytes = 0;
};

struct PoolInitInfo {
    std::string deviceName;
    uint64_t heapBytes = 0;
    std::vector<DeviceEntryInfo> devices;
};

// Create the Vulkan instance, select a physical device per policy, create
// the VkDevice, and construct the Chonk Buffer pool. Throws std::runtime_error
// on failure. Idempotent-guard: call only when the pool is not yet created.
PoolInitInfo initPoolCore();

// Live pool (nullptr before initPoolCore / after shutdownPool).
vvm::UnifiedMemoryPool* pool();

// Allocations whose lifetime is tied to the pool (alloc_keep etc.). Deallocated
// by shutdownPoolCore before the pool is destroyed.
std::vector<vvm::Allocation>& keptAllocations();

// Tear down the pool (allocator must be reset first).
void shutdownPoolCore();

}  // namespace vvm_torch
