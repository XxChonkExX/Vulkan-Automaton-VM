#pragma once
// HipMemoryBackend: IDeviceMemoryBackend over the HIP runtime.
//
// The HIP runtime is loaded DYNAMICALLY (amdhip64_7.dll / amdhip64.dll) -
// no link-time dependency, graceful nullptr from the factory when HIP is
// absent. This mirrors the DStorage backend's loader pattern.
//
// Memory model mapping (Vulkan vocabulary -> HIP):
//   - one synthetic DEVICE_LOCAL heap  = total device memory
//   - one synthetic memory type (0)    = DEVICE_LOCAL (device pointers)
//   - allocate   = hipMalloc           (type index must be 0)
//   - free       = hipFree
//   - map/unmap  = unsupported for device memory (nullptr/false); host-visible
//                  staging via hipHostMalloc is a v2 concern
//   - buffers    = the device pointer itself (create_buffer echoes it; the
//                  offset form is unsupported - allocations are standalone)
//   - budget     = hipMemGetInfo (live, per-device - BETTER than a static
//                  heap fraction)
//
// Device selection: DeviceConfig::backendDeviceIndex (hipSetDevice at
// creation; all allocation happens on that device).

#include "vulkan_vm/mem_backend.hpp"

namespace vvm {

class VVM_API HipMemoryBackend final : public IDeviceMemoryBackend {
public:
    // Returns nullptr when the HIP runtime is not loadable or the device
    // index does not exist. deviceIndex < 0 selects device 0.
    static std::unique_ptr<HipMemoryBackend> create(int deviceIndex);

    const char* name() const override { return "hip"; }

    std::vector<BackendHeap>    heaps() const override;
    std::vector<BackendMemType> memoryTypes() const override;
    BackendBudget heapBudget(uint32_t heapIndex) const override;
    bool  type_is_host_visible(uint32_t typeIndex) const override;

    BackendMemory allocate(const BackendAllocRequest& req,
                           int* resultError) override;
    void free(BackendMemory mem) override;
    void* map(BackendMemory mem, bool* ok) override;
    void  unmap(BackendMemory mem) override;

    BackendBuffer create_buffer(BackendMemory mem, uint64_t offset,
                                uint64_t size, uint64_t usageBits,
                                bool exportable, int* resultError) override;
    bool bind_buffer(BackendBuffer buf, BackendMemory mem) override;
    void destroy_buffer(BackendBuffer buf) override;
    uint64_t buffer_device_address(BackendBuffer buf) const override;

    bool has_device_address() const override { return true; }
    bool supports_export(ExternalHandleType type) const override;

private:
    HipMemoryBackend() = default;
    uint64_t totalMem_ = 0;
};

// Lightweight enumeration (no pool needed). HIP is AMD-only so vendor is
// implied 0x1002. Safe to call when the runtime is absent (returns 0/false).
VVM_API int  hip_enumerate_count();
VVM_API bool hip_runtime_present();
VVM_API bool hip_enumerate_device(int idx, char* nameOut, size_t nameLen,
                                  uint64_t* totalMemOut, bool* integratedOut);

} // namespace vvm
