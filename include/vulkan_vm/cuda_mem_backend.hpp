#pragma once
// CudaMemoryBackend: IDeviceMemoryBackend over the NVIDIA CUDA driver API.
//
// nvcuda.dll is loaded DYNAMICALLY (ships with every NVIDIA driver) - no
// link-time CUDA dependency, graceful nullptr from the factory when NVIDIA
// is absent. Mirrors HipMemoryBackend's loader pattern one-to-one.
//
// Memory model mapping (Vulkan vocabulary -> CUDA driver API):
//   - one synthetic DEVICE_LOCAL heap  = total device memory
//   - one synthetic memory type (0)    = DEVICE_LOCAL (device pointers)
//   - allocate   = cuMemAlloc_v2        (type index must be 0; needs a
//                  current context - primary context retained per device)
//   - free       = cuMemFree_v2
//   - map/unmap  = unsupported for device memory (nullptr/false)
//   - buffers    = the device pointer itself (echoed, offset form allowed -
//                  exactly like the HIP backend)
//   - budget     = cuMemGetInfo_v2 (live, per-device)
//   - address    = the pointer itself (has_device_address() == true)
//
// Device selection: DeviceConfig::backendDeviceIndex (primary context per
// device; every entry point re-affinitizes, same discipline as HIP).

#include "vulkan_vm/mem_backend.hpp"

namespace vvm {

class VVM_API CudaMemoryBackend final : public IDeviceMemoryBackend {
public:
    // Returns nullptr when nvcuda is not loadable or the device index does
    // not exist. deviceIndex < 0 selects device 0.
    static std::unique_ptr<CudaMemoryBackend> create(int deviceIndex);

    const char* name() const override { return "cuda"; }

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
    CudaMemoryBackend() = default;
    uint64_t totalMem_ = 0;
    int32_t deviceIndex_ = -1;   // cuMemAlloc targets the CURRENT context -
                                 // every entry point re-affinitizes to this
                                 // device's primary context (HIP discipline).
};

// Lightweight enumeration (no pool needed). NVIDIA vendor implied 0x10DE.
// Safe to call when the driver is absent (returns 0/false).
VVM_API int  cuda_enumerate_count();
VVM_API bool cuda_runtime_present();
VVM_API bool cuda_enumerate_device(int idx, char* nameOut, size_t nameLen,
                                   uint64_t* totalMemOut, bool* integratedOut);

} // namespace vvm
