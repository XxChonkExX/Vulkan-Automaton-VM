#pragma once
// L0MemoryBackend: IDeviceMemoryBackend over Intel's Level Zero.
//
// ze_loader.dll is loaded DYNAMICALLY (ships with Intel GPU drivers on
// Windows) - no SDK, no link-time dependency, graceful nullptr from the
// factory when Level Zero is absent. Same pattern as the HIP backend.
//
// Memory model mapping (Vulkan vocabulary -> Level Zero):
//   - one synthetic DEVICE_LOCAL heap  = largest zeDeviceGetMemoryProperties
//                                        totalSize (memory ordinal 0)
//   - one synthetic memory type (0)    = DEVICE_LOCAL (device pointers)
//   - allocate   = zeMemAllocDevice    (ordinal 0, driver-default alignment)
//   - free       = zeMemFree
//   - map/unmap  = unsupported for device memory in v1 (zeMemAllocHost later)
//   - buffers    = the device pointer itself (echo semantics, like HIP)
//   - budget     = valid=false in v1 (Sysman zesDeviceGetMemoryState is a
//                  follow-up); the pool falls back to the static heap size
//
// Struct layouts below are signature-accurate to ze_api.h v1.18 and were
// verified against the official header (including the totalSize-before-name
// ordering and ZE_MAX_DEVICE_NAME=256).

#include "vulkan_vm/mem_backend.hpp"

namespace vvm {

class VVM_API L0MemoryBackend final : public IDeviceMemoryBackend {
public:
    // Returns nullptr when ze_loader.dll is missing or the device index
    // does not exist. deviceIndex < 0 selects device 0.
    static std::unique_ptr<L0MemoryBackend> create(int deviceIndex);

    const char* name() const override { return "level0"; }

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
    L0MemoryBackend() = default;
    // Opaque Level Zero handles (context/device) and the heap size, stored
    // as uint64 so this header needs no Level Zero types.
    uint64_t context_ = 0;
    uint64_t device_  = 0;
    uint64_t totalMem_ = 0;
    char deviceName_[256] = {};
};

} // namespace vvm
