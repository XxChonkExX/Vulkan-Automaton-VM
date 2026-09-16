#pragma once

// L3 backend — DirectStorage zero-copy bridge (Windows).
//
// The bridge (real, testable today — d3d12.h ships in the Windows SDK):
//   LUID-match the Vulkan physical device to its DXGI adapter
//   -> D3D12CreateDevice on that adapter
//   -> ID3D12Device::CreateHeap (D3D12_HEAP_FLAG_SHARED | ALLOW_ONLY_BUFFERS)
//   -> ID3D12Device::CreatePlacedResource over the whole heap (the producer's
//      destination; D3D12 resource and Vulkan buffer alias the same pages)
//   -> ID3D12Device::CreateSharedHandle (NT handle)
//   -> pool->importMemory (VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT)
// Result: DirectStorage DMA writes land directly in the memory the Vulkan
// buffer aliases — zero copies, zero staging.
//
// The producer (SSD -> shared heap via DirectStorage) requires the
// DirectStorage SDK (NuGet `Microsoft.DirectStorage`, provides dstorage.h)
// and the redistributable runtime (dstorage.dll + dstoragecore.dll, ships
// with DirectStorage games). It is compiled only when dstorage.h is present;
// until then the backend reports the producer unsupported while the import
// bridge itself remains fully usable (any D3D12 producer — including a test
// harness — can fill the heap).

#include "vulkan_vm/storage/request_queue.hpp"
#include "vulkan_vm/storage/ioring_backend.hpp" // BackendConfig, backend using-decls
#include "vulkan_vm/core.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vvm {
namespace storage {
namespace backend {

using queue::StreamBackend;
using queue::IORequest;

// LUID of the adapter a Vulkan physical device lives on (Windows only).
struct VVM_API AdapterLuid {
    uint32_t lowPart = 0;
    int32_t highPart = 0;
    bool valid = false;
};

class VVM_API DStorageBackend final : public StreamBackend {
public:
    explicit DStorageBackend(BackendConfig cfg) : cfg_(std::move(cfg)) {}
    ~DStorageBackend() override;

    DStorageBackend(const DStorageBackend&) = delete;
    DStorageBackend& operator=(const DStorageBackend&) = delete;

    // --- capability probes (no side effects) ---
    // dstorage.dll + DStorageGetFactory export present (producer runnable).
    static bool runtimeAvailable();
    // d3d12 runtime loadable (bridge runnable).
    static bool d3d12Available();

    // Resolve the DXGI LUID of a Vulkan physical device via
    // VkPhysicalDeviceIDProperties (deviceLUIDValid required).
    static Result luidFor(VkPhysicalDevice pd, AdapterLuid* out);

    // --- zero-copy bridge ---
    // Creates the LUID-matched shared heap + placed resource and imports the
    // heap into `pool` as device-local memory (R1: pool owns the Allocation).
    // `out` receives a buffer aliasing the same pages the D3D12 resource
    // covers — DirectStorage DMA writes are visible to shaders with no copy.
    Result importToPool(UnifiedMemoryPool* pool, VkPhysicalDevice pd,
                        VkDeviceSize bytes, Allocation* out);

    // --- StreamBackend (producer path) ---
    // The producer requires the DirectStorage SDK (dstorage.h). Without it
    // the backend reports canSubmit() == false so callers fail loud instead
    // of livelocking on 0-accept; the import bridge above remains fully
    // usable (any D3D12 producer - including a test harness - can fill the
    // heap). No link dependency on the SDK either way.
    bool canSubmit() const override;
    uint32_t submitBatch(const std::vector<IORequest>& batch) override;
    uint32_t pollCompletions(std::vector<uint64_t>* outIds,
                             std::vector<uint64_t>* outFailed = nullptr) override;
    const char* name() const override {
        return runtimeAvailable() ? "dstorage" : "dstorage(no-runtime)";
    }

private:
    BackendConfig cfg_;
    void* d3dDevice_ = nullptr;    // ID3D12Device*
    void* dstResource_ = nullptr;  // ID3D12Resource* placed over the heap
    void* dxgiAdapter_ = nullptr;  // IDXGIAdapter1* (kept for the producer)
};

} // namespace backend
} // namespace storage
} // namespace vvm