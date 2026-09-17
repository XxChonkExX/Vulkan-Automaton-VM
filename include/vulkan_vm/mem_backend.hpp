#pragma once
// ============================================================================
// IDeviceMemoryBackend - the vendor seam for the Chonk Buffer pool.
//
// The pool's POLICY (buddy allocator, chunk tiers, adaptive block sizing,
// budget heuristics, best-fit first block) is vendor-neutral. This interface
// is the MECHANISM seam: everything a vendor driver actually has to do.
//
//   - VulkanMemoryBackend  (shipped)  : VkDeviceMemory / VkBuffer
//   - HipMemoryBackend     (planned)  : hipMalloc device pointers
//   - L0MemoryBackend      (planned)  : zeMemAllocDevice
//
// Handle discipline: handles are opaque uint64 tokens that are ONLY ever
// passed back to the backend that produced them. The Vulkan backend stores
// VkDeviceMemory/VkBuffer casts (lossless, pointer-sized); a HIP backend
// stores device pointers. This lets BlockInfo/Allocation keep their current
// storage without breaking the public ABI.
//
// Property/heap flag language: VK_MEMORY_PROPERTY_*/VK_MEMORY_HEAP_* bit
// values are the universal vocabulary. Vendors map their concepts into these
// bits (e.g. HIP managed memory -> DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT).
// Pool-side selection policy already speaks these bits.
//
// v1 scope (this revision): the memory plane - discovery, live budget,
// allocate/free/map/unmap, buffer create/bind/destroy, device addresses.
// Export/import (Win32 handles), offload migration and the transfer-queue
// copy engine remain Vulkan-direct inside UnifiedMemoryPool until the HIP
// adapter lands (they need vendor queuing semantics, not just allocation).
// ============================================================================

#include <cstdint>
#include <memory>
#include <vector>

#include "vulkan_vm/core.hpp"
#include "vulkan_vm/utils.hpp"

namespace vvm {

using BackendMemory = uint64_t;   // 0 == null
using BackendBuffer = uint64_t;   // 0 == null

// Error-code convention: backends return their native error code packed so
// it can never collide with success (0). Vulkan backend packs VkResult.
// NOTE: encoded values are ALWAYS negative (encode maps any int N to
// -1000000-N), including for negative native codes: VK_ERROR_OOM (-2)
// encodes to -999998, not below -1000000. Decode must accept every
// negative int (found 2026-09: every driver error logged "unknown").
inline int encode_backend_error(int nativeCode) { return -1000000 - nativeCode; }
inline bool decode_backend_error(int code, int* nativeOut) {
    if (code < 0) { *nativeOut = -1000000 - code; return true; }
    return false;
}

struct BackendMemType {
    uint32_t heapIndex      = 0;
    uint32_t propertyFlags  = 0;   // VK_MEMORY_PROPERTY_* bits
};

struct BackendHeap {
    uint64_t size           = 0;
    uint32_t flags          = 0;   // VK_MEMORY_HEAP_* bits
};

struct BackendBudget {
    uint64_t budgetBytes    = 0;
    uint64_t usedBytes      = 0;
    bool     valid          = false;
};

struct BackendAllocRequest {
    uint64_t      size          = 0;
    uint32_t      typeIndex     = 0;
    bool          exportable    = false;
    // Vendor equivalent of the DEVICE_ADDRESS allocate flag (Vulkan:
    // VkMemoryAllocateFlagsInfo with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT).
    bool          deviceAddress = false;
    // Vendor equivalent of VK_EXT_memory_priority; <= 0 means unset.
    float         priority      = 0.0f;
    // A buffer previously created UNBOUND via create_buffer(mem=0): the
    // backend adds its dedicated-allocation hint (Vulkan:
    // VkMemoryDedicatedAllocateInfo). 0 = none.
    BackendBuffer dedicatedFor  = 0;
};

class IDeviceMemoryBackend {
public:
    virtual ~IDeviceMemoryBackend() = default;

    virtual const char* name() const = 0;

    // ---- Discovery ------------------------------------------------------
    virtual std::vector<BackendHeap>    heaps() const       = 0;
    virtual std::vector<BackendMemType> memoryTypes() const = 0;
    // Live budget (VK_EXT_memory_budget analogue). valid=false when the
    // vendor stack cannot report it; callers fall back to heuristics.
    virtual BackendBudget heapBudget(uint32_t heapIndex) const = 0;
    virtual bool  type_is_host_visible(uint32_t typeIndex) const = 0;

    // ---- Memory plane ---------------------------------------------------
    // resultError (optional) receives a backend-native error code on failure
    // (Vulkan backend: VkResult) for caller-side logging. Returns 0 on
    // failure.
    virtual BackendMemory allocate(const BackendAllocRequest& req,
                                   int* resultError) = 0;
    virtual void free(BackendMemory mem) = 0;

    // Whole-allocation map. Returns nullptr when the memory is not
    // host-visible (or on failure); ok (optional) distinguishes the two.
    virtual void* map(BackendMemory mem, bool* ok) = 0;
    virtual void  unmap(BackendMemory mem) = 0;

    // ---- Buffer plane ---------------------------------------------------
    // usageBits use VK_BUFFER_USAGE_* bit values as the universal vocabulary.
    // Pass mem == 0 to create an UNBOUND buffer (dedicated-allocation flow:
    // create unbound -> allocate dedicatedFor=buffer -> bind).
    virtual BackendBuffer create_buffer(BackendMemory mem, uint64_t offset,
                                        uint64_t size, uint64_t usageBits,
                                        bool exportable, int* resultError) = 0;
    // Bind an unbound buffer to memory. No-op when the backend bound at
    // create time (mem != 0 path).
    virtual bool bind_buffer(BackendBuffer buf, BackendMemory mem) = 0;
    virtual void destroy_buffer(BackendBuffer buf) = 0;
    // Returns 0 when buffer device addresses are unavailable.
    virtual uint64_t buffer_device_address(BackendBuffer buf) const = 0;

    // ---- Capabilities ---------------------------------------------------
    // Vendor equivalent of the bufferDeviceAddress device feature (Vulkan:
    // feature at device creation; HIP: always available on device pointers).
    virtual bool has_device_address() const = 0;
    virtual bool supports_export(ExternalHandleType type) const = 0;
};

// Factory. kind selects the vendor implementation; returns nullptr for kinds
// whose adapter is not linked into this build. DeviceConfig::memBackendKind
// carries one of these values (-1 = Auto, resolved by the pool).
enum class MemBackendKind : int32_t { Auto = -1, Vulkan = 0, Hip = 1, Level0 = 2 };

VVM_API std::unique_ptr<IDeviceMemoryBackend> create_memory_backend(MemBackendKind kind,
                                                                     const DeviceConfig& cfg);

} // namespace vvm

