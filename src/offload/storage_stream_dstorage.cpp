// Path B: DirectStorage interop helpers (query + producer-handle import).
//
// Two surfaces live here, neither needing the DirectStorage SDK:
//   - dstorageRuntimeAvailable(): the PRODUCER path needs the SDK build
//     (VVM_HAS_DSTORAGE) AND the redistributable runtime (dstorage.dll +
//     DStorageGetFactory export) on the machine. Both gates are checked.
//   - dstorageImportToPool(): imports a PRODUCER-CREATED D3D12 shared heap
//     NT handle (e.g. from an external DirectStorage flow) via
//     pool->importMemory (D3D12_HEAP). Takes ownership: consumed by the
//     driver on success, closed on failure (R3/R4).
// The SDK-backed producer itself (IDStorageFactory queue over a heap the
// bridge creates) lives in DStorageBackend and stays gated on dstorage.h.

#include "vulkan_vm/storage_stream.hpp"

#if defined(VVM_HAS_DSTORAGE) && defined(VVM_PLATFORM_WINDOWS)
#include "vulkan_vm/storage/dstorage_backend.hpp"
#endif

namespace vvm::storage {

bool dstorageRuntimeAvailable() {
#if defined(VVM_HAS_DSTORAGE) && defined(VVM_PLATFORM_WINDOWS)
    return backend::DStorageBackend::runtimeAvailable();
#else
    return false;
#endif
}

Result dstorageImportToPool(UnifiedMemoryPool* pool,
                            void* d3d12SharedHeapNtHandle,
                            VkDeviceSize bytes,
                            Allocation* out) {
    if (!pool || !out) {
        return Result::error(ErrorCode::InvalidConfig, "null pool/out");
    }
    if (!d3d12SharedHeapNtHandle || bytes == 0) {
        return Result::error(ErrorCode::InvalidConfig, "null handle/zero bytes");
    }
#if defined(VVM_HAS_DSTORAGE) && defined(VVM_PLATFORM_WINDOWS)
    ExternalMemoryInfo info;
    info.type = ExternalHandleType::D3D12Heap;
    info.handle =
        ExternalHandle(static_cast<HANDLE>(d3d12SharedHeapNtHandle));
    info.size = bytes;
    info.memoryTypeIndex = UINT32_MAX;  // hint only; pool re-selects on import
    info.dedicatedAllocation = true;
    // Canonical streaming usage (matches the bridge's own imports).
    constexpr VkBufferUsageFlags kUsage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    auto alloc = pool->importMemory(std::move(info), kUsage);
    if (!alloc) {
        return Result::error(ErrorCode::ImportFailed,
                             "D3D12_HEAP NT-handle import refused by driver");
    }
    *out = std::move(*alloc);
    return Result::success();
#else
    (void)d3d12SharedHeapNtHandle;
    (void)bytes;
    return Result::error(ErrorCode::UnsupportedFeature,
                         "DirectStorage path not built (cmake -DVVM_BUILD_DSTORAGE=ON + SDK)");
#endif
}

}  // namespace vvm::storage
