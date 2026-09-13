// Path B: DirectStorage + D3D12 -> Vulkan import (experimental).
// Without VVM_HAS_DSTORAGE (DirectStorage SDK + d3d12.lib linkage) this TU
// compiles to capability stubs so the core library never depends on the SDK.
// With VVM_HAS_DSTORAGE=1, fill in: IDStorageFactory::OpenFile,
// IDStorageQueue::EnqueueRequest into a D3D12 placed resource on a
// D3D12_HEAP_FLAG_SHARED heap, Submit+ fence, CreateSharedHandle (NT handle),
// then vkImportMemoryWin32HandleKHR with VK_EXTERNAL_MEMORY_HANDLE_TYPE
// D3D12_HEAP_BIT into pool->importMemory. LUID-match the Vulkan
// VkPhysicalDevice (vkGetPhysicalDeviceProperties ID / DXGI AdapterLuid) to
// the D3D12 device — D3D12CreateDevice(NULL) picks the iGPU on laptops.

#include "vulkan_vm/storage_stream.hpp"

namespace vvm::storage {

bool dstorageRuntimeAvailable() {
#ifdef VVM_HAS_DSTORAGE
    // TODO: Load dstorage.dll, DStorageGetFactory, check BypassIO + NVMe.
    return true;
#else
    return false;
#endif
}

Result dstorageImportToPool(UnifiedMemoryPool* /*pool*/,
                            void* /*d3d12SharedHeapNtHandle*/,
                            VkDeviceSize /*bytes*/,
                            Allocation* /*out*/) {
#ifdef VVM_HAS_DSTORAGE
    // TODO(mvp-b): implement shared-heap import via pool->importMemory with
    // ExternalHandleType::D3D12Heap. Keep R3/R4: NT handle consumed on success.
    return Result::error(ErrorCode::UnsupportedFeature, "dstorage import TODO");
#else
    return Result::error(ErrorCode::UnsupportedFeature,
                         "DirectStorage path not built (cmake -DVVM_BUILD_DSTORAGE=ON + SDK)");
#endif
}

} // namespace vvm::storage
