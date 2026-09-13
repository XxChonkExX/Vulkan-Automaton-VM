#include "vulkan_vm/storage/dstorage_backend.hpp"

#ifdef VVM_PLATFORM_WINDOWS

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstring>

namespace vvm {
namespace storage {
namespace backend {

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------
bool DStorageBackend::runtimeAvailable() {
    HMODULE m = ::GetModuleHandleW(L"dstorage.dll");
    if (!m) m = ::LoadLibraryW(L"dstorage.dll");
    if (!m) return false;
    return ::GetProcAddress(m, "DStorageGetFactory") != nullptr;
}

bool DStorageBackend::d3d12Available() {
    HMODULE m = ::GetModuleHandleW(L"d3d12.dll");
    if (!m) m = ::LoadLibraryW(L"d3d12.dll");
    return m != nullptr;
}

Result DStorageBackend::luidFor(VkPhysicalDevice pd, AdapterLuid* out) {
    if (!pd || !out) return Result::error(ErrorCode::InvalidConfig, "null pd/out");

    // Vulkan 1.1 core: VkPhysicalDeviceIDProperties.deviceLUID (8 bytes).
    VkPhysicalDeviceIDProperties idProps{};
    idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &idProps;
    vkGetPhysicalDeviceProperties2(pd, &props2);

    if (!idProps.deviceLUIDValid) {
        return Result::error(ErrorCode::UnsupportedFeature,
                             "deviceLUID not valid (driver lacks LUID export)");
    }
    std::memcpy(&out->lowPart, idProps.deviceLUID, 4);
    std::memcpy(&out->highPart, idProps.deviceLUID + 4, 4);
    out->valid = true;
    return Result::success();
}

// ---------------------------------------------------------------------------
// Zero-copy bridge: LUID-matched D3D12 shared heap -> Vulkan import
// ---------------------------------------------------------------------------
Result DStorageBackend::importToPool(UnifiedMemoryPool* pool, VkPhysicalDevice pd,
                                     VkDeviceSize bytes, Allocation* out) {
    if (!pool || !out) return Result::error(ErrorCode::InvalidConfig, "null pool/out");
    if (bytes == 0) return Result::error(ErrorCode::InvalidConfig, "zero bytes");

    // D3D12 heap sizes must be 64 KiB-aligned.
    const UINT64 heapBytes = (static_cast<UINT64>(bytes) + 0xFFFF) & ~0xFFFFull;

    // 1) LUID of the Vulkan device's adapter.
    AdapterLuid luid;
    Result lr = luidFor(pd, &luid);
    if (!lr) return lr;

    // 2) DXGI adapter with the matching LUID.
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = ::CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return Result::error(ErrorCode::ExportFailed, "CreateDXGIFactory1 failed");

    ComPtr<IDXGIAdapter1> adapter;
    bool found = false;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) continue;
        if (desc.AdapterLuid.LowPart == luid.lowPart &&
            static_cast<int32_t>(desc.AdapterLuid.HighPart) == luid.highPart) {
            found = true;
            break;
        }
        adapter.Reset();
    }
    if (!found) {
        return Result::error(ErrorCode::ImportFailed,
                             "no DXGI adapter matches the Vulkan device LUID");
    }

    // 3) D3D12 device on that adapter.
    ComPtr<ID3D12Device> dev;
    hr = ::D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_1, IID_PPV_ARGS(&dev));
    if (FAILED(hr)) return Result::error(ErrorCode::ImportFailed, "D3D12CreateDevice failed");

    // 4) Shared heap, buffers only (DirectStorage destination).
    D3D12_HEAP_DESC heapDesc{};
    heapDesc.SizeInBytes = heapBytes;
    heapDesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapDesc.Alignment = D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT; // 4 MiB; large heaps
    heapDesc.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;

    ComPtr<ID3D12Heap> heap;
    hr = dev->CreateHeap(&heapDesc, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) {
        // Retry with 64 KiB alignment (small heaps).
        heapDesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        hr = dev->CreateHeap(&heapDesc, IID_PPV_ARGS(&heap));
    }
    if (FAILED(hr)) return Result::error(ErrorCode::ImportFailed, "CreateHeap(shared) failed");

    // 5) Placed resource over the whole heap — the producer's destination.
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Alignment = 0;
    rd.Width = heapBytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Buffers are inherently simultaneously accessible — no extra flags
    // (ALLOW_UNORDERED_ACCESS breaks CreateSharedHandle; SIMULTANEOUS_ACCESS
    // is texture-only and E_INVALIDARG on a buffer).

    ComPtr<ID3D12Resource> resource;
    hr = dev->CreatePlacedResource(heap.Get(), 0, &rd,
                                   D3D12_RESOURCE_STATE_COMMON, nullptr,
                                   IID_PPV_ARGS(&resource));
    if (FAILED(hr)) return Result::error(ErrorCode::ImportFailed, "CreatePlacedResource failed");

    // 6) NT handle + import. Handle-type fallback: HEAP first (DirectStorage's
    // native destination), then a shared committed resource.
    HANDLE ntHandle = nullptr;
    hr = dev->CreateSharedHandle(heap.Get(), nullptr, GENERIC_ALL, nullptr, &ntHandle);
    if (FAILED(hr)) return Result::error(ErrorCode::ExportFailed, "CreateSharedHandle failed");

    const VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    auto tryImport = [&](ExternalHandleType type, HANDLE handle) -> std::optional<Allocation> {
        ExternalMemoryInfo info;
        info.type = type;
        info.handle = ExternalHandle(handle);
        info.size = static_cast<VkDeviceSize>(heapBytes);
        info.memoryTypeIndex = UINT32_MAX; // hint only; pool re-selects on import
        info.dedicatedAllocation = true;
        return pool->importMemory(std::move(info), usage);
    };

    // HEAP import attempt.
    {
        auto alloc = tryImport(ExternalHandleType::D3D12Heap, ntHandle);
        if (alloc) {
            // Keep the D3D12 plumbing alive for the producer path.
            d3dDevice_ = dev.Detach();
            dstResource_ = resource.Detach();
            dxgiAdapter_ = adapter.Detach();
            *out = std::move(*alloc);
            return Result::success();
        }
        // Refused — the ExternalHandle destructor closed the handle. Re-create
        // for the resource-import fallbacks below.
        ntHandle = nullptr;
        dev->CreateSharedHandle(heap.Get(), nullptr, GENERIC_ALL, nullptr, &ntHandle);
    }

    // RESOURCE fallback via a shared COMMITTED resource. CreateSharedHandle
    // works on committed resources, heaps, and fences — NOT placed resources.
    // A shared committed resource is the classic D3D12<->Vulkan interop path
    // (AMD/Intel support it where raw heap imports are refused); its implicit
    // heap is the shared memory, so zero-copy is preserved.
    {
        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC crd = rd;
        ComPtr<ID3D12Resource> committed;
        hr = dev->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_SHARED, &crd,
                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                          IID_PPV_ARGS(&committed));
        if (SUCCEEDED(hr)) {
            HANDLE rh = nullptr;
            hr = dev->CreateSharedHandle(committed.Get(), nullptr, GENERIC_ALL, nullptr, &rh);
            if (SUCCEEDED(hr)) {
                auto alloc = tryImport(ExternalHandleType::D3D12Resource, rh);
                if (alloc) {
                    d3dDevice_ = dev.Detach();
                    dstResource_ = committed.Detach();
                    dxgiAdapter_ = adapter.Detach();
                    if (ntHandle) ::CloseHandle(ntHandle);
                    *out = std::move(*alloc);
                    return Result::success();
                }
            } else if (rh) {
                ::CloseHandle(rh);
            }
        }
    }

    if (ntHandle) ::CloseHandle(ntHandle);
    return Result::error(ErrorCode::ImportFailed,
                         "driver imports neither D3D12_HEAP nor shared-committed D3D12_RESOURCE");
}

// ---------------------------------------------------------------------------
// StreamBackend (producer path) — gated on the DirectStorage SDK.
//
// When dstorage.h is present (NuGet Microsoft.DirectStorage), the producer is:
//   DStorageGetFactory -> IDStorageFactory1::SetStagingBufferSize
//   -> OpenFile (BypassIO) -> CreateQueue (DSTORAGE_REQUEST_SOURCE_FILE,
//   DSTORAGE_REQUEST_DESTINATION_BUFFER) -> EnqueueRequest{fileOffset, size,
//   dst: resource + offset} -> Submit -> IDStorageStatusArray fence polls.
// Without the SDK these stay honest stubs: the import bridge above is usable
// by any D3D12 producer (including a test harness).
// ---------------------------------------------------------------------------
DStorageBackend::~DStorageBackend() {
    if (dstResource_) static_cast<IUnknown*>(dstResource_)->Release();
    if (d3dDevice_) static_cast<IUnknown*>(d3dDevice_)->Release();
    if (dxgiAdapter_) static_cast<IUnknown*>(dxgiAdapter_)->Release();
}

uint32_t DStorageBackend::submitBatch(const std::vector<IORequest>& batch) {
    (void)batch;
    return 0; // TODO(producer): EnqueueRequest into dstResource_ when SDK lands
}

uint32_t DStorageBackend::pollCompletions(std::vector<uint64_t>* outIds,
                                          std::vector<uint64_t>* outFailed) {
    (void)outIds;
    (void)outFailed;
    return 0; // TODO(producer): IDStorageStatusArray fence polls
}

} // namespace backend
} // namespace storage
} // namespace vvm

#else // !VVM_PLATFORM_WINDOWS

#include "vulkan_vm/storage/dstorage_backend.hpp"

namespace vvm {
namespace storage {
namespace backend {

DStorageBackend::~DStorageBackend() = default;
bool DStorageBackend::runtimeAvailable() { return false; }
bool DStorageBackend::d3d12Available() { return false; }
Result DStorageBackend::luidFor(VkPhysicalDevice, AdapterLuid*) {
    return Result::error(ErrorCode::UnsupportedFeature, "Windows-only");
}
Result DStorageBackend::importToPool(UnifiedMemoryPool*, VkPhysicalDevice, VkDeviceSize, Allocation*) {
    return Result::error(ErrorCode::UnsupportedFeature, "Windows-only bridge");
}
uint32_t DStorageBackend::submitBatch(const std::vector<IORequest>&) { return 0; }
uint32_t DStorageBackend::pollCompletions(std::vector<uint64_t>*, std::vector<uint64_t>*) { return 0; }

} // namespace backend
} // namespace storage
} // namespace vvm

#endif // VVM_PLATFORM_WINDOWS