// ============================================================================
// CudaMemoryBackend implementation - dynamic nvcuda loader + cuMemAlloc plane.
// See cuda_mem_backend.hpp for the mapping rationale.
// ============================================================================

#include "vulkan_vm/cuda_mem_backend.hpp"
#include "vulkan_vm/utils.hpp"

#include <mutex>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>

namespace vvm {

namespace {

// Minimal CUDA driver-API surface, resolved dynamically. Signature-accurate
// to cuda.h (CUDA_SUCCESS == 0); only what the memory plane needs.
// CUdevice/CUcontext/CUdeviceptr are opaque ints/pointers; the integrated
// attribute is CUDA-numbered 18 (HIP uses 16 for its own enum - not the
// same numbering, verified against cuda.h).
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
struct CUctx_st;
typedef struct CUctx_st* CUcontext;
const int CU_DEVICE_ATTRIBUTE_INTEGRATED = 18;

struct CudaApi {
    int (*cuInit)(unsigned int flags) = nullptr;
    int (*cuDeviceGetCount)(int* count) = nullptr;
    int (*cuDeviceGet)(CUdevice* device, int ordinal) = nullptr;
    int (*cuDeviceGetName)(char* name, int len, CUdevice dev) = nullptr;
    int (*cuDeviceTotalMem_v2)(size_t* bytes, CUdevice dev) = nullptr;
    int (*cuDeviceGetAttribute)(int* pi, int attrib, CUdevice dev) = nullptr;
    int (*cuDevicePrimaryCtxRetain)(CUcontext* pctx, CUdevice dev) = nullptr;
    int (*cuCtxSetCurrent)(CUcontext ctx) = nullptr;
    int (*cuMemAlloc_v2)(CUdeviceptr* dptr, size_t bytesize) = nullptr;
    int (*cuMemFree_v2)(CUdeviceptr dptr) = nullptr;
    int (*cuMemGetInfo_v2)(size_t* free, size_t* total) = nullptr;
    bool ok = false;
    bool enumOk = false;
};

const CudaApi& cudaApi() {
    static CudaApi api = [] {
        CudaApi a{};
        // nvcuda.dll ships with every NVIDIA driver; no toolkit needed.
        HMODULE m = ::GetModuleHandleW(L"nvcuda.dll");
        if (!m) m = ::LoadLibraryW(L"nvcuda.dll");
        if (!m) return a;
        auto resolve = [&](const char* name) {
            return reinterpret_cast<void*>(::GetProcAddress(m, name));
        };
        a.cuInit                 = reinterpret_cast<int (*)(unsigned int)>(resolve("cuInit"));
        a.cuDeviceGetCount       = reinterpret_cast<int (*)(int*)>(resolve("cuDeviceGetCount"));
        a.cuDeviceGet            = reinterpret_cast<int (*)(CUdevice*, int)>(resolve("cuDeviceGet"));
        a.cuDeviceGetName        = reinterpret_cast<int (*)(char*, int, CUdevice)>(resolve("cuDeviceGetName"));
        a.cuDeviceTotalMem_v2    = reinterpret_cast<int (*)(size_t*, CUdevice)>(resolve("cuDeviceTotalMem_v2"));
        if (!a.cuDeviceTotalMem_v2)
            a.cuDeviceTotalMem_v2 = reinterpret_cast<int (*)(size_t*, CUdevice)>(resolve("cuDeviceTotalMem"));
        a.cuDeviceGetAttribute   = reinterpret_cast<int (*)(int*, int, CUdevice)>(resolve("cuDeviceGetAttribute"));
        a.cuDevicePrimaryCtxRetain = reinterpret_cast<int (*)(CUcontext*, CUdevice)>(resolve("cuDevicePrimaryCtxRetain"));
        a.cuCtxSetCurrent        = reinterpret_cast<int (*)(CUcontext)>(resolve("cuCtxSetCurrent"));
        a.cuMemAlloc_v2          = reinterpret_cast<int (*)(CUdeviceptr*, size_t)>(resolve("cuMemAlloc_v2"));
        if (!a.cuMemAlloc_v2)
            a.cuMemAlloc_v2 = reinterpret_cast<int (*)(CUdeviceptr*, size_t)>(resolve("cuMemAlloc"));
        a.cuMemFree_v2           = reinterpret_cast<int (*)(CUdeviceptr)>(resolve("cuMemFree_v2"));
        if (!a.cuMemFree_v2)
            a.cuMemFree_v2 = reinterpret_cast<int (*)(CUdeviceptr)>(resolve("cuMemFree"));
        a.cuMemGetInfo_v2        = reinterpret_cast<int (*)(size_t*, size_t*)>(resolve("cuMemGetInfo_v2"));
        if (!a.cuMemGetInfo_v2)
            a.cuMemGetInfo_v2 = reinterpret_cast<int (*)(size_t*, size_t*)>(resolve("cuMemGetInfo"));
        bool core = a.cuInit && a.cuDeviceGetCount && a.cuDeviceGet &&
                    a.cuMemAlloc_v2 && a.cuMemFree_v2 && a.cuMemGetInfo_v2 &&
                    a.cuDevicePrimaryCtxRetain && a.cuCtxSetCurrent;
        if (core && a.cuInit(0) == 0) {
            a.ok = true;
            a.enumOk = a.cuDeviceGetName && a.cuDeviceTotalMem_v2 && a.cuDeviceGetAttribute;
        }
        return a;
    }();
    return api;
}

} // namespace

int cuda_enumerate_count() {
    const CudaApi& api = cudaApi();
    if (!api.ok) return 0;
    int count = 0;
    if (api.cuDeviceGetCount(&count) != 0 || count < 0) return 0;
    return count;
}

bool cuda_runtime_present() {
    const CudaApi& api = cudaApi();
    return api.ok;
}

bool cuda_enumerate_device(int idx, char* nameOut, size_t nameLen,
                           uint64_t* totalMemOut, bool* integratedOut) {
    const CudaApi& api = cudaApi();
    if (!api.enumOk) return false;
    int count = 0;
    if (api.cuDeviceGetCount(&count) != 0 || idx < 0 || idx >= count) return false;
    CUdevice dev = 0;
    if (api.cuDeviceGet(&dev, idx) != 0) return false;
    char name[256] = {};
    if (api.cuDeviceGetName(name, sizeof(name), dev) != 0) return false;
    size_t total = 0;
    if (api.cuDeviceTotalMem_v2(&total, dev) != 0) return false;
    int integrated = 0;
    if (api.cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev) != 0) {
        integrated = 0;
    }
    if (nameOut && nameLen > 0) {
        size_t n = 0;
        while (n + 1 < nameLen && n < sizeof(name) && name[n] != 0) { nameOut[n] = name[n]; ++n; }
        nameOut[n] = 0;
    }
    if (totalMemOut) *totalMemOut = total;
    if (integratedOut) *integratedOut = integrated != 0;
    return true;
}

namespace {

constexpr int kCudaErrorBase = -4000000;  // distinct from Vulkan/HIP/L0 ranges
int asError(int cudaErr) { return encode_backend_error(kCudaErrorBase - cudaErr); }

} // namespace

// Per-device primary contexts (retained once, process lifetime). cuMemAlloc
// targets the calling thread's CURRENT context, so every entry point sets
// its device's context first (HIP affinitization discipline).
static CUcontext cudaPrimaryCtx(int deviceIndex) {
    static std::mutex mtx;
    static std::unordered_map<int, CUcontext> ctxs;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = ctxs.find(deviceIndex);
    if (it != ctxs.end()) return it->second;
    const CudaApi& api = cudaApi();
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    if (api.ok && api.cuDeviceGet(&dev, deviceIndex) == 0 &&
        api.cuDevicePrimaryCtxRetain(&ctx, dev) == 0 && ctx) {
        ctxs[deviceIndex] = ctx;
        return ctx;
    }
    return nullptr;
}

static bool cudaUseDevice(int deviceIndex) {
    const CudaApi& api = cudaApi();
    if (!api.ok) return false;
    CUcontext ctx = cudaPrimaryCtx(deviceIndex);
    if (!ctx) return false;
    return api.cuCtxSetCurrent(ctx) == 0;
}

std::unique_ptr<CudaMemoryBackend> CudaMemoryBackend::create(int deviceIndex) {
    const CudaApi& api = cudaApi();
    if (!api.ok) {
        VVM_LOG_WARN("cuda backend: NVIDIA driver API (nvcuda.dll) not loadable");
        return nullptr;
    }
    int count = 0;
    if (api.cuDeviceGetCount(&count) != 0 || count <= 0) {
        VVM_LOG_WARN("cuda backend: no CUDA devices detected");
        return nullptr;
    }
    const int idx = deviceIndex < 0 ? 0 : deviceIndex;
    if (idx >= count) {
        VVM_LOG_WARN("cuda backend: device index {} >= count {}", idx, count);
        return nullptr;
    }
    if (!cudaUseDevice(idx)) {
        VVM_LOG_WARN("cuda backend: primary context setup failed for device {}", idx);
        return nullptr;
    }
    size_t freeBytes = 0, totalBytes = 0;
    if (api.cuMemGetInfo_v2(&freeBytes, &totalBytes) != 0) {
        VVM_LOG_WARN("cuda backend: cuMemGetInfo failed");
        return nullptr;
    }
    auto backend = std::unique_ptr<CudaMemoryBackend>(new CudaMemoryBackend());
    backend->totalMem_ = totalBytes;
    backend->deviceIndex_ = static_cast<int32_t>(idx);
    VVM_LOG_INFO("cuda backend: device {} ({} MB), driver API loaded dynamically",
                 idx, static_cast<uint64_t>(totalBytes) / (1024 * 1024));
    return backend;
}

std::vector<BackendHeap> CudaMemoryBackend::heaps() const {
    // Synthetic model: one DEVICE_LOCAL heap covering all device memory.
    return {{totalMem_, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT}};
}

std::vector<BackendMemType> CudaMemoryBackend::memoryTypes() const {
    // Synthetic model: type 0 = DEVICE_LOCAL device pointers.
    return {{0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}};
}

BackendBudget CudaMemoryBackend::heapBudget(uint32_t heapIndex) const {
    BackendBudget out{};
    if (heapIndex != 0) return out;
    const CudaApi& api = cudaApi();
    if (!cudaUseDevice(deviceIndex_)) return out;
    size_t freeBytes = 0, totalBytes = 0;
    if (api.ok && api.cuMemGetInfo_v2(&freeBytes, &totalBytes) == 0) {
        out.budgetBytes = totalBytes;
        out.usedBytes   = totalBytes - freeBytes;
        out.valid       = true;
    }
    return out;
}

bool CudaMemoryBackend::type_is_host_visible(uint32_t typeIndex) const {
    // Device memory is not host-mappable; type 0 is device-local only.
    (void)typeIndex;
    return false;
}

BackendMemory CudaMemoryBackend::allocate(const BackendAllocRequest& req,
                                          int* resultError) {
    if (resultError) *resultError = 0;
    if (req.typeIndex != 0) {
        if (resultError) *resultError = asError(1);
        return 0;
    }
    const CudaApi& api = cudaApi();
    if (!api.ok || !cudaUseDevice(deviceIndex_)) {
        if (resultError) *resultError = asError(2);
        return 0;
    }
    CUdeviceptr ptr = 0;
    const int rc = api.cuMemAlloc_v2(&ptr, static_cast<size_t>(req.size));
    if (rc != 0 || !ptr) {
        if (resultError) *resultError = asError(rc ? rc : 3);
        return 0;
    }
    // deviceAddress/priority/dedicated hints are CUDA no-ops: a device
    // pointer IS the address, and allocations are standalone.
    return static_cast<uint64_t>(ptr);
}

void CudaMemoryBackend::free(BackendMemory mem) {
    if (mem == 0) return;
    const CudaApi& api = cudaApi();
    // Affinity: free must run under the owning device's context.
    if (!cudaUseDevice(deviceIndex_)) return;
    if (api.ok) api.cuMemFree_v2(static_cast<CUdeviceptr>(mem));
}

void* CudaMemoryBackend::map(BackendMemory mem, bool* ok) {
    // CUDA device memory is not host-mappable; host-visible staging is v2.
    if (ok) *ok = false;
    (void)mem;
    return nullptr;
}

void CudaMemoryBackend::unmap(BackendMemory mem) {
    (void)mem;
}

BackendBuffer CudaMemoryBackend::create_buffer(BackendMemory mem, uint64_t offset,
                                               uint64_t size, uint64_t usageBits,
                                               bool exportable, int* resultError) {
    // A CUDA device pointer IS the buffer: sub-allocations are
    // pointer+offset, exactly how compute consumes them (HIP discipline:
    // lifetime follows the memory handle; destroy is a no-op).
    if (resultError) *resultError = 0;
    if (mem == 0) {
        if (resultError) *resultError = asError(4);
        return 0;
    }
    (void)size; (void)usageBits; (void)exportable;
    return mem + offset;
}

bool CudaMemoryBackend::bind_buffer(BackendBuffer buf, BackendMemory mem) {
    // Nothing to bind: pointer == buffer == memory.
    return buf != 0 && buf == mem;
}

void CudaMemoryBackend::destroy_buffer(BackendBuffer buf) {
    // Ownership follows the memory handle; never free sub-pointers.
    (void)buf;
}

uint64_t CudaMemoryBackend::buffer_device_address(BackendBuffer buf) const {
    return buf;   // a CUDA device pointer IS the address
}

bool CudaMemoryBackend::supports_export(ExternalHandleType type) const {
    // CUDA<->Vulkan external-memory import (cuImportExternalMemory) is a v2
    // concern; standalone allocations need nothing here.
    (void)type;
    return false;
}

} // namespace vvm

#else
// ============================================================================
// Non-Windows: dlopen-based libcuda loader (same surface as the Windows
// branch; libcuda.so.1 ships with every NVIDIA driver).
// ============================================================================

#include "vulkan_vm/cuda_mem_backend.hpp"
#include "vulkan_vm/utils.hpp"

#include <dlfcn.h>
#include <unordered_map>
#include <mutex>

namespace vvm {

namespace {

typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
struct CUctx_st;
typedef struct CUctx_st* CUcontext;
const int CU_DEVICE_ATTRIBUTE_INTEGRATED = 18;

struct CudaApi {
    int (*cuInit)(unsigned int flags) = nullptr;
    int (*cuDeviceGetCount)(int* count) = nullptr;
    int (*cuDeviceGet)(CUdevice* device, int ordinal) = nullptr;
    int (*cuDeviceGetName)(char* name, int len, CUdevice dev) = nullptr;
    int (*cuDeviceTotalMem_v2)(size_t* bytes, CUdevice dev) = nullptr;
    int (*cuDeviceGetAttribute)(int* pi, int attrib, CUdevice dev) = nullptr;
    int (*cuDevicePrimaryCtxRetain)(CUcontext* pctx, CUdevice dev) = nullptr;
    int (*cuCtxSetCurrent)(CUcontext ctx) = nullptr;
    int (*cuMemAlloc_v2)(CUdeviceptr* dptr, size_t bytesize) = nullptr;
    int (*cuMemFree_v2)(CUdeviceptr dptr) = nullptr;
    int (*cuMemGetInfo_v2)(size_t* free, size_t* total) = nullptr;
    bool ok = false;
    bool enumOk = false;
};

const CudaApi& cudaApi() {
    static CudaApi api = [] {
        CudaApi a{};
        void* m = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        if (!m) m = ::dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
        if (!m) return a;
        auto resolve = [&](const char* name) {
            return reinterpret_cast<void*>(::dlsym(m, name));
        };
        a.cuInit                   = reinterpret_cast<int (*)(unsigned int)>(resolve("cuInit"));
        a.cuDeviceGetCount         = reinterpret_cast<int (*)(int*)>(resolve("cuDeviceGetCount"));
        a.cuDeviceGet              = reinterpret_cast<int (*)(CUdevice*, int)>(resolve("cuDeviceGet"));
        a.cuDeviceGetName          = reinterpret_cast<int (*)(char*, int, CUdevice)>(resolve("cuDeviceGetName"));
        a.cuDeviceTotalMem_v2      = reinterpret_cast<int (*)(size_t*, CUdevice)>(resolve("cuDeviceTotalMem_v2"));
        if (!a.cuDeviceTotalMem_v2)
            a.cuDeviceTotalMem_v2  = reinterpret_cast<int (*)(size_t*, CUdevice)>(resolve("cuDeviceTotalMem"));
        a.cuDeviceGetAttribute     = reinterpret_cast<int (*)(int*, int, CUdevice)>(resolve("cuDeviceGetAttribute"));
        a.cuDevicePrimaryCtxRetain = reinterpret_cast<int (*)(CUcontext*, CUdevice)>(resolve("cuDevicePrimaryCtxRetain"));
        a.cuCtxSetCurrent          = reinterpret_cast<int (*)(CUcontext)>(resolve("cuCtxSetCurrent"));
        a.cuMemAlloc_v2            = reinterpret_cast<int (*)(CUdeviceptr*, size_t)>(resolve("cuMemAlloc_v2"));
        if (!a.cuMemAlloc_v2)
            a.cuMemAlloc_v2 = reinterpret_cast<int (*)(CUdeviceptr*, size_t)>(resolve("cuMemAlloc"));
        a.cuMemFree_v2             = reinterpret_cast<int (*)(CUdeviceptr)>(resolve("cuMemFree_v2"));
        if (!a.cuMemFree_v2)
            a.cuMemFree_v2 = reinterpret_cast<int (*)(CUdeviceptr)>(resolve("cuMemFree"));
        a.cuMemGetInfo_v2          = reinterpret_cast<int (*)(size_t*, size_t*)>(resolve("cuMemGetInfo_v2"));
        if (!a.cuMemGetInfo_v2)
            a.cuMemGetInfo_v2 = reinterpret_cast<int (*)(size_t*, size_t*)>(resolve("cuMemGetInfo"));
        bool core = a.cuInit && a.cuDeviceGetCount && a.cuDeviceGet &&
                    a.cuMemAlloc_v2 && a.cuMemFree_v2 && a.cuMemGetInfo_v2 &&
                    a.cuDevicePrimaryCtxRetain && a.cuCtxSetCurrent;
        if (core && a.cuInit(0) == 0) {
            a.ok = true;
            a.enumOk = a.cuDeviceGetName && a.cuDeviceTotalMem_v2 && a.cuDeviceGetAttribute;
        }
        return a;
    }();
    return api;
}

} // namespace

int cuda_enumerate_count() {
    const CudaApi& api = cudaApi();
    if (!api.ok) return 0;
    int count = 0;
    if (api.cuDeviceGetCount(&count) != 0 || count < 0) return 0;
    return count;
}

bool cuda_runtime_present() {
    const CudaApi& api = cudaApi();
    return api.ok;
}

bool cuda_enumerate_device(int idx, char* nameOut, size_t nameLen,
                           uint64_t* totalMemOut, bool* integratedOut) {
    const CudaApi& api = cudaApi();
    if (!api.enumOk) return false;
    int count = 0;
    if (api.cuDeviceGetCount(&count) != 0 || idx < 0 || idx >= count) return false;
    CUdevice dev = 0;
    if (api.cuDeviceGet(&dev, idx) != 0) return false;
    char name[256] = {};
    if (api.cuDeviceGetName(name, sizeof(name), dev) != 0) return false;
    size_t total = 0;
    if (api.cuDeviceTotalMem_v2(&total, dev) != 0) return false;
    int integrated = 0;
    if (api.cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev) != 0) {
        integrated = 0;
    }
    if (nameOut && nameLen > 0) {
        size_t n = 0;
        while (n + 1 < nameLen && n < sizeof(name) && name[n] != 0) { nameOut[n] = name[n]; ++n; }
        nameOut[n] = 0;
    }
    if (totalMemOut) *totalMemOut = total;
    if (integratedOut) *integratedOut = integrated != 0;
    return true;
}

namespace {

constexpr int kCudaErrorBase = -4000000;  // distinct from Vulkan/HIP/L0 ranges
int asError(int cudaErr) { return encode_backend_error(kCudaErrorBase - cudaErr); }

} // namespace

static CUcontext cudaPrimaryCtx(int deviceIndex) {
    static std::mutex mtx;
    static std::unordered_map<int, CUcontext> ctxs;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = ctxs.find(deviceIndex);
    if (it != ctxs.end()) return it->second;
    const CudaApi& api = cudaApi();
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    if (api.ok && api.cuDeviceGet(&dev, deviceIndex) == 0 &&
        api.cuDevicePrimaryCtxRetain(&ctx, dev) == 0 && ctx) {
        ctxs[deviceIndex] = ctx;
        return ctx;
    }
    return nullptr;
}

static bool cudaUseDevice(int deviceIndex) {
    const CudaApi& api = cudaApi();
    if (!api.ok) return false;
    CUcontext ctx = cudaPrimaryCtx(deviceIndex);
    if (!ctx) return false;
    return api.cuCtxSetCurrent(ctx) == 0;
}

std::unique_ptr<CudaMemoryBackend> CudaMemoryBackend::create(int deviceIndex) {
    const CudaApi& api = cudaApi();
    if (!api.ok) {
        VVM_LOG_WARN("cuda backend: CUDA driver API (libcuda) not loadable");
        return nullptr;
    }
    int count = 0;
    if (api.cuDeviceGetCount(&count) != 0 || count <= 0) {
        VVM_LOG_WARN("cuda backend: no CUDA devices detected");
        return nullptr;
    }
    const int idx = deviceIndex < 0 ? 0 : deviceIndex;
    if (idx >= count) {
        VVM_LOG_WARN("cuda backend: device index {} >= count {}", idx, count);
        return nullptr;
    }
    if (!cudaUseDevice(idx)) {
        VVM_LOG_WARN("cuda backend: primary context setup failed for device {}", idx);
        return nullptr;
    }
    size_t freeBytes = 0, totalBytes = 0;
    if (api.cuMemGetInfo_v2(&freeBytes, &totalBytes) != 0) {
        VVM_LOG_WARN("cuda backend: cuMemGetInfo failed");
        return nullptr;
    }
    auto backend = std::unique_ptr<CudaMemoryBackend>(new CudaMemoryBackend());
    backend->totalMem_ = totalBytes;
    backend->deviceIndex_ = static_cast<int32_t>(idx);
    VVM_LOG_INFO("cuda backend: device {} ({} MB), driver API loaded dynamically",
                 idx, static_cast<uint64_t>(totalBytes) / (1024 * 1024));
    return backend;
}

std::vector<BackendHeap> CudaMemoryBackend::heaps() const {
    return {{totalMem_, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT}};
}

std::vector<BackendMemType> CudaMemoryBackend::memoryTypes() const {
    return {{0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}};
}

BackendBudget CudaMemoryBackend::heapBudget(uint32_t heapIndex) const {
    BackendBudget out{};
    if (heapIndex != 0) return out;
    const CudaApi& api = cudaApi();
    if (!cudaUseDevice(deviceIndex_)) return out;
    size_t freeBytes = 0, totalBytes = 0;
    if (api.ok && api.cuMemGetInfo_v2(&freeBytes, &totalBytes) == 0) {
        out.budgetBytes = totalBytes;
        out.usedBytes   = totalBytes - freeBytes;
        out.valid       = true;
    }
    return out;
}

bool CudaMemoryBackend::type_is_host_visible(uint32_t typeIndex) const {
    (void)typeIndex;
    return false;
}

BackendMemory CudaMemoryBackend::allocate(const BackendAllocRequest& req,
                                          int* resultError) {
    if (resultError) *resultError = 0;
    if (req.typeIndex != 0) {
        if (resultError) *resultError = asError(1);
        return 0;
    }
    const CudaApi& api = cudaApi();
    if (!api.ok || !cudaUseDevice(deviceIndex_)) {
        if (resultError) *resultError = asError(2);
        return 0;
    }
    CUdeviceptr ptr = 0;
    const int rc = api.cuMemAlloc_v2(&ptr, static_cast<size_t>(req.size));
    if (rc != 0 || !ptr) {
        if (resultError) *resultError = asError(rc ? rc : 3);
        return 0;
    }
    return static_cast<uint64_t>(ptr);
}

void CudaMemoryBackend::free(BackendMemory mem) {
    if (mem == 0) return;
    const CudaApi& api = cudaApi();
    if (!cudaUseDevice(deviceIndex_)) return;
    if (api.ok) api.cuMemFree_v2(static_cast<CUdeviceptr>(mem));
}

void* CudaMemoryBackend::map(BackendMemory mem, bool* ok) {
    if (ok) *ok = false;
    (void)mem;
    return nullptr;
}

void CudaMemoryBackend::unmap(BackendMemory mem) {
    (void)mem;
}

BackendBuffer CudaMemoryBackend::create_buffer(BackendMemory mem, uint64_t offset,
                                               uint64_t size, uint64_t usageBits,
                                               bool exportable, int* resultError) {
    if (resultError) *resultError = 0;
    if (mem == 0) {
        if (resultError) *resultError = asError(4);
        return 0;
    }
    (void)size; (void)usageBits; (void)exportable;
    return mem + offset;
}

bool CudaMemoryBackend::bind_buffer(BackendBuffer buf, BackendMemory mem) {
    return buf != 0 && buf == mem;
}

void CudaMemoryBackend::destroy_buffer(BackendBuffer buf) {
    (void)buf;
}

uint64_t CudaMemoryBackend::buffer_device_address(BackendBuffer buf) const {
    return buf;
}

bool CudaMemoryBackend::supports_export(ExternalHandleType type) const {
    (void)type;
    return false;
}

} // namespace vvm

#endif
