// ============================================================================
// HipMemoryBackend implementation - dynamic amdhip64 loader + hipMalloc plane.
// See hip_mem_backend.hpp for the mapping rationale.
// ============================================================================

#include "vulkan_vm/hip_mem_backend.hpp"
#include "vulkan_vm/utils.hpp"

#ifdef _WIN32
#include <windows.h>

namespace vvm {

namespace {

// Minimal HIP runtime surface, resolved dynamically. Signature-accurate to
// the HIP API; only what the memory plane needs.
struct HipApi {
    int  (*hipMalloc)(void** ptr, size_t size) = nullptr;
    int  (*hipFree)(void* ptr) = nullptr;
    int  (*hipMemGetInfo)(size_t* freeBytes, size_t* totalBytes) = nullptr;
    int  (*hipSetDevice)(int device) = nullptr;
    int  (*hipGetDeviceCount)(int* count) = nullptr;
    int  (*hipDeviceTotalMem)(size_t* bytes, int device) = nullptr;
    int  (*hipGetDeviceProperties)(void* prop /* >1KB zeroed buffer */, int device) = nullptr;
    int  (*hipDeviceGetAttribute)(int* pi, int attrib, int device) = nullptr;
    bool ok = false;
    bool enumOk = false;
};

const int kHipAttrIntegrated = 18;   // hipDeviceAttributeIntegrated (CUDA-compatible numbering)

const HipApi& hipApi() {
    static HipApi api = [] {
        HipApi a{};
        HMODULE m = ::GetModuleHandleW(L"amdhip64_7.dll");
        if (!m) m = ::LoadLibraryW(L"amdhip64_7.dll");
        if (!m) m = ::LoadLibraryW(L"amdhip64.dll");
        if (!m) return a;
        auto resolve = [&](const char* name) {
            return reinterpret_cast<void*>(::GetProcAddress(m, name));
        };
        a.hipMalloc         = reinterpret_cast<int (*)(void**, size_t)>(resolve("hipMalloc"));
        a.hipFree           = reinterpret_cast<int (*)(void*)>(resolve("hipFree"));
        a.hipMemGetInfo     = reinterpret_cast<int (*)(size_t*, size_t*)>(resolve("hipMemGetInfo"));
        a.hipSetDevice      = reinterpret_cast<int (*)(int)>(resolve("hipSetDevice"));
        a.hipGetDeviceCount = reinterpret_cast<int (*)(int*)>(resolve("hipGetDeviceCount"));
        a.hipDeviceTotalMem = reinterpret_cast<int (*)(size_t*, int)>(resolve("hipDeviceTotalMem"));
        a.hipGetDeviceProperties = reinterpret_cast<int (*)(void*, int)>(resolve("hipGetDeviceProperties"));
        a.hipDeviceGetAttribute  = reinterpret_cast<int (*)(int*, int, int)>(resolve("hipDeviceGetAttribute"));
        a.ok = a.hipMalloc && a.hipFree && a.hipMemGetInfo && a.hipSetDevice;
        a.enumOk = a.ok && a.hipDeviceTotalMem && a.hipGetDeviceProperties && a.hipDeviceGetAttribute;
        return a;
    }();
    return api;
}

} // namespace

int hip_enumerate_count() {
    const HipApi& api = hipApi();
    if (!api.ok) return 0;
    int count = 0;
    if (api.hipGetDeviceCount(&count) != 0 || count < 0) return 0;
    return count;
}

bool hip_runtime_present() {
    const HipApi& api = hipApi();
    return api.ok;
}

bool hip_enumerate_device(int idx, char* nameOut, size_t nameLen,
                          uint64_t* totalMemOut, bool* integratedOut) {
    const HipApi& api = hipApi();
    if (!api.enumOk) return false;
    int count = 0;
    if (api.hipGetDeviceCount(&count) != 0 || idx < 0 || idx >= count) return false;
    // hipDeviceProp_t: name[256] at offset 0 (CUDA-compatible layout; we
    // only read the name, from a 4 KB zeroed buffer so the write-back is
    // always safe even if the struct is smaller).
    alignas(16) unsigned char prop[4096] = {};
    if (api.hipGetDeviceProperties(prop, idx) != 0) return false;
    size_t total = 0;
    if (api.hipDeviceTotalMem(&total, idx) != 0) return false;
    int integrated = 0;
    if (api.hipDeviceGetAttribute(&integrated, kHipAttrIntegrated, idx) != 0) {
        integrated = 0;
    }
    if (nameOut && nameLen > 0) {
        prop[255] = 0;
        size_t n = 0;
        while (n + 1 < nameLen && n < 255 && prop[n] != 0) { nameOut[n] = (char)prop[n]; ++n; }
        nameOut[n] = 0;
    }
    if (totalMemOut) *totalMemOut = total;
    if (integratedOut) *integratedOut = integrated != 0;
    return true;
}

namespace {

constexpr int kHipErrorBase = -2000000;   // distinct from the Vulkan range
int asError(int hipErr) { return encode_backend_error(kHipErrorBase - hipErr); }

} // namespace

std::unique_ptr<HipMemoryBackend> HipMemoryBackend::create(int deviceIndex) {
    const HipApi& api = hipApi();
    if (!api.ok) {
        VVM_LOG_WARN("hip backend: AMD HIP runtime (amdhip64_7.dll) not loadable");
        return nullptr;
    }
    int count = 0;
    if (api.hipGetDeviceCount(&count) != 0 || count <= 0) {
        VVM_LOG_WARN("hip backend: no HIP devices detected");
        return nullptr;
    }
    const int idx = deviceIndex < 0 ? 0 : deviceIndex;
    if (idx >= count) {
        VVM_LOG_WARN("hip backend: device index {} >= count {}", idx, count);
        return nullptr;
    }
    if (api.hipSetDevice(idx) != 0) {
        VVM_LOG_WARN("hip backend: hipSetDevice({}) failed", idx);
        return nullptr;
    }
    size_t freeBytes = 0, totalBytes = 0;
    if (api.hipMemGetInfo(&freeBytes, &totalBytes) != 0) {
        VVM_LOG_WARN("hip backend: hipMemGetInfo failed");
        return nullptr;
    }
    auto backend = std::unique_ptr<HipMemoryBackend>(new HipMemoryBackend());
    backend->totalMem_ = totalBytes;
    VVM_LOG_INFO("hip backend: device {} ({} MB), runtime loaded dynamically",
                 idx, static_cast<uint64_t>(totalBytes) / (1024 * 1024));
    return backend;
}

std::vector<BackendHeap> HipMemoryBackend::heaps() const {
    // Synthetic model: one DEVICE_LOCAL heap covering all device memory.
    return {{totalMem_, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT}};
}

std::vector<BackendMemType> HipMemoryBackend::memoryTypes() const {
    // Synthetic model: type 0 = DEVICE_LOCAL device pointers.
    return {{0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}};
}

BackendBudget HipMemoryBackend::heapBudget(uint32_t heapIndex) const {
    BackendBudget out{};
    if (heapIndex != 0) return out;
    const HipApi& api = hipApi();
    size_t freeBytes = 0, totalBytes = 0;
    if (api.ok && api.hipMemGetInfo(&freeBytes, &totalBytes) == 0) {
        out.budgetBytes = totalBytes;
        out.usedBytes   = totalBytes - freeBytes;
        out.valid       = true;
    }
    return out;
}

bool HipMemoryBackend::type_is_host_visible(uint32_t typeIndex) const {
    // Device memory on HIP is not host-mappable; type 0 is device-local only.
    (void)typeIndex;
    return false;
}

BackendMemory HipMemoryBackend::allocate(const BackendAllocRequest& req,
                                         int* resultError) {
    if (resultError) *resultError = 0;
    if (req.typeIndex != 0) {
        if (resultError) *resultError = asError(1);
        return 0;
    }
    const HipApi& api = hipApi();
    if (!api.ok) {
        if (resultError) *resultError = asError(2);
        return 0;
    }
    void* ptr = nullptr;
    const int rc = api.hipMalloc(&ptr, static_cast<size_t>(req.size));
    if (rc != 0 || !ptr) {
        if (resultError) *resultError = asError(rc ? rc : 3);
        return 0;
    }
    // deviceAddress/pointer priority/dedicated hints are HIP no-ops: a device
    // pointer IS the address, and allocations are standalone.
    return reinterpret_cast<uint64_t>(ptr);
}

void HipMemoryBackend::free(BackendMemory mem) {
    if (mem == 0) return;
    const HipApi& api = hipApi();
    if (api.ok) api.hipFree(reinterpret_cast<void*>(mem));
}

void* HipMemoryBackend::map(BackendMemory mem, bool* ok) {
    // HIP device memory is not host-mappable; host-visible staging is v2.
    if (ok) *ok = false;
    (void)mem;
    return nullptr;
}

void HipMemoryBackend::unmap(BackendMemory mem) {
    (void)mem;
}

BackendBuffer HipMemoryBackend::create_buffer(BackendMemory mem, uint64_t offset,
                                              uint64_t size, uint64_t usageBits,
                                              bool exportable, int* resultError) {
    // A HIP device pointer IS the buffer. Allocations are standalone:
    // mem must match (no sub-allocation offset form in v1).
    (void)size; (void)usageBits; (void)exportable;
    if (resultError) *resultError = 0;
    if (mem == 0 || (mem + offset) != mem) {
        // offset must be 0 for standalone allocations
        if (offset != 0 && resultError) *resultError = asError(4);
        return offset == 0 ? mem : 0;
    }
    return mem;
}

bool HipMemoryBackend::bind_buffer(BackendBuffer buf, BackendMemory mem) {
    // Nothing to bind: pointer == buffer == memory.
    return buf != 0 && buf == mem;
}

void HipMemoryBackend::destroy_buffer(BackendBuffer buf) {
    // Ownership follows the memory handle (allocate/free); buffer destroy is
    // a no-op so double-destroy of the echoed pointer cannot double-free.
    (void)buf;
}

uint64_t HipMemoryBackend::buffer_device_address(BackendBuffer buf) const {
    return buf;   // a HIP device pointer IS the address
}

bool HipMemoryBackend::supports_export(ExternalHandleType type) const {
    // Cross-API export/import on Windows HIP (hipExternalMemory_t) is a v2
    // concern; standalone allocations need nothing here.
    (void)type;
    return false;
}

} // namespace vvm

#else
// ============================================================================
// Non-Windows: dlopen-based amdhip64 loader (same surface as the Windows
// branch; the HIP runtime is loaded dynamically - ROCm installs to
// /opt/rocm/lib with ldconfig, or set LD_LIBRARY_PATH).
// ============================================================================

#include "vulkan_vm/hip_mem_backend.hpp"
#include "vulkan_vm/utils.hpp"

#include <dlfcn.h>

namespace vvm {

namespace {

// Same minimal HIP runtime surface as the Windows branch.
struct HipApi {
    int  (*hipMalloc)(void** ptr, size_t size) = nullptr;
    int  (*hipFree)(void* ptr) = nullptr;
    int  (*hipMemGetInfo)(size_t* freeBytes, size_t* totalBytes) = nullptr;
    int  (*hipSetDevice)(int device) = nullptr;
    int  (*hipGetDeviceCount)(int* count) = nullptr;
    int  (*hipDeviceTotalMem)(size_t* bytes, int device) = nullptr;
    int  (*hipGetDeviceProperties)(void* prop, int device) = nullptr;
    int  (*hipDeviceGetAttribute)(int* pi, int attrib, int device) = nullptr;
    bool ok = false;
    bool enumOk = false;
};

const int kHipAttrIntegrated = 18;

const HipApi& hipApi() {
    static HipApi api = [] {
        HipApi a{};
        // ROCm installs to /opt/rocm/lib with ldconfig registration; plain
        // dlopen picks it up, with an explicit fallback for bare setups.
        void* m = ::dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
        if (!m) m = ::dlopen("/opt/rocm/lib/libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
        if (!m) return a;
        auto resolve = [&](const char* name) {
            return reinterpret_cast<void*>(::dlsym(m, name));
        };
        a.hipMalloc              = reinterpret_cast<int (*)(void**, size_t)>(resolve("hipMalloc"));
        a.hipFree                = reinterpret_cast<int (*)(void*)>(resolve("hipFree"));
        a.hipMemGetInfo          = reinterpret_cast<int (*)(size_t*, size_t*)>(resolve("hipMemGetInfo"));
        a.hipSetDevice           = reinterpret_cast<int (*)(int)>(resolve("hipSetDevice"));
        a.hipGetDeviceCount      = reinterpret_cast<int (*)(int*)>(resolve("hipGetDeviceCount"));
        a.hipDeviceTotalMem      = reinterpret_cast<int (*)(size_t*, int)>(resolve("hipDeviceTotalMem"));
        a.hipGetDeviceProperties = reinterpret_cast<int (*)(void*, int)>(resolve("hipGetDeviceProperties"));
        a.hipDeviceGetAttribute  = reinterpret_cast<int (*)(int*, int, int)>(resolve("hipDeviceGetAttribute"));
        a.ok = a.hipMalloc && a.hipFree && a.hipMemGetInfo && a.hipSetDevice;
        a.enumOk = a.ok && a.hipDeviceTotalMem && a.hipGetDeviceProperties && a.hipDeviceGetAttribute;
        return a;
    }();
    return api;
}

constexpr int kHipErrorBase = -2000000;
int asError(int hipErr) { return encode_backend_error(kHipErrorBase - hipErr); }

} // namespace

int hip_enumerate_count() {
    const HipApi& api = hipApi();
    if (!api.ok) return 0;
    int count = 0;
    if (api.hipGetDeviceCount(&count) != 0 || count < 0) return 0;
    return count;
}

bool hip_runtime_present() {
    const HipApi& api = hipApi();
    return api.ok;
}

bool hip_enumerate_device(int idx, char* nameOut, size_t nameLen,
                          uint64_t* totalMemOut, bool* integratedOut) {
    const HipApi& api = hipApi();
    if (!api.enumOk) return false;
    int count = 0;
    if (api.hipGetDeviceCount(&count) != 0 || idx < 0 || idx >= count) return false;
    // hipDeviceProp_t: name[256] at offset 0 (CUDA-compatible layout; we
    // only read the name, from a 4 KB zeroed buffer so the write-back is
    // always safe even if the struct is smaller).
    alignas(16) unsigned char prop[4096] = {};
    if (api.hipGetDeviceProperties(prop, idx) != 0) return false;
    size_t total = 0;
    if (api.hipDeviceTotalMem(&total, idx) != 0) return false;
    int integrated = 0;
    if (api.hipDeviceGetAttribute(&integrated, kHipAttrIntegrated, idx) != 0) {
        integrated = 0;
    }
    if (nameOut && nameLen > 0) {
        prop[255] = 0;
        size_t n = 0;
        while (n + 1 < nameLen && n < 255 && prop[n] != 0) { nameOut[n] = (char)prop[n]; ++n; }
        nameOut[n] = 0;
    }
    if (totalMemOut) *totalMemOut = total;
    if (integratedOut) *integratedOut = integrated != 0;
    return true;
}

std::unique_ptr<HipMemoryBackend> HipMemoryBackend::create(int deviceIndex) {
    const HipApi& api = hipApi();
    if (!api.ok) {
        VVM_LOG_WARN("hip backend: AMD HIP runtime (libamdhip64.so) not loadable");
        return nullptr;
    }
    int count = 0;
    if (api.hipGetDeviceCount(&count) != 0 || count <= 0) {
        VVM_LOG_WARN("hip backend: no HIP devices detected");
        return nullptr;
    }
    const int idx = deviceIndex < 0 ? 0 : deviceIndex;
    if (idx >= count) {
        VVM_LOG_WARN("hip backend: device index {} >= count {}", idx, count);
        return nullptr;
    }
    if (api.hipSetDevice(idx) != 0) {
        VVM_LOG_WARN("hip backend: hipSetDevice({}) failed", idx);
        return nullptr;
    }
    size_t freeBytes = 0, totalBytes = 0;
    if (api.hipMemGetInfo(&freeBytes, &totalBytes) != 0) {
        VVM_LOG_WARN("hip backend: hipMemGetInfo failed");
        return nullptr;
    }
    auto backend = std::unique_ptr<HipMemoryBackend>(new HipMemoryBackend());
    backend->totalMem_ = totalBytes;
    VVM_LOG_INFO("hip backend: device {} ({} MB), runtime loaded dynamically",
                 idx, static_cast<uint64_t>(totalBytes) / (1024 * 1024));
    return backend;
}

std::vector<BackendHeap> HipMemoryBackend::heaps() const {
    // Synthetic model: one DEVICE_LOCAL heap covering all device memory.
    return {{totalMem_, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT}};
}

std::vector<BackendMemType> HipMemoryBackend::memoryTypes() const {
    // Synthetic model: type 0 = DEVICE_LOCAL device pointers.
    return {{0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}};
}

BackendBudget HipMemoryBackend::heapBudget(uint32_t heapIndex) const {
    BackendBudget out{};
    if (heapIndex != 0) return out;
    const HipApi& api = hipApi();
    size_t freeBytes = 0, totalBytes = 0;
    if (api.ok && api.hipMemGetInfo(&freeBytes, &totalBytes) == 0) {
        out.budgetBytes = totalBytes;
        out.usedBytes   = totalBytes - freeBytes;
        out.valid       = true;
    }
    return out;
}

bool HipMemoryBackend::type_is_host_visible(uint32_t typeIndex) const {
    // Device memory on HIP is not host-mappable; type 0 is device-local only.
    (void)typeIndex;
    return false;
}

BackendMemory HipMemoryBackend::allocate(const BackendAllocRequest& req,
                                         int* resultError) {
    if (resultError) *resultError = 0;
    if (req.typeIndex != 0) {
        if (resultError) *resultError = asError(1);
        return 0;
    }
    const HipApi& api = hipApi();
    if (!api.ok) {
        if (resultError) *resultError = asError(2);
        return 0;
    }
    void* ptr = nullptr;
    const int rc = api.hipMalloc(&ptr, static_cast<size_t>(req.size));
    if (rc != 0 || !ptr) {
        if (resultError) *resultError = asError(rc ? rc : 3);
        return 0;
    }
    // deviceAddress/pointer priority/dedicated hints are HIP no-ops: a device
    // pointer IS the address, and allocations are standalone.
    return reinterpret_cast<uint64_t>(ptr);
}

void HipMemoryBackend::free(BackendMemory mem) {
    if (mem == 0) return;
    const HipApi& api = hipApi();
    if (api.ok) api.hipFree(reinterpret_cast<void*>(mem));
}

void* HipMemoryBackend::map(BackendMemory mem, bool* ok) {
    // HIP device memory is not host-mappable; host-visible staging is v2.
    if (ok) *ok = false;
    (void)mem;
    return nullptr;
}

void HipMemoryBackend::unmap(BackendMemory mem) {
    (void)mem;
}

BackendBuffer HipMemoryBackend::create_buffer(BackendMemory mem, uint64_t offset,
                                              uint64_t size, uint64_t usageBits,
                                              bool exportable, int* resultError) {
    // A HIP device pointer IS the buffer. Allocations are standalone:
    // mem must match (no sub-allocation offset form in v1).
    (void)size; (void)usageBits; (void)exportable;
    if (resultError) *resultError = 0;
    if (mem == 0 || offset != 0) {
        if (offset != 0 && resultError) *resultError = asError(4);
        return offset == 0 ? mem : 0;
    }
    return mem;
}

bool HipMemoryBackend::bind_buffer(BackendBuffer buf, BackendMemory mem) {
    // Nothing to bind: pointer == buffer == memory.
    return buf != 0 && buf == mem;
}

void HipMemoryBackend::destroy_buffer(BackendBuffer) {}

uint64_t HipMemoryBackend::buffer_device_address(BackendBuffer buf) const {
    // A HIP device pointer IS the device address (when BDA is in play).
    return buf;
}

bool HipMemoryBackend::supports_export(ExternalHandleType) const {
    // Cross-API export/import on Linux HIP (hipExternalMemory_t / DMA-BUF)
    // is a v2 concern; standalone allocations need nothing here.
    return false;
}

} // namespace vvm

#endif  // _WIN32

