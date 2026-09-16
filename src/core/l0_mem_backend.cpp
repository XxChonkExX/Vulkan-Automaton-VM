// ============================================================================
// L0MemoryBackend implementation - dynamic ze_loader + zeMemAllocDevice plane.
// See l0_mem_backend.hpp for the mapping rationale. Struct layouts are
// signature-accurate to ze_api.h v1.18 (verified against the official header).
// ============================================================================

#include "vulkan_vm/l0_mem_backend.hpp"
#include "vulkan_vm/utils.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace vvm {

namespace {

// ---- Minimal Level Zero surface (only the types/functions we call) ----

typedef int32_t ze_result_t;                 // ZE_RESULT_SUCCESS == 0
typedef void* ze_driver_handle_t;
typedef void* ze_device_handle_t;
typedef void* ze_context_handle_t;

constexpr int32_t kZeSuccess = 0;
constexpr uint32_t kZeStypeContextDesc        = 0xd;   // ze_context_desc_t
constexpr uint32_t kZeStypeDeviceMemAllocDesc = 0x15;  // ze_device_mem_alloc_desc_t
constexpr uint32_t kZeStypeDeviceMemProps     = 0x7;   // ze_device_memory_properties_t
constexpr uint32_t kZeStypeDeviceProps        = 0x3;   // ze_device_properties_t (classic)
constexpr uint32_t kZeMaxDeviceName           = 256;

struct ZeContextDesc {
    uint32_t stype;
    const void* pNext;
    uint32_t flags;
};

struct ZeDeviceMemAllocDesc {
    uint32_t stype;
    const void* pNext;
    uint32_t flags;    // 0 = default (cached)
    uint32_t ordinal;  // device-local memory ordinal
};

struct ZeDeviceMemoryProperties {
    uint32_t stype;
    void* pNext;
    uint32_t flags;
    uint32_t maxClockRate;
    uint32_t maxBusWidth;
    uint64_t totalSize;
    char name[kZeMaxDeviceName];
};

struct ZeApi {
    ze_result_t (*zeInit)(uint32_t flags) = nullptr;
    ze_result_t (*zeDriverGet)(uint32_t* pCount, ze_driver_handle_t* phDrivers) = nullptr;
    ze_result_t (*zeDeviceGet)(ze_driver_handle_t hDriver, uint32_t* pCount,
                               ze_device_handle_t* phDevices) = nullptr;
    ze_result_t (*zeContextCreate)(ze_driver_handle_t hDriver, const ZeContextDesc* desc,
                                   ze_context_handle_t* phContext) = nullptr;
    ze_result_t (*zeContextDestroy)(ze_context_handle_t hContext) = nullptr;
    ze_result_t (*zeMemAllocDevice)(ze_context_handle_t hContext,
                                    const ZeDeviceMemAllocDesc* device_desc,
                                    size_t size, size_t alignment,
                                    ze_device_handle_t hDevice, void** pptr) = nullptr;
    ze_result_t (*zeMemFree)(ze_context_handle_t hContext, void* ptr) = nullptr;
    ze_result_t (*zeDeviceGetMemoryProperties)(ze_device_handle_t hDevice, uint32_t* pCount,
                                               ZeDeviceMemoryProperties* pMemProperties) = nullptr;
    ze_result_t (*zeDeviceGetPropertiesRaw)(ze_device_handle_t hDevice, void* props /*4096B*/) = nullptr;
    // Context handle owned by this process's backend instance (resolved from
    // the shared api struct at create()).
    ze_context_handle_t context = nullptr;
    bool ok = false;
    bool enumOk = false;
};

ZeApi g_zeApi = {};

bool loadZeApi() {
    if (g_zeApi.ok || g_zeApi.zeInit) return g_zeApi.ok;
#ifdef _WIN32
    HMODULE m = ::GetModuleHandleW(L"ze_loader.dll");
    if (!m) m = ::LoadLibraryW(L"ze_loader.dll");
    if (!m) return false;
    auto r = [&](const char* n) { return reinterpret_cast<void*>(::GetProcAddress(m, n)); };
#else
    void* m = ::dlopen("libze_loader.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!m) m = ::dlopen("libze_loader.so", RTLD_NOW | RTLD_GLOBAL);
    if (!m) return false;
    auto r = [&](const char* n) { return ::dlsym(m, n); };
#endif
    g_zeApi.zeInit = reinterpret_cast<ze_result_t (*)(uint32_t)>(r("zeInit"));
    g_zeApi.zeDriverGet =
        reinterpret_cast<ze_result_t (*)(uint32_t*, ze_driver_handle_t*)>(r("zeDriverGet"));
    g_zeApi.zeDeviceGet =
        reinterpret_cast<ze_result_t (*)(ze_driver_handle_t, uint32_t*, ze_device_handle_t*)>(r("zeDeviceGet"));
    g_zeApi.zeContextCreate =
        reinterpret_cast<ze_result_t (*)(ze_driver_handle_t, const ZeContextDesc*,
                                         ze_context_handle_t*)>(r("zeContextCreate"));
    g_zeApi.zeContextDestroy =
        reinterpret_cast<ze_result_t (*)(ze_context_handle_t)>(r("zeContextDestroy"));
    g_zeApi.zeMemAllocDevice =
        reinterpret_cast<ze_result_t (*)(ze_context_handle_t, const ZeDeviceMemAllocDesc*,
                                         size_t, size_t, ze_device_handle_t, void**)>(r("zeMemAllocDevice"));
    g_zeApi.zeMemFree =
        reinterpret_cast<ze_result_t (*)(ze_context_handle_t, void*)>(r("zeMemFree"));
    g_zeApi.zeDeviceGetMemoryProperties =
        reinterpret_cast<ze_result_t (*)(ze_driver_handle_t, uint32_t*,
                                         ZeDeviceMemoryProperties*)>(r("zeDeviceGetMemoryProperties"));
    g_zeApi.zeDeviceGetPropertiesRaw =
        reinterpret_cast<ze_result_t (*)(ze_device_handle_t, void*)>(r("zeDeviceGetProperties"));
    g_zeApi.ok = g_zeApi.zeInit && g_zeApi.zeDriverGet && g_zeApi.zeDeviceGet &&
                 g_zeApi.zeContextCreate && g_zeApi.zeMemAllocDevice &&
                 g_zeApi.zeMemFree && g_zeApi.zeDeviceGetMemoryProperties;
    g_zeApi.enumOk = g_zeApi.ok && g_zeApi.zeDeviceGetPropertiesRaw != nullptr;
    return g_zeApi.ok;
}

bool loaderPresent() {
#ifdef _WIN32
    HMODULE m = ::GetModuleHandleW(L"ze_loader.dll");
    if (!m) m = ::LoadLibraryW(L"ze_loader.dll");
    return m != nullptr;
#else
    void* m = ::dlopen("libze_loader.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!m) m = ::dlopen("libze_loader.so", RTLD_NOW | RTLD_GLOBAL);
    return m != nullptr;
#endif
}

} // namespace

int l0_enumerate_count() {
    // Cheap loader-presence probe: the full enumeration (below) re-resolves.
    if (!loaderPresent()) return 0;
    if (!loadZeApi() || !g_zeApi.enumOk) return 0;
    if (g_zeApi.zeInit(0) != kZeSuccess) return 0;
    uint32_t driverCount = 0;
    if (g_zeApi.zeDriverGet(&driverCount, nullptr) != kZeSuccess) return 0;
    int total = 0;
    for (uint32_t d = 0; d < driverCount; ++d) {
        ze_driver_handle_t driver = nullptr;
        uint32_t one = 1;
        if (g_zeApi.zeDriverGet(&one, &driver) != kZeSuccess || !driver) continue;
        uint32_t devCount = 0;
        if (g_zeApi.zeDeviceGet(driver, &devCount, nullptr) == kZeSuccess) {
            total += static_cast<int>(devCount);
        }
    }
    return total;
}

bool l0_runtime_present() {
    return loadZeApi();
}

bool l0_enumerate_device(int idx, char* nameOut, size_t nameLen,
                         uint64_t* totalMemOut, uint32_t* vendorOut,
                         uint32_t* deviceOut, bool* integratedOut) {
    if (!loadZeApi() || !g_zeApi.enumOk) return false;
    if (g_zeApi.zeInit(0) != kZeSuccess) return false;
    uint32_t driverCount = 0;
    if (g_zeApi.zeDriverGet(&driverCount, nullptr) != kZeSuccess) return false;

    // Flat (driver, device) enumeration, same as create().
    struct Pair { ze_driver_handle_t driver; ze_device_handle_t device; };
    std::vector<Pair> all;
    for (uint32_t d = 0; d < driverCount; ++d) {
        ze_driver_handle_t driver = nullptr;
        uint32_t one = 1;
        if (g_zeApi.zeDriverGet(&one, &driver) != kZeSuccess || !driver) continue;
        uint32_t devCount = 0;
        if (g_zeApi.zeDeviceGet(driver, &devCount, nullptr) != kZeSuccess) continue;
        std::vector<ze_device_handle_t> list(devCount);
        if (g_zeApi.zeDeviceGet(driver, &devCount, list.data()) != kZeSuccess) continue;
        for (uint32_t i = 0; i < devCount; ++i) all.push_back({driver, list[i]});
    }
    if (idx < 0 || static_cast<size_t>(idx) >= all.size()) return false;
    ze_device_handle_t device = all[static_cast<size_t>(idx)].device;

    // ze_device_properties_t read through the verified layout (see file
    // header comment): vendorId@20, deviceId@24, flags@28 (INTEGRATED=bit0),
    // name[256]@112. 4 KB zeroed buffer keeps the write-back safe.
    alignas(16) unsigned char props[4096] = {};
    *reinterpret_cast<uint32_t*>(props) = kZeStypeDeviceProps;   // classic stype
    if (g_zeApi.zeDeviceGetPropertiesRaw(device, props) != kZeSuccess) return false;
    const uint32_t vendorId  = *reinterpret_cast<uint32_t*>(props + 20);
    const uint32_t deviceId  = *reinterpret_cast<uint32_t*>(props + 24);
    const uint32_t flags     = *reinterpret_cast<uint32_t*>(props + 28);
    if (nameOut && nameLen > 0) {
        props[112 + 255] = 0;
        size_t n = 0;
        while (n + 1 < nameLen && n < 255 && props[112 + n] != 0) {
            nameOut[n] = static_cast<char>(props[112 + n]);
            ++n;
        }
        nameOut[n] = 0;
    }

    // Largest memory totalSize (totalSize u64 @32 in
    // ze_device_memory_properties_t).
    uint32_t memCount = 0;
    uint64_t total = 0;
    if (g_zeApi.zeDeviceGetMemoryProperties(device, &memCount, nullptr) == kZeSuccess && memCount > 0) {
        std::vector<ZeDeviceMemoryProperties> mp(memCount);
        for (auto& p : mp) { p.stype = kZeStypeDeviceMemProps; p.pNext = nullptr; }
        if (g_zeApi.zeDeviceGetMemoryProperties(device, &memCount, mp.data()) == kZeSuccess) {
            for (const auto& p : mp) {
                if (p.totalSize > total) total = p.totalSize;
            }
        }
    }

    if (totalMemOut) *totalMemOut = total;
    if (vendorOut) *vendorOut = vendorId;
    if (deviceOut) *deviceOut = deviceId;
    if (integratedOut) *integratedOut = (flags & 1u) != 0;
    return true;
}

namespace {

constexpr int kL0ErrorBase = -3000000;   // distinct from Vulkan/HIP ranges
int asError(int32_t zeErr) { return encode_backend_error(kL0ErrorBase - static_cast<int>(zeErr)); }

} // namespace

std::unique_ptr<L0MemoryBackend> L0MemoryBackend::create(int deviceIndex) {
    if (!loadZeApi()) {
        VVM_LOG_WARN("l0 backend: ze_loader.dll not loadable");
        return nullptr;
    }
    if (g_zeApi.zeInit(0 /* ZE_INIT_FLAG_GPU_ONLY not set: allow all drivers */) != kZeSuccess) {
        VVM_LOG_WARN("l0 backend: zeInit failed");
        return nullptr;
    }
    uint32_t driverCount = 0;
    if (g_zeApi.zeDriverGet(&driverCount, nullptr) != kZeSuccess || driverCount == 0) {
        VVM_LOG_WARN("l0 backend: no Level Zero drivers");
        return nullptr;
    }

    // Enumerate (driver, device) pairs across drivers; index into the flat
    // device list but remember which driver owns the selected device.
    std::vector<ze_device_handle_t> devices;
    std::vector<ze_driver_handle_t> owners;
    for (uint32_t d = 0; d < driverCount; ++d) {
        ze_driver_handle_t driver = nullptr;
        uint32_t one = 1;
        if (g_zeApi.zeDriverGet(&one, &driver) != kZeSuccess || !driver) continue;
        uint32_t devCount = 0;
        if (g_zeApi.zeDeviceGet(driver, &devCount, nullptr) != kZeSuccess) continue;
        std::vector<ze_device_handle_t> list(devCount);
        if (g_zeApi.zeDeviceGet(driver, &devCount, list.data()) != kZeSuccess) continue;
        devices.insert(devices.end(), list.begin(), list.begin() + devCount);
        owners.insert(owners.end(), devCount, driver);
    }
    if (devices.empty()) {
        VVM_LOG_WARN("l0 backend: no Level Zero devices");
        return nullptr;
    }
    const int idx = deviceIndex < 0 ? 0 : deviceIndex;
    if (static_cast<size_t>(idx) >= devices.size()) {
        VVM_LOG_WARN("l0 backend: device index {} >= count {}", idx, devices.size());
        return nullptr;
    }
    ze_device_handle_t device = devices[static_cast<size_t>(idx)];
    ze_driver_handle_t driver = owners[static_cast<size_t>(idx)];

    // Context (per create; the process-wide g_zeApi holds the loader only).
    ZeContextDesc ctxDesc{kZeStypeContextDesc, nullptr, 0};
    ze_context_handle_t context = nullptr;
    if (g_zeApi.zeContextCreate(driver, &ctxDesc, &context) != kZeSuccess) {
        VVM_LOG_WARN("l0 backend: zeContextCreate failed");
        return nullptr;
    }

    // Memory properties: use the largest memory ordinal's totalSize.
    uint32_t memCount = 0;
    if (g_zeApi.zeDeviceGetMemoryProperties(device, &memCount, nullptr) != kZeSuccess || memCount == 0) {
        VVM_LOG_WARN("l0 backend: zeDeviceGetMemoryProperties returned no memories");
        return nullptr;
    }
    std::vector<ZeDeviceMemoryProperties> props(memCount);
    for (auto& p : props) {
        p.stype = kZeStypeDeviceMemProps;
        p.pNext = nullptr;
    }
    if (g_zeApi.zeDeviceGetMemoryProperties(device, &memCount, props.data()) != kZeSuccess) {
        VVM_LOG_WARN("l0 backend: zeDeviceGetMemoryProperties query failed");
        return nullptr;
    }
    uint64_t total = 0;
    const char* memName = "";
    for (const auto& p : props) {
        if (p.totalSize > total) {
            total = p.totalSize;
            memName = p.name;
        }
    }

    auto backend = std::unique_ptr<L0MemoryBackend>(new L0MemoryBackend());
    backend->totalMem_ = total;
    // Store the context/device in the shared api (single instance per
    // process in v1; multiple L0 pools would need per-instance storage).
    g_zeApi.context = context;
    backend->context_ = reinterpret_cast<uint64_t>(context);
    backend->device_ = reinterpret_cast<uint64_t>(device);
    std::snprintf(backend->deviceName_, sizeof(backend->deviceName_), "%s", memName);
    VVM_LOG_INFO("l0 backend: device {} '{}' ({} MB), ze_loader loaded dynamically",
                 idx, memName, total / (1024 * 1024));
    return backend;
}

std::vector<BackendHeap> L0MemoryBackend::heaps() const {
    return {{totalMem_, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT}};
}

std::vector<BackendMemType> L0MemoryBackend::memoryTypes() const {
    return {{0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}};
}

BackendBudget L0MemoryBackend::heapBudget(uint32_t heapIndex) const {
    // v1: no live budget (Sysman zesDeviceGetMemoryState follow-up). The
    // pool's wouldExceedBudget falls back to the static heap size.
    (void)heapIndex;
    return {};
}

bool L0MemoryBackend::type_is_host_visible(uint32_t typeIndex) const {
    (void)typeIndex;
    return false;   // zeMemAllocDevice memory is device-only in v1
}

BackendMemory L0MemoryBackend::allocate(const BackendAllocRequest& req,
                                        int* resultError) {
    if (resultError) *resultError = 0;
    if (req.typeIndex != 0) {
        if (resultError) *resultError = asError(1);
        return 0;
    }
    if (!g_zeApi.ok || !g_zeApi.context) {
        if (resultError) *resultError = asError(2);
        return 0;
    }
    ZeDeviceMemAllocDesc desc{kZeStypeDeviceMemAllocDesc, nullptr, 0, 0};
    void* ptr = nullptr;
    const ze_result_t rc = g_zeApi.zeMemAllocDevice(
        g_zeApi.context, &desc, static_cast<size_t>(req.size),
        0 /* driver-default alignment */, reinterpret_cast<ze_device_handle_t>(device_),
        &ptr);
    if (rc != kZeSuccess || !ptr) {
        if (resultError) *resultError = asError(rc ? rc : 3);
        return 0;
    }
    return reinterpret_cast<uint64_t>(ptr);
}

void L0MemoryBackend::free(BackendMemory mem) {
    if (mem == 0 || !g_zeApi.context) return;
    g_zeApi.zeMemFree(g_zeApi.context, reinterpret_cast<void*>(mem));
}

void* L0MemoryBackend::map(BackendMemory mem, bool* ok) {
    if (ok) *ok = false;
    (void)mem;
    return nullptr;   // device memory; zeMemAllocHost is the v2 host path
}

void L0MemoryBackend::unmap(BackendMemory mem) {
    (void)mem;
}

BackendBuffer L0MemoryBackend::create_buffer(BackendMemory mem, uint64_t offset,
                                             uint64_t size, uint64_t usageBits,
                                             bool exportable, int* resultError) {
    // A Level Zero device pointer IS the buffer (echo semantics).
    (void)size; (void)usageBits; (void)exportable;
    if (resultError) *resultError = 0;
    if (offset != 0) {
        if (resultError) *resultError = asError(4);
        return 0;
    }
    return mem;
}

bool L0MemoryBackend::bind_buffer(BackendBuffer buf, BackendMemory mem) {
    return buf != 0 && buf == mem;
}

void L0MemoryBackend::destroy_buffer(BackendBuffer buf) {
    (void)buf;   // ownership follows the memory handle
}

uint64_t L0MemoryBackend::buffer_device_address(BackendBuffer buf) const {
    return buf;
}

bool L0MemoryBackend::supports_export(ExternalHandleType type) const {
    (void)type;
    return false;   // win32 handle export/import is a v2 concern
}

} // namespace vvm
