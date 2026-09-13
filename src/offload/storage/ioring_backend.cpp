#include "vulkan_vm/storage/ioring_backend.hpp"

#include <cstdlib>
#include <cstring>

// Dynamic loading keeps this TU linkable on any SDK and degradable on any OS.
// Types come from the SDK header when present (NTDDI_WIN10_CO+).
#if defined(VVM_PLATFORM_WINDOWS) && defined(__has_include)
#  if __has_include(<ioringapi.h>)
#    define VVM_HAS_IORING_HEADER 1
#  endif
#endif

#ifdef VVM_PLATFORM_WINDOWS

#ifdef VVM_HAS_IORING_HEADER
#include <ioringapi.h>
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace vvm {
namespace storage {
namespace backend {

constexpr uint64_t kAlign = 4096; // NO_BUFFERING sector alignment

// ---------------------------------------------------------------------------
// Dynamic IoRing entry points (types from ioringapi.h when available).
// ---------------------------------------------------------------------------
#ifdef VVM_HAS_IORING_HEADER
using PFN_CreateIoRing = HRESULT(WINAPI*)(IORING_VERSION, IORING_CREATE_FLAGS, UINT32, UINT32, HIORING*);
using PFN_SubmitIoRing = HRESULT(WINAPI*)(HIORING, UINT32, UINT32, UINT32*);
using PFN_PopIoRingCompletion = HRESULT(WINAPI*)(HIORING, IORING_CQE*);
using PFN_BuildIoRingReadFile = HRESULT(WINAPI*)(HIORING, IORING_HANDLE_REF, IORING_BUFFER_REF, UINT32, UINT64, UINT_PTR, IORING_SQE_FLAGS);
using PFN_BuildIoRingRegisterBuffers = HRESULT(WINAPI*)(HIORING, UINT32, IORING_BUFFER_INFO const*, UINT_PTR);
using PFN_CloseIoRing = void(WINAPI*)(HIORING);
using PFN_QueryIoRingCapabilities = HRESULT(WINAPI*)(IORING_CAPABILITIES*);

struct IoRingApi {
    PFN_CreateIoRing create = nullptr;
    PFN_SubmitIoRing submit = nullptr;
    PFN_PopIoRingCompletion pop = nullptr;
    PFN_BuildIoRingReadFile buildRead = nullptr;
    PFN_BuildIoRingRegisterBuffers buildRegister = nullptr;
    PFN_CloseIoRing close = nullptr;
    PFN_QueryIoRingCapabilities queryCaps = nullptr;

    static const IoRingApi& get() {
        static const IoRingApi api = [] {
            IoRingApi a;
            HMODULE m = ::GetModuleHandleW(L"kernelbase.dll");
            if (!m) m = ::GetModuleHandleW(L"kernel32.dll");
            if (!m) return a;
            a.create = reinterpret_cast<PFN_CreateIoRing>(::GetProcAddress(m, "CreateIoRing"));
            a.submit = reinterpret_cast<PFN_SubmitIoRing>(::GetProcAddress(m, "SubmitIoRing"));
            a.pop = reinterpret_cast<PFN_PopIoRingCompletion>(::GetProcAddress(m, "PopIoRingCompletion"));
            a.buildRead = reinterpret_cast<PFN_BuildIoRingReadFile>(::GetProcAddress(m, "BuildIoRingReadFile"));
            a.buildRegister = reinterpret_cast<PFN_BuildIoRingRegisterBuffers>(::GetProcAddress(m, "BuildIoRingRegisterBuffers"));
            a.close = reinterpret_cast<PFN_CloseIoRing>(::GetProcAddress(m, "CloseIoRing"));
            a.queryCaps = reinterpret_cast<PFN_QueryIoRingCapabilities>(::GetProcAddress(m, "QueryIoRingCapabilities"));
            return a;
        }();
        return api;
    }
};
#endif // VVM_HAS_IORING_HEADER

// ---------------------------------------------------------------------------
IoRingBackend::IoRingBackend(BackendConfig cfg) : cfg_(std::move(cfg)) {}

IoRingBackend::~IoRingBackend() { close(); }

void* IoRingBackend::slotPtr(uint32_t slot) const {
    if (!arena_ || slot >= cfg_.slotCount) return nullptr;
    return static_cast<char*>(arena_) + static_cast<size_t>(slot) * cfg_.slotBytes;
}

void IoRingBackend::destroyRing() {
#ifdef VVM_HAS_IORING_HEADER
    if (ring_ && IoRingApi::get().close) {
        IoRingApi::get().close(static_cast<HIORING>(ring_));
    }
#endif
    ring_ = nullptr;
}

void IoRingBackend::degradeToOverlapped() {
    destroyRing();
    ioringMode_ = false;
}

Result IoRingBackend::open() {
    if (open_) return Result::success();
    if (!cfg_.validate()) return Result::error(ErrorCode::InvalidConfig, "bad backend config");

    // 1) Arena: VirtualAlloc is 64 KiB-aligned; slots are contiguous.
    SIZE_T bytes = static_cast<SIZE_T>(cfg_.slotCount) * cfg_.slotBytes;
    arena_ = ::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!arena_) return Result::error(ErrorCode::AllocationFailed, "VirtualAlloc failed");

    // 2) Pack file: NO_BUFFERING (O_DIRECT semantics) + OVERLAPPED.
    int wlen = ::MultiByteToWideChar(CP_UTF8, 0, cfg_.packPath.c_str(), -1, nullptr, 0);
    if (wlen <= 0) { close(); return Result::error(ErrorCode::InvalidConfig, "bad path"); }
    std::wstring wpath(static_cast<size_t>(wlen), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, cfg_.packPath.c_str(), -1, wpath.data(), wlen);

    file_ = ::CreateFileW(wpath.c_str(), GENERIC_READ | GENERIC_WRITE,
                          0, nullptr, OPEN_ALWAYS,
                          FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) {
        file_ = nullptr;
        close();
        return Result::error(ErrorCode::InvalidConfig, "CreateFileW failed: " + cfg_.packPath);
    }

#ifdef VVM_HAS_IORING_HEADER
    // 3) Probe IoRing: kernel support only (UM emulation is not a win).
    const IoRingApi& api = IoRingApi::get();
    bool usable = cfg_.allowIoRing && api.create && api.submit && api.pop
               && api.buildRead && api.buildRegister && api.close && api.queryCaps;
    if (usable) {
        IORING_CAPABILITIES caps{};
        if (FAILED(api.queryCaps(&caps))) usable = false;
        if (usable && (caps.FeatureFlags & IORING_FEATURE_UM_EMULATION)) usable = false;
    }

    if (usable) {
        IORING_CREATE_FLAGS flags{};
        flags.Required = IORING_CREATE_REQUIRED_FLAGS_NONE;
        flags.Advisory = IORING_CREATE_ADVISORY_FLAGS_NONE;

        HIORING h = nullptr;
        // v2 fixes a completion-event race; v1 as fallback.
        HRESULT hr = api.create(IORING_VERSION_2, flags, cfg_.queueDepth, cfg_.queueDepth, &h);
        if (FAILED(hr)) hr = api.create(IORING_VERSION_1, flags, cfg_.queueDepth, cfg_.queueDepth, &h);
        if (FAILED(hr)) {
            degradeToOverlapped();
        } else {
            ring_ = h;
            // Register the staging arena (one IORING_BUFFER_INFO per slot).
            std::vector<IORING_BUFFER_INFO> infos(cfg_.slotCount);
            for (uint32_t s = 0; s < cfg_.slotCount; ++s) {
                infos[s].Address = slotPtr(s);
                infos[s].Length = cfg_.slotBytes;
            }
            hr = api.buildRegister(static_cast<HIORING>(ring_),
                                   static_cast<UINT32>(infos.size()), infos.data(),
                                   /*userData=*/0);
            if (SUCCEEDED(hr)) {
                UINT32 submitted = 0;
                hr = api.submit(static_cast<HIORING>(ring_), 0, 0, &submitted);
            }
            // Drain until the registration CQE arrives (link semantics: any
            // read referencing a buffer index waits for this to complete).
            bool registered = SUCCEEDED(hr);
            if (registered) {
                for (int spin = 0; spin < 100000 && registered; ++spin) {
                    IORING_CQE cqe{};
                    HRESULT pr = api.pop(static_cast<HIORING>(ring_), &cqe);
                    if (pr == S_OK) {
                        registered = SUCCEEDED(cqe.ResultCode);
                        break;
                    }
                    if (pr != S_FALSE) { registered = false; break; }
                    ::Sleep(0);
                }
            }
            if (registered) {
                ioringMode_ = true;
            } else {
                degradeToOverlapped();
            }
        }
    } else {
        degradeToOverlapped();
    }
#else
    ioringMode_ = false; // SDK without ioringapi.h: OVERLAPPED floor only
#endif

    open_ = true;
    return Result::success();
}

void IoRingBackend::close() {
    if (!open_ && !arena_ && !file_ && !ring_) return;

    // Drain outstanding overlapped requests before destroying anything.
    for (auto& p : pending_) {
        DWORD n = 0;
        ::CancelIoEx(static_cast<HANDLE>(file_), &p.ov);
        ::GetOverlappedResult(static_cast<HANDLE>(file_), &p.ov, &n, TRUE);
    }
    pending_.clear();

    destroyRing();

    if (file_) { ::CloseHandle(static_cast<HANDLE>(file_)); file_ = nullptr; }
    if (arena_) { ::VirtualFree(arena_, 0, MEM_RELEASE); arena_ = nullptr; }
    inFlight_ = 0;
    open_ = false;
}

// ---------------------------------------------------------------------------
uint32_t IoRingBackend::submitBatch(const std::vector<IORequest>& batch) {
    if (!open_) return 0;

    uint32_t accepted = 0;
#ifdef VVM_HAS_IORING_HEADER
    const IoRingApi& api = IoRingApi::get();
#endif

    for (const auto& r : batch) {
        // O_DIRECT alignment discipline (4 KiB offset + size).
        if (r.size == 0 || r.dstSlot >= cfg_.slotCount
            || (r.fileOffset % kAlign) != 0 || (r.size % kAlign) != 0
            || r.size > cfg_.slotBytes) {
            continue; // rejected, not counted as accepted
        }

        bool issued = false;

#ifdef VVM_HAS_IORING_HEADER
        if (ioringMode_ && r.isRead) {
            IORING_BUFFER_REF buf = IoRingBufferRefFromIndexAndOffset(r.dstSlot, 0);
            HRESULT hr = api.buildRead(static_cast<HIORING>(ring_),
                                       IoRingHandleRefFromHandle(static_cast<HANDLE>(file_)),
                                       buf, r.size, r.fileOffset,
                                       static_cast<UINT_PTR>(r.id), IOSQE_FLAGS_NONE);
            issued = SUCCEEDED(hr);
        } else
#endif
        {
            // OVERLAPPED floor: reads (Overlapped mode) and ALL writes.
            Pending p;
            p.id = r.id;
            LARGE_INTEGER li;
            li.QuadPart = static_cast<LONGLONG>(r.fileOffset);
            p.ov.Offset = li.LowPart;
            p.ov.OffsetHigh = li.HighPart;
            p.ov.hEvent = nullptr;

            BOOL ok;
            if (r.isRead) {
                ok = ::ReadFile(static_cast<HANDLE>(file_), slotPtr(r.dstSlot),
                                r.size, nullptr, &p.ov);
            } else {
                ok = ::WriteFile(static_cast<HANDLE>(file_), slotPtr(r.dstSlot),
                                 r.size, nullptr, &p.ov);
            }
            // ERROR_IO_PENDING -> in flight; TRUE -> completed synchronously
            // (still reaped via GetOverlappedResult); other -> rejected.
            if (ok || ::GetLastError() == ERROR_IO_PENDING) {
                pending_.push_back(std::move(p));
                issued = true;
            }
        }

        if (issued) {
            ++accepted;
            ++inFlight_;
        }
    }

#ifdef VVM_HAS_IORING_HEADER
    // One flush submits the whole batch of ring entries (doorbell coalescing
    // is what the kernel does per SubmitIoRing).
    if (ioringMode_ && accepted > 0 && api.submit) {
        UINT32 submitted = 0;
        api.submit(static_cast<HIORING>(ring_), 0, 0, &submitted);
    }
#endif

    return accepted;
}

uint32_t IoRingBackend::pollCompletions(std::vector<uint64_t>* outIds,
                                        std::vector<uint64_t>* outFailed) {
    if (!open_) return 0;
    uint32_t reaped = 0;

#ifdef VVM_HAS_IORING_HEADER
    const IoRingApi& api = IoRingApi::get();

    // IoRing reads: tag-based reap straight off the CQ (no linear search).
    if (ioringMode_) {
        for (;;) {
            IORING_CQE cqe{};
            HRESULT hr = api.pop(static_cast<HIORING>(ring_), &cqe);
            if (hr != S_OK) break; // S_FALSE = empty
            const uint64_t id = static_cast<uint64_t>(cqe.UserData);
            if (inFlight_) --inFlight_;
            ++reaped;
            if (SUCCEEDED(cqe.ResultCode)) {
                if (outIds) outIds->push_back(id);
            } else {
                if (outFailed) outFailed->push_back(id);
            }
        }
    }
#endif

    // OVERLAPPED floor sweep (writes in any mode; reads in Overlapped mode).
    for (auto it = pending_.begin(); it != pending_.end();) {
        DWORD n = 0;
        if (::GetOverlappedResult(static_cast<HANDLE>(file_), &it->ov, &n, FALSE)) {
            if (outIds) outIds->push_back(it->id);
            if (inFlight_) --inFlight_;
            ++reaped;
            it = pending_.erase(it);
        } else if (::GetLastError() == ERROR_IO_INCOMPLETE) {
            ++it;
        } else {
            if (outFailed) outFailed->push_back(it->id);
            if (inFlight_) --inFlight_;
            ++reaped;
            it = pending_.erase(it);
        }
    }

    return reaped;
}

// ---------------------------------------------------------------------------
// Factory (mirrors network_factory.cpp: env selection + compile-time gates)
// ---------------------------------------------------------------------------
BackendKind selectedBackendKind() {
    static const BackendKind b = [] {
        if (const char* e = std::getenv("VVM_STORAGE_BACKEND")) {
            if (!std::strcmp(e, "ioring")) return BackendKind::IoRing;
            if (!std::strcmp(e, "overlapped")) return BackendKind::Overlapped;
        }
        return BackendKind::Auto;
    }();
    return b;
}

std::unique_ptr<StreamBackend> createBackend(BackendKind kind, const BackendConfig& cfg) {
    switch (kind) {
        case BackendKind::IoRing:
        case BackendKind::Overlapped:
        case BackendKind::Auto:
            return std::make_unique<IoRingBackend>(cfg);
    }
    return std::make_unique<IoRingBackend>(cfg);
}

} // namespace backend
} // namespace storage
} // namespace vvm

#else // !VVM_PLATFORM_WINDOWS

// Non-Windows: the floor backend is unavailable; stubs keep the library
// linkable so Linux CI compiles unchanged.
#include "vulkan_vm/storage/ioring_backend.hpp"

namespace vvm {
namespace storage {
namespace backend {

IoRingBackend::IoRingBackend(BackendConfig cfg) : cfg_(std::move(cfg)) {}
IoRingBackend::~IoRingBackend() = default;
Result IoRingBackend::open() { return Result::error(ErrorCode::UnsupportedFeature, "Windows-only backend"); }
void IoRingBackend::close() {}
void* IoRingBackend::slotPtr(uint32_t) const { return nullptr; }
uint32_t IoRingBackend::submitBatch(const std::vector<IORequest>&) { return 0; }
uint32_t IoRingBackend::pollCompletions(std::vector<uint64_t>*, std::vector<uint64_t>*) { return 0; }
BackendKind selectedBackendKind() { return BackendKind::Auto; }
std::unique_ptr<StreamBackend> createBackend(BackendKind, const BackendConfig&) { return nullptr; }

} // namespace backend
} // namespace storage
} // namespace vvm

#endif // VVM_PLATFORM_WINDOWS