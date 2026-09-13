#pragma once

// L3 backend — Windows IoRing + OVERLAPPED floor (pure Vulkan stack side).
//
// Reads go through Windows IoRing (Build 22000+) when the kernel supports it:
//   CreateIoRing (SQ/CQ) -> BuildIoRingRegisterBuffers (pinned staging slots)
//   -> BuildIoRingReadFile (UserData = IORequest.id) -> SubmitIoRing
//   -> PopIoRingCompletion (tag-based reap, no linear search).
// Writes use the OVERLAPPED pool (public IoRing write requires API v3+; the
// OVERLAPPED floor is honest, portable, and read+write from WinXP onward).
//
// The API is loaded dynamically from kernelbase.dll — no iouring import-lib
// dependency, works on any SDK, and degrades cleanly:
//   * CreateIoRing missing (pre-22000)      -> OVERLAPPED mode
//   * IORING_FEATURE_UM_EMULATION only      -> OVERLAPPED mode (no kernel
//     benefit, so we do not fake one)
//   * registration fails                    -> OVERLAPPED mode
//
// O_DIRECT semantics: the pack file is opened with FILE_FLAG_NO_BUFFERING,
// which requires 4 KiB-aligned offset AND size (the L0 pack aligns offsets;
// consumers pad shard sizes to 4 KiB for the streaming path).

#include "vulkan_vm/storage/request_queue.hpp"
#include "vulkan_vm/core.hpp" // vvm::Result

#include <memory>
#include <string>
#include <vector>

#ifndef VVM_API
#if defined(VVM_BUILD_SHARED) && defined(VVM_EXPORT)
#  if defined(_MSC_VER)
#    define VVM_API __declspec(dllexport)
#  else
#    define VVM_API __attribute__((visibility("default")))
#  endif
#elif defined(VVM_BUILD_SHARED) && defined(_MSC_VER)
#  define VVM_API __declspec(dllimport)
#else
#  define VVM_API
#endif
#endif

namespace vvm {
namespace storage {
namespace backend {

using queue::IORequest;      // StreamBackend's request type (vvm::storage::queue)
using queue::StreamBackend;  // the L2 backend seam this class implements

// Swappable backend selection (mirrors network_factory.cpp / VVM_RDMA_BACKEND).
enum class BackendKind : uint32_t {
    Auto = 0,       // IoRing when kernel-supported, else Overlapped
    IoRing = 1,     // force IoRing (fails open() -> caller falls back)
    Overlapped = 2, // force the portable floor
};

struct VVM_API BackendConfig {
    std::string packPath;          // file to serve
    uint32_t slotCount = 8;        // staging slots to register with the ring
    uint32_t slotBytes = 1u << 20; // per-slot bytes (IO granularity, 4 KiB-aligned)
    uint32_t queueDepth = 256;     // IoRing SQ/CQ size
    bool allowIoRing = true;       // false forces the OVERLAPPED floor

    bool validate() const {
        return !packPath.empty() && slotCount != 0 && slotBytes != 0
            && (slotBytes % 4096) == 0 && queueDepth != 0;
    }
};

class VVM_API IoRingBackend final : public StreamBackend {
public:
    explicit IoRingBackend(BackendConfig cfg);
    ~IoRingBackend() override;

    IoRingBackend(const IoRingBackend&) = delete;
    IoRingBackend& operator=(const IoRingBackend&) = delete;

    // Probes IoRing support, opens the pack (NO_BUFFERING|OVERLAPPED),
    // registers the staging arena. Never throws: on any IoRing failure it
    // degrades to OVERLAPPED mode and still opens.
    Result open();
    void close();

    bool isOpen() const { return open_; }
    // "ioring" or "overlapped" — what submitBatch/pollCompletions actually use.
    const char* activeMode() const { return ioringMode_ ? "ioring" : "overlapped"; }
    bool ioringActive() const { return ioringMode_; }

    // Staging arena (backend-owned; dstSlot indexes it).
    void* slotPtr(uint32_t slot) const;
    uint32_t slotBytes() const { return cfg_.slotBytes; }
    uint32_t slotCount() const { return cfg_.slotCount; }

    // StreamBackend
    uint32_t submitBatch(const std::vector<IORequest>& batch) override;
    uint32_t pollCompletions(std::vector<uint64_t>* outIds,
                             std::vector<uint64_t>* outFailed = nullptr) override;
    const char* name() const override { return activeMode(); }

private:
    void destroyRing();
    void degradeToOverlapped();
    bool issueReadIoRing(const IORequest& r);
    bool issueOverlapped(const IORequest& r);

    BackendConfig cfg_;
    bool open_ = false;
    bool ioringMode_ = false;

    void* arena_ = nullptr;            // VirtualAlloc'd, slotCount * slotBytes

    // IoRing plumbing (dynamic-loaded; null when unavailable).
    void* ring_ = nullptr;             // HIORING
    std::vector<uint8_t> regInfo_;     // IORING_BUFFER_INFO array (must stay valid)

    // OVERLAPPED pending pool (reads in Overlapped mode + all writes).
    struct Pending {
        uint64_t id = 0;
        OVERLAPPED ov{};               // must stay valid until completion
    };
    std::vector<Pending> pending_;

    void* file_ = nullptr;             // HANDLE
    uint64_t inFlight_ = 0;
};

VVM_API BackendKind selectedBackendKind(); // VVM_STORAGE_BACKEND env (auto|ioring|overlapped)
VVM_API std::unique_ptr<StreamBackend> createBackend(BackendKind kind, const BackendConfig& cfg);

} // namespace backend
} // namespace storage
} // namespace vvm