// nvme_bench: raw drive ceiling probe through the L3 IoRing backend.
// Sequential vs random-position reads on the same file, Little's Law depth.
// Usage: nvme_bench <file> <seq|rand> <gigabytes> [depth] [iosize-mb]
// Real file I/O, no GPU. Windows-only (OVERLAPPED floor elsewhere).

#include "vulkan_vm/storage/ioring_backend.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef VVM_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>

using namespace vvm::storage::backend;
using vvm::storage::queue::IORequest;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: nvme_bench <file> <seq|rand> <gigabytes> [depth] [iosize-mb]\n";
        return 1;
    }
    const std::string path = argv[1];
    const bool seq = std::string(argv[2]) == "seq";
    const uint64_t targetBytes = static_cast<uint64_t>(std::atoll(argv[3])) << 30;
    const uint32_t depth = argc > 4 ? static_cast<uint32_t>(std::atoi(argv[4])) : 32;
    const uint32_t ioMiB = argc > 5 ? static_cast<uint32_t>(std::atoi(argv[5])) : 4;
    const uint32_t ioBytes = ioMiB << 20;
    const bool forceOverlapped = argc > 6 && std::string(argv[6]) == "ovlp";

    // file size (for random range); raw volumes via IOCTL_DISK_GET_LENGTH_INFO
    LARGE_INTEGER fsize{};
    {
        HANDLE h = ::CreateFileA(path.c_str(), GENERIC_READ,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                 nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) { std::cerr << "open failed\n"; return 1; }
        GET_LENGTH_INFORMATION lenInfo{};
        DWORD br = 0;
        if (::DeviceIoControl(h, IOCTL_DISK_GET_LENGTH_INFO, nullptr, 0,
                              &lenInfo, sizeof(lenInfo), &br, nullptr)) {
            fsize.QuadPart = lenInfo.Length.QuadPart;
        } else if (!::GetFileSizeEx(h, &fsize)) {
            std::cerr << "size failed\n";
            ::CloseHandle(h);
            return 1;
        }
        ::CloseHandle(h);
    }
    const uint64_t fileBytes = static_cast<uint64_t>(fsize.QuadPart);
    std::cout << "file: " << (fileBytes >> 30) << " GiB, mode=" << (seq ? "seq" : "rand")
              << ", target=" << (targetBytes >> 30) << " GiB, depth=" << depth
              << ", io=" << ioMiB << " MiB\n";

    BackendConfig cfg;
    cfg.packPath = path;
    cfg.slotCount = depth;
    cfg.slotBytes = ioBytes;
    cfg.queueDepth = 256;
    cfg.allowIoRing = !forceOverlapped;
    IoRingBackend be(cfg);
    vvm::Result r = be.open();
    if (!r) { std::cerr << "backend open failed: " << r.message << "\n"; return 1; }
    std::cout << "backend mode: " << be.activeMode()
              << (be.bypassIo() ? " +bypassio" : " (no bypassio)") << "\n";

    std::mt19937_64 rng(1234);
    const uint64_t maxOff = (fileBytes > ioBytes) ? (fileBytes - ioBytes) & ~4095ull : 0;

    uint64_t issued = 0, done = 0, bytes = 0;
    uint64_t nextSeq = 0;
    std::vector<IORequest> batch;
    int spins = 0;

    auto t0 = std::chrono::steady_clock::now();
    while (bytes < targetBytes && spins < 20000000) {
        ++spins;
        // fill window
        uint32_t inFlight = 0; // tracked implicitly: issued - done
        while ((issued - done) < depth && bytes + issued * 0 < targetBytes) {
            if (done * ioBytes + (issued - done) * ioBytes >= targetBytes) break;
            IORequest req;
            req.id = issued + 1;
            if (seq) {
                req.fileOffset = (nextSeq * ioBytes) % (maxOff ? maxOff : ioBytes);
                ++nextSeq;
            } else {
                req.fileOffset = (rng() % (maxOff / 4096 + 1)) * 4096;
            }
            req.size = ioBytes;
            req.dstSlot = static_cast<uint32_t>(issued % depth);
            req.isRead = true;
            batch.push_back(req);
            ++issued;
            inFlight = static_cast<uint32_t>(issued - done);
            (void)inFlight;
        }
        if (!batch.empty()) be.submitBatch(batch);
        batch.clear();
        std::vector<uint64_t> d, f;
        be.pollCompletions(&d, &f);
        done += d.size();
        bytes = done * ioBytes;
        if (!f.empty()) { std::cerr << "failures: " << f.size() << "\n"; return 1; }
        if (d.empty()) ::Sleep(0);
    }
    auto t1 = std::chrono::steady_clock::now();

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double gbps = (bytes / 1073741824.0) / secs;
    std::cout << (seq ? "seq" : "rand") << ": " << done << " ios, "
              << (bytes >> 30) << " GiB in " << secs << "s = " << gbps << " GB/s\n";
    be.close();
    return 0;
}

#else
int main() {
    std::cout << "nvme_bench: SKIP (Windows-only)\n";
    return 0;
}
#endif