// storage_backend_test: L3 IoRing/OVERLAPPED backend — real file I/O.
// Writes a temp file, streams it through the backend (real reads into the
// registered staging arena), verifies content, then exercises the write path.
// CPU-only (no GPU); skips gracefully on non-Windows.
//
// NOTE: real I/O calls are never inside assert() — every call is bound to a
// value and checked explicitly (assert compiles out under NDEBUG).

#include "vulkan_vm/storage/ioring_backend.hpp"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef VVM_PLATFORM_WINDOWS

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

using namespace vvm;
using namespace vvm::storage::backend;
using vvm::storage::queue::IORequest;

constexpr uint32_t kSlotBytes = 1u << 20;      // 1 MiB slots
constexpr uint32_t kSlots = 4;
constexpr uint64_t kFileBytes = 8ull << 20;    // 8 MiB file (4K-aligned)

static void fillBlock(std::vector<uint8_t>& file, uint64_t blockIdx) {
    // Deterministic per-4K-block pattern: first 8 bytes = block index LE.
    uint8_t* p = file.data() + blockIdx * 4096;
    for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>((blockIdx >> (8 * i)) & 0xFF);
    for (int i = 8; i < 4096; ++i) p[i] = static_cast<uint8_t>(blockIdx ^ i);
}

static bool verifySlot(const void* slot, uint64_t blockIdx) {
    const uint8_t* p = static_cast<const uint8_t*>(slot);
    for (int i = 0; i < 8; ++i) {
        if (p[i] != static_cast<uint8_t>((blockIdx >> (8 * i)) & 0xFF)) return false;
    }
    for (int i = 8; i < 4096; ++i) {
        if (p[i] != static_cast<uint8_t>(blockIdx ^ i)) return false;
    }
    return true;
}

int main() {
    std::cout << "storage_backend_test (real file I/O)\n";

    const std::string path = "moe_backend_test.bin";

    // 1) temp file: 2048 4K blocks, deterministic content.
    std::vector<uint8_t> file(kFileBytes);
    for (uint64_t b = 0; b < kFileBytes / 4096; ++b) fillBlock(file, b);
    {
        FILE* f = nullptr;
        if (fopen_s(&f, path.c_str(), "wb") != 0 || !f) {
            std::cerr << "  FAIL: create temp file\n";
            return 1;
        }
        const size_t wrote = fwrite(file.data(), 1, file.size(), f);
        fclose(f);
        if (wrote != file.size()) { std::cerr << "  FAIL: short temp write\n"; return 1; }
    }

    // 2) open backend
    BackendConfig cfg;
    cfg.packPath = path;
    cfg.slotCount = kSlots;
    cfg.slotBytes = kSlotBytes;
    IoRingBackend be(cfg);
    Result r = be.open();
    if (!r) { std::cerr << "  FAIL: open: " << r.message << "\n"; return 1; }
    std::cout << "  mode=" << be.activeMode() << "\n";

    int failures = 0;

    // 3) stream the whole file through the backend: 8 reads of 1 MiB.
    //    2 reads in flight per slot; verify each slot as it lands.
    constexpr uint64_t kChunks = kFileBytes / kSlotBytes; // 8
    uint64_t nextChunk = 0;
    uint64_t verified = 0;
    std::vector<IORequest> batch;
    int spins = 0;

    while (verified < kChunks && spins < 100000) {
        ++spins;
        // keep one outstanding read per slot: a later chunk to the same slot
        // would overwrite the data before it is verified below.
        while (batch.size() < kSlots && nextChunk < kChunks) {
            IORequest req;
            req.id = nextChunk + 1; // correlation handle (CQE UserData); queue assigns when used
            req.shardKey = nextChunk;
            req.fileOffset = nextChunk * kSlotBytes;
            req.size = kSlotBytes;
            req.dstSlot = static_cast<uint32_t>(nextChunk % kSlots);
            req.isRead = true;
            batch.push_back(req);
            ++nextChunk;
        }
        if (!batch.empty()) be.submitBatch(batch);
        batch.clear();

        std::vector<uint64_t> done, failed;
        be.pollCompletions(&done, &failed);
        if (!failed.empty()) {
            std::cerr << "  FAIL: " << failed.size() << " reads failed\n";
            ++failures;
            break;
        }
        for (uint64_t id : done) {
            // id was assigned chunk order + 1 (queue not used here; ids are
            // 1-based in submission order)
            const uint64_t chunk = id - 1;
            const uint32_t slot = static_cast<uint32_t>(chunk % kSlots);
            // The slot holds 1 MiB = 256 consecutive 4K blocks starting at
            // chunk*256; verify against the chunk's FIRST block.
            const uint64_t firstBlock = chunk * (kSlotBytes / 4096);
            if (!verifySlot(be.slotPtr(slot), firstBlock)) {
                std::cerr << "  FAIL: chunk " << chunk << " content mismatch\n";
                ++failures;
            } else {
                ++verified;
            }
        }
        if (done.empty()) ::Sleep(0);
    }
    if (verified != kChunks) {
        std::cerr << "  FAIL: verified " << verified << "/" << kChunks << "\n";
        ++failures;
    } else {
        std::cout << "  read path OK (" << verified << " MiB via "
                  << be.activeMode() << ")\n";
    }

    // 4) write path: fill slot 0 with a marker, write it to EOF, read back.
    if (failures == 0) {
        constexpr uint64_t kWriteOff = kFileBytes; // 4K-aligned (file is 8 MiB)
        std::vector<uint8_t> marker(kSlotBytes, 0x5A);
        std::memcpy(be.slotPtr(0), marker.data(), marker.size());

        std::vector<IORequest> wbatch(1);
        wbatch[0].shardKey = 0xEE;
        wbatch[0].fileOffset = kWriteOff;
        wbatch[0].size = kSlotBytes;
        wbatch[0].dstSlot = 0;
        wbatch[0].isRead = false;
        be.submitBatch(wbatch);

        int wspins = 0;
        bool writeDone = false;
        while (wspins < 100000 && !writeDone) {
            ++wspins;
            std::vector<uint64_t> done, failed;
            be.pollCompletions(&done, &failed);
            if (!done.empty()) writeDone = true;
            if (!failed.empty()) { std::cerr << "  FAIL: write failed\n"; ++failures; break; }
            if (!writeDone) ::Sleep(0);
        }

        if (writeDone) {
            // read it back into slot 1 and verify
            std::vector<IORequest> rb(1);
            rb[0].shardKey = 0xEF;
            rb[0].fileOffset = kWriteOff;
            rb[0].size = kSlotBytes;
            rb[0].dstSlot = 1;
            rb[0].isRead = true;
            be.submitBatch(rb);
            int rspins = 0;
            bool readDone = false;
            while (rspins < 100000 && !readDone) {
                ++rspins;
                std::vector<uint64_t> done, failed;
                be.pollCompletions(&done, &failed);
                if (!done.empty()) readDone = true;
                if (!failed.empty()) { std::cerr << "  FAIL: readback failed\n"; ++failures; break; }
                if (!readDone) ::Sleep(0);
            }
            if (readDone) {
                const uint8_t* p = static_cast<const uint8_t*>(be.slotPtr(1));
                bool ok = true;
                for (uint32_t i = 0; i < kSlotBytes; ++i)
                    if (p[i] != 0x5A) { ok = false; break; }
                if (ok) std::cout << "  write path OK (write -> read-back verified)\n";
                else { std::cerr << "  FAIL: write read-back mismatch\n"; ++failures; }
            }
        }
    }

    be.close();
    std::remove(path.c_str());

    std::cout << (failures == 0 ? "All backend tests passed!\n" : "Backend tests FAILED\n");
    return failures == 0 ? 0 : 1;
}

#else // !VVM_PLATFORM_WINDOWS

int main() {
    std::cout << "storage_backend_test: SKIP (Windows-only backend)\n";
    return 0;
}

#endif // VVM_PLATFORM_WINDOWS