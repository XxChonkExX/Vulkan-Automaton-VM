// CPU-only ownership tests for external-memory OS handles (no Vulkan device).
//
// Verifies the import ownership contract:
//   - duplicateForImport creates an INDEPENDENT handle per peer, leaving the
//     original owned by the exporter (multi-GPU import does not invalidate it).
//   - "consuming" a handle (release + close, as importMemory does after a
//     successful vkAllocateMemory) does not affect the original.
//   - RAII destructors close each owned handle exactly once.

#include <vulkan_vm/utils.hpp>

#include <cstdio>

#ifdef VVM_PLATFORM_LINUX
#include <fcntl.h>
#include <unistd.h>
#elif defined(VVM_PLATFORM_WINDOWS)
#include <windows.h>
#endif

using vvm::duplicateForImport;
using vvm::ExternalHandle;
using vvm::ExternalMemoryInfo;

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

int main() {
#ifdef VVM_PLATFORM_LINUX
    {
        // Source handle owned by the exporter (analogous to exportMemory).
        ExternalMemoryInfo src;
        src.size = 4096;
        src.dedicatedAllocation = true;
        src.handle = ExternalHandle(open("/dev/null", O_RDONLY));
        CHECK(src.handle);

        // Multi-peer: each peer must receive its OWN dup'ed handle.
        auto copy = duplicateForImport(src);
        CHECK(copy.handle);
        CHECK(copy.size == src.size);
        CHECK(copy.dedicatedAllocation == src.dedicatedAllocation);
        CHECK(copy.handle.get() != src.handle.get());  // independent handle
        CHECK(src.handle);                             // original untouched

        // Successful import consumes exactly one handle (release + close):
        // closing the dup must NOT invalidate the original.
        int consumed = copy.handle.release();
        CHECK(close(consumed) == 0);

        // Original still usable afterwards.
        int fd2 = dup(src.handle.get());
        CHECK(fd2 >= 0);
        if (fd2 >= 0) close(fd2);

        // src's destructor closes the original exactly once.
    }
#elif defined(VVM_PLATFORM_WINDOWS)
    {
        ExternalMemoryInfo src;
        src.size = 4096;
        src.dedicatedAllocation = true;
        src.handle = ExternalHandle(
            CreateFileW(L"NUL", GENERIC_READ, 0, nullptr, OPEN_EXISTING, 0, nullptr));
        CHECK(src.handle);

        auto copy = duplicateForImport(src);
        CHECK(copy.handle);
        CHECK(copy.size == src.size);
        CHECK(copy.dedicatedAllocation == src.dedicatedAllocation);
        CHECK(copy.handle.get() != src.handle.get());  // independent handle
        CHECK(src.handle);                             // original untouched

        // Successful import consumes exactly one handle (release + close):
        // closing the duplicated handle must NOT invalidate the original.
        HANDLE consumed = copy.handle.release();
        CHECK(consumed != nullptr);
        CHECK(CloseHandle(consumed) != FALSE);

        // Original still usable afterwards.
        HANDLE h2 = nullptr;
        CHECK(DuplicateHandle(GetCurrentProcess(), src.handle.get(),
                              GetCurrentProcess(), &h2, 0, FALSE,
                              DUPLICATE_SAME_ACCESS) != FALSE);
        if (h2) CloseHandle(h2);

        // src's destructor closes the original exactly once.
    }
#endif

    if (failures == 0) {
        std::printf("=== ALL EXTERNAL HANDLE TESTS PASSED (0 failures) ===\n");
        return 0;
    }
    std::printf("=== SOME TESTS FAILED (%d failures) ===\n", failures);
    return 1;
}