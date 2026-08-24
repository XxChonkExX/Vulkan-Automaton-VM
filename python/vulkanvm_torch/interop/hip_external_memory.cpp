// hip_external_memory.cpp - HIP <-> Vulkan external-memory bridge impl.

#include "hip_external_memory.hpp"

namespace vvm_torch {

void* hipImportFromFd(int fd, size_t size, hipExternalMemory_t* outExt) {
    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = size;
    hipExternalMemory_t ext = nullptr;
    if (hipImportExternalMemory(&ext, &desc) != hipSuccess) return nullptr;
    void* base = nullptr;
    hipExternalMemoryBufferDesc buf{};
    buf.offset = 0;
    buf.size = size;
    if (hipExternalMemoryGetMappedBuffer(&base, ext, &buf) != hipSuccess) {
        hipDestroyExternalMemory(ext);
        return nullptr;
    }
    *outExt = ext;
    return base;
}

}  // namespace vvm_torch
