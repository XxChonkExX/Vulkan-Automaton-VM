# ANV (Battlemage G31): SIGSEGV in vkAllocateMemory when importing another driver's external memory

**Target:** Mesa gitlab (freedesktop.org) — `mesa/mesa`, component `Drivers/Vulkan/Intel`
**Status:** draft — verify against latest main before filing; re-run repro on current Mesa first.

## Summary

On Mesa 26.0.3-1ubuntu1 (Ubuntu 26.04), importing external memory that was
exported by a *different vendor's* driver (RADV / Radeon RX 7900 XTX) into an
ANV logical device (Battlemage G31) segfaults inside `libvulkan_intel.so`
during `vkAllocateMemory` with an import pNext chain
(`VK_KHR_external_memory_fd` opaque-fd **or** `VK_EXT_external_memory_dma_buf`).

The identical code path succeeds RADV→RADV (two RADV logical devices), and the
same ANV device successfully allocates non-imported memory throughout the same
process.

## Environment

- Ubuntu 26.04 (Resolute), kernel 7.0.0-30-generic, GNOME/Wayland
- Mesa 26.0.3-1ubuntu1: RADV NAVI31 + ANV BMG G31 (+ lavapipe, Raphael iGPU)
- Vulkan API 1.4.335 reported by both drivers
- CPU: Ryzen 9 7900X; X870E board; Resizable BAR active

## Repro

Repository: https://github.com/XxChonkExX/Vulkan-Automaton-VM (MIT).

```bash
git clone https://github.com/XxChonkExX/Vulkan-Automaton-VM
cd Vulkan-Automaton-VM
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVVM_BUILD_NETWORK=OFF
cmake --build build -j$(nproc)
./build/tests/multi_gpu_test        # cross-vendor pair selection: XTX -> BMG
```

Observed:

```
Thread 1 "multi_gpu_test" received signal SIGSEGV
#0  () at /usr/lib/x86_64-linux-gnu/libvulkan_intel.so
#1  vvm::UnifiedMemoryPool::importMemory(...) 
    inside vkAllocateMemory(device=ANV, pAllocateInfo{
        VkImportMemoryFdInfoKHR{OPAQUE_FD or DMA_BUF, fd},
        VkMemoryDedicatedAllocateInfo{buffer},
        VkMemoryAllocateFlagsInfo{DEVICE_ADDRESS_BIT} })
```

Minimal shape of the failing call (all on the ANV device):

- `vkCreateBuffer` with
  `VkExternalMemoryBufferCreateInfo{OPAQUE_FD|DMA_BUF}` — succeeds
- `vkGetBufferMemoryRequirements` — succeeds
- `vkAllocateMemory(allocationSize=max(exportSize,req.size), memoryTypeIndex=<DEVICE_LOCAL>,
   pNext: dedicated(buffer) → importFd → flagsInfo(DEVICE_ADDRESS))` — **SIGSEGV**

## What we ruled out (bisected)

| Hypothesis | Result |
|---|---|
| Handle type (DMA-BUF vs OPAQUE_FD) | crashes with both |
| `VK_EXT_memory_budget` enabled on importer | crash persists without it |
| Missing `VK_KHR_dedicated_allocation` | crash persists (core in 1.3 anyway) |
| Memory type selection | same index works for RADV→RADV |
| KHR validation-layer misuse before the call | clean up to the crash |
| Exporter-side allocation validity | export succeeded; fd consumed once |

Cross-check within the same repo/test-suite: allocating the pair as
AMD↔AMD (`tests/multi_gpu_test` prefers a same-vendor pair) performs the exact
same dedicated+BDA import **successfully** on this machine — the failure is
specific to crossing into ANV.

## Notes for triage

- Happy to test candidate patches; machine is a standing dual-vendor rig.
- A standalone ~150-line C reproducer can be extracted from
  `src/core/unified_memory_pool.cpp::importMemory` if useful.
- Workaround currently shipped downstream: refuse same-process cross-vendor
  zero-copy imports entirely (`zcAllowedForPair()` in
  `src/network/multi_node_manager.cpp`).
