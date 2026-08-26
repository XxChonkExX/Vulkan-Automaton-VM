# VulkanVM Lifetime & Ownership Contract

> **This document is normative.** If code in this repository violates these
> rules, it is a bug. If you consume VulkanVM from another project, these are
> the guarantees you may rely on — and the rules you must follow.
>
> Status: **0.3 contract**. APIs may still change before release, but the
> ownership MODEL below is the intended stable model.

---

## 1. Object graph

```
Device (VkInstance/VkDevice - owned by the EMBEDDER)
  └── UnifiedMemoryPool            (created against a DeviceConfig)
        ├── Block(s)               (VkDeviceMemory + buddy allocator, private)
        │     └── Allocation(s)    (sub-allocated VkBuffer + offset/size)
        ├── Dedicated Allocation(s)(own VkDeviceMemory, exportable/imported)
        └── OffloadManager         (optional; host shadow + migration engine)
              └── MigrationOperation(s)

Allocation --exportMemory--> ExternalMemoryInfo (owns an OS handle: fd/HANDLE)
ExternalMemoryInfo --importMemory--> Allocation (on the PEER pool)
Allocation --(Linux)--> dma-buf fd --HIP import--> HIP external memory
                                    --mapped--> PyTorch storage pointer
```

## 2. Core rules

| # | Rule |
|---|------|
| R1 | **The pool owns every Allocation.** An `Allocation` is a handle (RAII via `UniqueAllocation`); it does not own the VkDevice, the block memory, or the pool. |
| R2 | **The embedder owns the VkDevice.** The pool never destroys the device or instance it was created against. Device lifetime must strictly enclose pool lifetime. |
| R3 | **`exportMemory` transfers handle ownership on success.** After a successful export, the returned `ExternalMemoryInfo` owns the OS handle; its destructor closes it. `handle.release()` detaches the fd/HANDLE for hand-off to another subsystem (e.g. HIP import) — the receiver then owns it. On FAILURE the info still owns the handle. |
| R4 | **`importMemory` consumes the handle.** On success the OS handle is transferred to the driver and the info is emptied. Use `duplicateForImport` when the same memory must be imported by multiple peers — each peer needs its own handle. |
| R5 | **An external alias cannot outlive its Vulkan allocation.** A HIP external-memory import, a mapped PyTorch storage, or an imported peer `Allocation` are all *aliases*. Destroying (or offloading-and-releasing) the source Vulkan allocation while an alias exists is undefined behavior. |
| R6 | **Pool destruction requires all aliases gone and all allocations freed.** `~UnifiedMemoryPool` deallocates surviving sub-allocations, but external aliases (HIP imports, PyTorch tensors, peer pools) are the *embedder's* responsibility to release first. |
| R7 | **Sub-allocated (non-dedicated) allocations are NOT exportable.** Cross-GPU export requires `allocateDedicatedExportable` — Vulkan external import requires dedicated allocations for reliable cross-device import. |
| R8 | **Offloaded allocations must be reloaded before use.** After `offloadToHost`, the device-side contents are undefined until `reloadToDevice` completes (`waitMigration`). Accessing a device pointer while offloaded is UB. Offload does NOT release the Vulkan allocation — the alias rule (R5) is unaffected by offload state. |
| R9 | **Thread safety.** All public `UnifiedMemoryPool` methods are mutex-guarded. `Allocation` handles are not — do not share a single handle across threads without external synchronization. Move, don't copy. |
| R10 | **Generation guard.** Every allocation carries a generation counter; using a stale handle after `deallocate` is rejected (`deallocate: stale allocation handle` warning), not UB. |

## 3. The PyTorch integration contract (H3)

The Chonk pluggable allocator (`python/vulkanvm_torch/allocator/`) and the
parameter-binding layer make aggressive use of PyTorch internals. This is the
compatibility contract that must hold:

### 3.1 Allocator ABI

- Installed via `torch.cuda.memory.CUDAPluggableAllocator` old-style 2-function
  ABI: `alloc_fn(ssize_t, int, void*)` / `free_fn(void*, size_t, void*)`.
- `alloc_fn` rejects negative sizes (`nullptr`), treats 0 as one 512-byte
  granularity unit (hipMalloc(0) semantics), and returns **device-accessible**
  pointers carved from DMA-BUF-imported Chonk blocks.
- `free_fn` tolerates double frees and pointers from other allocators
  (logged + ignored, never corrupts the free list).
- **Supported PyTorch versions:** those shipping the 2-function
  CUDAPluggableAllocator ABI (torch >= 2.1). Re-validate on every major torch
  bump; the ABI is stable but not guaranteed forever.

### 3.2 Parameter binding (`param.data` replacement)

The training path re-seats model parameters into Chonk memory:

```python
param_view = buffer_typed.narrow(0, offset, numel).view_as(param)
param_view.copy_(param.data)
param.data = param_view
```

Invariants this depends on (and that a torch upgrade may break):

| Invariant | Why it matters |
|---|---|
| `Tensor.data` replacement does not free the old storage while views exist | autograd saved tensors may alias the old storage |
| `.narrow().view_as()` produces a view sharing the imported storage | zero-copy guarantee |
| The HIP caching allocator does not free pointers it did not allocate | Chonk pointers pass through `free_fn` safely |
| Storage lifetime is bound to the tensor, not the parameter object | optimizer states alias the same storage |
| No allocator move/defrag while training steps run | pointers are raw device addresses |

**Rule:** parameter binding happens ONCE at setup, before the training loop;
nothing may reallocate or shutdown the pool while any bound parameter,
optimizer state, or autograd-saved tensor exists. `shutdown()` is a
teardown-only API.

### 3.3 Autograd

Custom `torch.autograd.Function` implementations
(`vulkanvm_autograd.hpp`) provide VulkanVM-compatible forward/backward and
may dispatch through Vulkan compute when an eligible backend is available;
otherwise they execute through ATen. The gradient mathematics follows the
standard PyTorch formulations (no custom calculus). Numerical validation
against ATen autograd is tracked as a first-class test suite (M4).

## 4. Network transport ownership

- `ExternalMemoryInfo` handles crossing a transport boundary follow R3/R4:
  the sender releases after the receiver confirms import (protocol-level
  ack), or the fd is duplicated per peer.
- Streamed tensor data (4 MiB slices) carries no ownership — bytes only.
- TLS contexts own their ALPN storage and are freed with the context
  (`TlsContext::cleanup`).
- See `docs/THREAT_MODEL.md` for trust boundaries.

## 5. Violation examples (do not do these)

```cpp
// BAD: alias outlives source
void* pytorch_ptr = chonk_alloc(...);        // alias into Chonk block
pool.shutdown();                             // R6 violated: alias now dangling

// BAD: handle double-ownership
auto info = pool.exportMemory(alloc, type);
int fd = info->handle.get();                 // peek OK
close(fd);                                   // R3 violated: info still owns fd

// BAD: import without duplication for a second peer
peer1.importMemory(std::move(info));         // consumes handle (R4)
peer2.importMemory(std::move(info));         // empty info -> failure

// BAD: use after offload
pool.offloadToHost(alloc);
readFromDevicePointer(alloc);                // R8 violated
```

## 6. Validation checklist for contributors

Before merging anything that touches allocation, export/import, or teardown:

- [ ] Every `new Block`/allocation path has a matching destroy path
      (see `tests/chonk_slab_test.cpp` for the fuzz harness pattern)
- [ ] Exported handles: who owns them after each branch (success AND failure)?
- [ ] Can any alias outlive its source under your new code path?
- [ ] Vulkan validation layers clean (`VVM_ENABLE_VALIDATION=ON` builds)
- [ ] Python integration: torch version recorded, allocator ABI re-validated

## 7. Tensor handle lifetime (Compute layer)

- A `TensorHandle` (`TensorAllocation`) owns its `VkBuffer` for as long as
  `allocation.buffer != VK_NULL_HANDLE`.
- **Default rule: explicit return.** Callers return tensors to their pool with
  `pool.deallocate(std::move(handle->allocation))`. `deallocate` zeroes the
  handles, so a later destruction of the same handle is a no-op.
- **Opt-in auto-free:** engine-created tensors (via
  `Transport::allocateTensor` / `allocateDistributed`) carry a releaser bound
  to their owning pool; if the last reference drops without an explicit
  return, the destructor releases the buffer automatically.
  Hand-built handles (tests/tools) have no releaser - they leak-warn on
  destruction while a buffer is still attached.
- **Leak tripwire:** set `VVM_TENSOR_LEAK_ABORT=1` to turn the warning into an
  abort in CI/debug runs.
- Handles must not outlive their transport/pool owner. The transport detaches
  cached replicas during its own destruction; user-held handles past that
  point are use-after-free by contract, not a supported state.
- Pool blocks and staging buffers follow the same explicit-return rule;
  internal migration stagings are RAII-released (`StagingReleaser` /
  freeing-deleter) and must never escape as raw `Allocation` copies.
