# External Semaphores

Export/import `VkSemaphore` payloads as OS handles for cross-device and
cross-process GPU synchronization — no host round-trip, no polling.

## Model

```
process A (or GPU 0)                process B (or GPU 1)
  exportable timeline  ---fd/HANDLE--->  import  --->  wait(value)
       signal(value)                         (same payload, second view)
```

The multi-process inference story: the renderer signals, the consumer
waits. The same opaque fd feeds foreign APIs (`cudaImportExternalSemaphore`,
HIP stream wait) without a Vulkan queue on their side.

## API (`vulkan_vm/cross_gpu/external_semaphore.hpp`)

| Function | Notes |
|---|---|
| `queryExternalSemaphoreCaps(phys, handleType, timeline)` | Instance-level, no device needed. Gate everything on this. |
| `createExportableSemaphore(device, handleType, timeline)` | Adds `VkExportSemaphoreCreateInfo`. Timeline needs the `timelineSemaphore` feature. |
| `exportSemaphoreHandle(device, sem, handleType)` | Fresh OS handle per call; the semaphore stays valid. |
| `importSemaphoreHandle(device, handle&&, handleType, timeline)` | **Consumes** the handle on success (R4). Dup first for N peers. |
| `signalTimelineSemaphore / waitTimelineSemaphore` | CPU-side, no queue. Wait returns false on timeout, never overruns it. |
| `defaultSemaphoreHandleType()` | `OPAQUE_FD` on Linux/Android, `OPAQUE_WIN32` on Windows. |

Handle types: timeline and binary both support `OPAQUE_FD` / `OPAQUE_WIN32`.
`OPAQUE_WIN32_KMT` and `SYNC_FD` are deliberately out of scope for v0.4.1
(KMT is adapter-local; sync-fd is a temporary-import regime of its own).

## Ownership rules

- Import consumes exactly one handle — the driver owns it afterwards.
  The caller's RAII `ExternalHandle` is released, not closed.
- Import failure consumes nothing: the handle stays valid and closes
  with its owner. Retry with a fresh dup is safe.
- `ExternalSemaphore` destroys the `VkSemaphore`; the `VkDevice` stays
  caller-owned. Destroy semaphores before their device (same split as
  `Allocation`/pool).
- Binary payloads: re-export only once back in the unsignaled state;
  timelines re-export freely.

## Handle-type matrix (verified)

| OS | Timeline export/import | Binary export/import |
|---|---|---|
| Windows (OPAQUE_WIN32) | ✅ XTX+B70 round-trip | ✅ queue signal/wait round-trip |
| Linux (OPAQUE_FD) | code-complete, needs hardware run | code-complete, needs hardware run |

## Tests

`tests/external_semaphore_test.cpp`: caps query always runs; the
export→dup→import→signal→wait round-trips run when a device with
`VK_KHR_external_semaphore` + the OS extension exists, else SKIP (exit 0).
CI runners take the SKIP path; the XTX+B70 box runs both round-trips.

## Future work (rest of Pillar 1)

- Backend-neutral completion tokens (R11 for HIP/L0/CUDA).
- Migration engine onto retirement; pipelined staging on top.
- Permanent-vs-temporary import policy for binary hand-off to foreign APIs.
