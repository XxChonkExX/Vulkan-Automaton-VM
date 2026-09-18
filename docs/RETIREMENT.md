# GPU-Lifetime-Safe Reclamation (Retirement Queue)

CPU object lifetime is not GPU completion lifetime. `deallocate()` returns
memory to the buddy allocator immediately; if the GPU is still reading that
range (a submitted copy, a dispatched shader, an in-flight migration), the
next allocation can reuse the same physical memory while the GPU still sees
the old contents. The retirement queue closes this hole without forcing
every caller to block.

## Model

```
Allocation  +  timeline semaphore + value  -->  retire()
        |
        v
  retired_[]  (memory stays accounted, generation stays live)
        |
        v
  collect()  -- non-blocking; reclaims each item whose value has signaled
```

- `pool.retire(alloc, timeline, value, cmd = NULL, cmdPool = NULL)` hands
  ownership to the pool. The timeline stays caller-owned (pool never
  destroys it) and must have been created on the pool's `VkDevice` — use
  `retireTicket()` for a pool-owned timeline plus a fresh signal value.
  The optional cmd/cmdPool pair (both or neither) defers one-shot teardown
  (e.g. a transient copy command pool) to reclaim time.
- Returns `false` with no state changed when GPU tracking is unavailable
  (non-Vulkan backend, no transfer pool for cmd): the caller must wait
  itself and `deallocate()` normally.
- `pool.collect()` reclaims every signaled item (frees cmd, destroys
  transient cmdPool, deallocates through the normal generation-guarded
  path) and returns the reclaim count. Never blocks. Query failure (device
  lost) keeps items queued — leaking is safer than freeing against a dead
  device.
- The generation stays live across the retire window, so the
  double-free guard sees exactly one `deallocate()` at collect time, and
  `trim()`/`defragment()` (which never touch dedicateds or live buddy
  ranges) cannot free retired memory early.
- Pool destruction frees uncollected retired items directly (dedicated
  memory only; sub-allocated ranges belong to their blocks) and destroys
  the ticket timeline.

## Async copy contract (`copyDeviceToDevice`)

- Caller fence provided: genuinely asynchronous. One submit signals both
  the caller fence and the ticket timeline; the imported alias + command
  pool retire instead of waiting. The caller waits its fence whenever it
  wants; `collect()` (also called opportunistically on entry) reclaims.
- `VK_NULL_HANDLE` fence: synchronous (internal fence + wait + immediate
  teardown), as before.
- No ticket timeline (feature off): falls back to synchronous wait on the
  caller fence.

## Scope (v0.4)

Vulkan pools only (timeline semaphores). HIP/L0/CUDA pools return `false`
from `retire()`; their paths keep explicit waits. The offload/migration
engine keeps its fence waits for now — porting it onto retirement is
future work, as are a backend-neutral completion-token abstraction,
pipelined staging, and external semaphores.

## Tests

`tests/retirement_test.cpp` (25 checks, all passing on XTX + B70):
- A: retire against a never-signaled value — `collect()` reclaims nothing,
  memory stays accounted; scope exit exercises the dtor path.
- B: signaled ticket — `collect()` reclaims exactly 1, stats return to
  baseline, second `collect()` is a no-op (exactly-once, no double-free).
- C: `copyDeviceToDevice` with a caller fence returns while the fence is
  still unsignaled on a 256 MiB copy (async proof), data verifies after
  the fence, `collect()` reclaims the retired teardown.
