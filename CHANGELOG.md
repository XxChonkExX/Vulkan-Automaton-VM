# Changelog

## v0.5.0 (in progress)

### Semaphores & completion (Pillar 1)

- **External semaphores** (`vulkan_vm/cross_gpu/external_semaphore.hpp`):
  export/import `VkSemaphore` payloads as OS handles (`OPAQUE_FD` /
  `OPAQUE_WIN32`) for cross-device and cross-process GPU sync. Timeline
  CPU signal/wait, binary queue round-trip, R4 consume-on-import
  ownership. Verified live on XTX+B70 (both kinds PASS);
  `tests/external_semaphore_test.cpp` skips cleanly without a capable
  device. (`docs/EXTERNAL_SEMAPHORES.md`)
- **Backend-neutral completion tokens** (`vulkan_vm/completion_token.hpp`):
  one retire/collect gate for every backend - `Ready` (no device needed),
  `VulkanTimeline` (the classic path, now delegated), `Foreign`
  (caller-supplied non-blocking consult, e.g. a hipEventQuery poll).
  Plus `retireToken()`, a CPU-only `completion_token_test`, and 14 new
  token checks in `retirement_test` (39 total, green on B70).
- **Migration engine on tokens**: every `MigrationOperation` carries its
  submit's timeline as `op.completionToken` (retire against it directly);
  `pollMigration()` consults the token instead of the fence, leaving
  fence waits only on the blocking path and context reuse. New
  `migration_token_test` (device-gated): token presence, poll/consult
  agreement, and a host->device->host pattern round-trip - green on
  hardware, 0 failures.
- **Pipelined host-staged transfers**: `copyDeviceToDeviceHostStaged`
  double-buffers chunk slots (slot-local cmd pools/fences) so the dst
  DMA leg overlaps the next chunk's src leg + memcpy; caller fence now
  rides the final chunk only (was submitted per-chunk). Measured
  B70<->XTX 256 MiB: 3.75 -> 8.61 GiB/s (2.3x) and 3.89 -> 7.19 GiB/s
  reverse, data verified, zero validation errors. Opt out with
  `VVM_STAGED_PIPELINE=0`.
- **Runtime rebalancing**: `MultiGPUPoolManager::migrateAllocation`
  (allocate on dst, copy, Ready-token retire of src, rollback on any
  failure) + `poolPressures()` (used/budget/driver-used per instance
  for policy brains). Verified live 0->1->0 with byte checks, pressure
  sanity, and retired-src reclamation (`multi_gpu_test` section M).
- **`VVM_PREFER_PURE_DEVICE_LOCAL`**: env knob for the existing
  ReBAR-exclusion swap (largest-heap pure type now a tested
  `findLargestHeapPureDeviceLocal` in utils + `memory_type_test`).
- **Fenced `copyBuffer` teardown fix (R11)**: the async path destroyed
  its transient command pool under in-flight work
  (VUID-vkDestroyCommandPool-commandPool-00041, reproduced live under
  validation). The submit now also signals a ticket timeline and the
  pool retires via the new `retireCommandPool()` (cmd-only retirement
  items); without timeline support it degrades to synchronous. Pool
  destruction reaps uncollected retired cmd pools. New section E in
  `retirement_test` (60 checks total, green on B70).

## v0.4.0 (2026-09-18)

Memory-lifetime correctness, cross-vendor hardening on the returned B70,
and release packaging. Full suite green (36 binaries; only skip is CUDA
with no NVIDIA hardware on box). CI: Linux GCC/Clang (core + full),
Windows MSVC (core + full), Android compile - all green.

### Lifetime & ownership (new contract rule R11)

- **Pool retirement queue** (`pool.retire()` / `collect()` / `retireTicket()`):
  GPU-lifetime-safe reclamation backed by timeline semaphores. Memory
  returns to the buddy allocator only after `collect()` observes the
  signaled value. Vulkan pools; other backends fail soft.
  (`docs/RETIREMENT.md`, `tests/retirement_test.cpp` — 25 checks)
- **`copyDeviceToDevice` is genuinely async on a caller fence**: signals
  the caller fence + a ticket timeline in one submit and retires the whole
  teardown instead of waiting. Sync path unchanged.
- **Exactly-once migration completion**: `waitMigration`/`pollMigration`
  share one consult-and-erase gate — double `onComplete` (shadow-region
  double-free class) is now impossible by construction.
- Normative `LIFETIME_CONTRACT.md` gains rule R11 (retired allocations stay
  live until GPU completion).

### Allocator hardening

- Saturating size arithmetic (`satAddU64`): hostile request sizes fail soft
  instead of wrapping to small grants.
- Block-geometry validation boundary (pow2 block, `block >= minAlignment`).
- Buddy waste metric records the true request size; concurrent fuzz
  upgraded (8 threads, random sizes/alignments, mid-run invariant checks,
  `VVM_BUDDY_FUZZ_ITERS` scale knob).

### Cross-vendor P2P (Windows)

- Direct import is same-vendor-only: AMD→NVIDIA refuses gracefully (-13),
  AMD→Intel **crashes inside the driver** — both now route to host-staged
  (3.5 GiB/s symmetric XTX↔B70) instead of feeding the attempt to the
  driver. `VVM_P2P_POLICY=host|force-direct` override for bring-up.
- `allocateDistributed` allocates dedicated-exportable on master (was
  structurally unable to export sub-allocations).
- Shared host arena verified on AMD↔Intel: 5.2 GiB/s zero-copy,
  6.1 GiB/s via the manager path (new records).
- Level Zero backend verified on the Arc Pro B70 (discovery, alloc plane,
  full pool over L0).

### Network hardening (THREAT_MODEL §5)

- Header slow-loris bound: tight scoped receive timeout for the 32-byte
  header on both paths (5 s, restores previous).
- Chunked client body/stream staging (was server-side only).
- Same-process zero-copy policy now refuses cross-vendor on Windows too
  (was Linux-only; `network_test` goes green via staged fallback + designed
  SKIP instead of crashing).
- `exportForRemote` promotion falls back to device-local when drivers
  refuse host-visible+exportable (Intel Arc).
- `udp_verb` Windows init fixed (WSAStartup, `INVALID_SOCKET` check,
  `WSAGetLastError`) plus a WRITE-contiguity off-by-one that dropped one
  chunk per ~3 runs: 14 failures → 10/10 clean.

### Storage

- `IoRingBackend::close` drains in-flight ring ops on both platforms
  (teardown-while-DMA use-after-free class).
- `storage_e2e_stream_test` self-generates its pack (CI-ready, no args).

### Cross-platform CI (release-day hardening)

- Linux GCC + Clang green (core + full): LP64 lambda return-type fix,
  `fread` short-read checks, switch exhaustiveness, dead deprecated
  `OffloadConfig` members removed.
- Test portability: `buddy_test` CHECK arity, `std::filesystem` temp dir
  in `storage_e2e_stream_test`, sequenced fuzz counter in
  `storage_queue_test` (genuine unsequenced UB), portable
  VirtualAlloc/mmap arenas in `shared_arena_test` - and the
  manager-owned `createSharedArena(size)` overload now builds on Linux.
- C++20 `[=, this]` captures; duplicate `VVM_BUILD_SHARED` definition
  dropped from the network target.

### Diagnostics & packaging

- New `vvm-info` tool: devices, heaps, queues, features, external caps,
  P2P policy matrix. External-memory query API exported from the DLL.
- `VVM_LOG_LEVEL` env control with zero-cost filtered logging.
- `docs/ENV_VARS.md`: single reference for all knobs.
- `CONTRIBUTING.md`, status banner, experiments quarantine,
  `[[nodiscard]]` on factories (caught one real ignored failure).

## v0.3.0

See `docs/0.3_RELEASE_NOTES.md` and `OPTIMIZATION_LOG.md`: layer
separation, exact-fit buddy + Chonk Chunks, llama.cpp parity, Android
AHardwareBuffer, Linux cross-vendor P2P + verbs campaign.
