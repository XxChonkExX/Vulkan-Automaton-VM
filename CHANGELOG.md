# Changelog

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
