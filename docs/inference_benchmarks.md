# Inference Benchmarks — Chonk Buffer Pooling Study (Phase 0)

> **Date:** 2026-08-23 · **Machine:** Windows 11, Ryzen (zen4), dual dGPU
> **Goal:** Establish whether pooled multi-GPU inference (RX 7900 XTX + Arc Pro B70) beats single-card, and set the performance bar a Chonk-buffer-backed llama.cpp backend must clear.

---

## Hardware / Software

| Component | Detail |
|---|---|
| GPU 0 | AMD Radeon RX 7900 XTX — 24 GB, driver 32.0.31019.2002 |
| GPU 1 | Intel Arc Pro B70 — 32 GB, driver **32.0.101.8861 → upgraded 32.0.101.8974** |
| Excluded | AMD Radeon iGPU (uma), Tenstorrent TTVK Host Model |
| Inference | llama.cpp **b10588**, official Windows Vulkan binaries (`llama-bench`) |
| Models | Qwen3.6-40B MTP Q4_K_M (24.10 GiB, 39.5B params); qwen2 3B Q4_K_M (1.79 GiB) |

## Critical Finding #1 — the B70 driver was broken

On stock driver `32.0.101.8861` (Jul 2026 Pro branch), the B70 was catastrophically slow:

| Test | Old driver (8861) | New driver (8974) | Speedup |
|---|---|---|---|
| B70 solo, 3B model tg128 | not measured* | 157.31 t/s | — |
| B70 solo, 40B model pp512 | 6.62 t/s | 533.67 t/s | **80×** |
| B70 solo, 40B model tg128 | **0.74 t/s** | 18.01 t/s | **24×** |

*3B numbers were only collected post-upgrade.

**Note:** the official installer (`gfx_win_101.8974.exe`, SHA256 verified against Intel's published hash) failed its own integrity check with *"file integrity check failed"* despite byte-perfect download. Workaround: extract payload (7-Zip opens the NSIS container) and `pnputil /add-driver Graphics\iigd_dch.inf /install` + device restart. Worth reporting upstream if reproducible.

## Critical Finding #2 — pooling wins once both cards compute

40B Q4_K_M (24.10 GiB weights):

| Config | pp512 | tg128 | pp8192 | tg256@8K |
|---|---:|---:|---:|---:|
| XTX solo — 24 GiB weights > ~23.2 GiB free ⇒ PCIe spill | 446 t/s | 13.44 t/s | — | — |
| B70 solo — fits fully in 32 GB | 533.7 t/s | 18.01 t/s | 388 t/s | 17.93 t/s |
| Layer-split both (default) | 508.8 t/s | 20.65 t/s | **583 t/s** | **20.68 t/s** |
| **Layer-split, bandwidth-weighted `-ts 1.43/1`** | 474 t/s | **21.89 t/s** | — | — |

- Pooled decode beats every solo config: **+54% over XTX solo, +21% over B70 solo**.
- At 8K context the split's prefill advantage *grows* (583 vs 388): each card reads fewer weight bytes/token and both engines stay busy.
- Tuning split ratio by memory bandwidth (~960 vs ~672 GB/s ≈ 1.43:1) adds ~6% decode.
- The XTX-only run confirms the spill penalty predicted by VRAM math: 13.44 vs expected ~30+ t/s resident. KV cache makes solo-XTX untenable for this model — exactly the "10 lbs in a 5 lb sack" problem.
- Small-model sanity (qwen2 3B): XTX solo 265.7 t/s tg128, B70 solo 157.3, naive split 174.3 — when either card alone holds the whole model, splitting adds pipeline overhead. **Pooling pays when the model exceeds one card's comfortable capacity, not before.**

## Why this matters for the Chonk Buffer

Stock ggml layer-split already reaches 21.9 t/s pooled. **That is the bar.** A Chonk-backed backend earns its keep by going beyond layer-granularity:

1. **Tensor-granular placement** via `ShardPlacer` (hot experts/layers on fastest memory, cold elsewhere) instead of whole-layer rows.
2. **KV cache in-pool** with offload tiers (`offloadToHost`/`reloadToDevice`) as DMA-driven third tier.
3. **Fragmentation-free packing** — proven at 116 GB sustained in training (OPTIMIZATION_LOG.md).
4. Later: X2 Strix Halo as an RDMA weight/KV node over the verified verbs path.

## Implementation decision

No long-lived hard fork needed conceptually — but since llama.cpp has no plugin ABI, the practical form is a **branch of a cloned llama.cpp containing a tiny patch surface**:

- NEW FILE: `ggml_backend_vvm_buffer_type` sourcing tensor buffers from `vvm::UnifiedMemoryPool` (the `allocateTensor()` API already returns exactly what ggml needs: VkBuffer + offset + SHADER_DEVICE_ADDRESS + transfer usages).
- HOOK: route ggml-vulkan's per-tensor buffer creation through the VVM buft (~1 call site).
- Sync strategy: keep patch ≤ 2 files, rebase regularly on upstream tags.

## Phase plan

- [x] **Phase 0** — baseline benches (this document)
- [x] **Phase 1** — VVM buffer type, parity achieved (see below)
- [ ] **Phase 2** — tensor placement across both cards vs layer rows; KV-in-pool
- [ ] **Phase 3** — offload tiers between requests
- [ ] **Phase 4** — RDMA memory node (X2 Strix Halo)

---

## Phase 1 results — Chonk-backed ggml buffer (2026-08-23)

Integration: branch `chonk-buffer` of llama.cpp @ `b10588`. `ggml_backend_buffer_type_alloc_buffer` routes through `vvm::UnifiedMemoryPool` when `GGML_VK_VVM_POOL=1`; `get_max_size` capped to the pool block size so ggml chunks its reservations into Chonk-sized blocks.

### The debugging ladder (each row is a real measurement)

| Step | B70 solo pp512 | B70 solo tg128 | Finding |
|---|---:|---:|---|
| Control (ggml native) | 531.66 | 17.96 | the bar |
| Pool v1 (2 GiB blocks) | 16.44 | 6.00 | 32x prefill collapse |
| + max-size cap (ggml chunks to 2 GiB) | 532.49 | 6.68 | prefill fixed; decode still 2.7x down |
| + exact-fit buddy (1 GiB blocks) | 532.49 | 6.68 | (block size was the prefill fix) |
| **1 GiB blocks** | **532.49** | 6.68 | prefill parity — 2 GiB allocations placed badly on Intel |
| + memory priority (VK_EXT_memory_priority) | 16.46 | 6.02 | not the cause (opt-in env; off in both paths) |
| **+ pure DEVICE_LOCAL (no ReBAR type)** | **533.57** | **17.84** | **parity — root cause was ReBAR-mapped pool blocks** |

### Root causes found (in order)

1. **Allocation size placement (Intel):** 2 GiB VkDeviceMemory allocations land badly on the Arc driver — 32x prefill collapse. 1 GiB (ggml's own suballocation granularity) fixes it. Fix: pool blockSize 1 GiB + `get_max_size` cap.
2. **ReBAR-mapped pool blocks (Intel):** pool blocks in the DEVICE_LOCAL|HOST_VISIBLE type lose ~3x decode bandwidth vs pure DEVICE_LOCAL. Control ggml uses the same ReBAR type but *mapped* (for upload); unmapped ReBAR blocks get degraded residency. Fix: `preferPureDeviceLocal` — pool excludes host-visible types. (ReBAR remains valuable for CPU->GPU upload paths, consistent with community guidance for Arc.)
3. **`maxBlocks = 0` doc/code mismatch (repo bug):** documented "unlimited", treated as zero. Fixed in `unified_memory_pool.cpp`.
4. **`UniqueAllocation::make` never defined (repo bug):** declared, never defined anywhere; link error for any RAII user. Defined now.
5. **Buddy power-of-2 rounding waste:** a 1.1 GB request consumed a 2 GB order. Fixed with **exact-fit grants**: grant `alignUp(size, minAlignment)` and return the tail to the free lists as buddy-aligned chunks; unified free path decomposes any grant (full power-of-two grants behave exactly as the classic buddy). Bounded waste: 256 KB instead of up to 2x.
6. **`splitTo` cleanup** (auditor item): pop source once, free only the right buddy per level.
7. **MSVC portability:** `VVM_API` was GCC-only `__attribute__`; `UniqueAllocation::make` and `BuddyAllocator` not exported from the DLL; root CMakeLists had a SHARED-lib dependency cycle (`vulkan_vm -> vulkan_vm_tensor -> network -> vulkan_vm`). All fixed; `placement_executor.cpp` now only built with network ON.

### Final numbers — 40B Q4_K_M (llama.cpp b10588, Vulkan, FA=1)

| Config | pp512 | tg128 | vs control |
|---|---:|---:|---|
| Control: split default | 520 | 20.33 | — |
| **Chonk: split default** | 499.75 | **19.80** | -2.6% |
| Control: split `-ts 1.43/1` | 474 | 21.89 | — |
| **Chonk: split `-ts 1.43/1`** | 478.47 | **20.86** | -4.7% |
| Control: B70 solo | 531.66 | 17.96 | — |
| **Chonk: B70 solo (pure local)** | 533.57 | **17.84** | -0.7% |
| Control: XTX solo | 446 | 13.44 | (spills to PCIe) |

Small-model sanity (qwen2 3B): XTX solo pool 248.9 vs control 269 tg128 (-7%); B70 solo pool 150.2 vs 155.7 (-3.5%). The small residual decode gap on sub-block-sized models is the cost of 256 KB-aligned sub-allocation placement; it vanishes at scale (the case the pool exists for).

### Runtime knobs

- `GGML_VK_VVM_POOL=1` — enable Chonk Buffer allocations
- `GGML_VVM_BLOCK_SIZE=<bytes>` — pool block size (default 1 GiB; must be power of 2, >= 64 MiB)
- `GGML_VVM_PURE_LOCAL=0` — allow ReBAR types (default: pure DEVICE_LOCAL)
- `GGML_VK_ENABLE_MEMORY_PRIORITY=1` — ggml's own opt-in priority (pool honors `PoolConfig::memoryPriority` when enabled)

### Buddy allocator hardening (repo)

- **Exact-fit grants** with tail decomposition (`pushFreeRange`); `coalesce()` contract documented (expects not-yet-pushed blocks)
- **`checkInvariants`** extended for exact-fit entries (minSize alignment, granted-size bounds)
- New tests: exact-fit grant/reuse, waste bound, ggml-style churn (6+6+2+1 MB mixed-order free, full coalescing recovery), 2000-iter exact-fit stress — all passing alongside the original suite

---

## Phase 1.5 — Chonk Chunks: small-block routing (2026-08-23)

Three routing upgrades to the pool, aimed at the small-allocation weakness:

1. **Best-fit block selection** — `allocate()` now picks the block with the *smallest* sufficient largest-free within the size class, instead of first-fit. Packs tightly; stops small allocations from spawning extra partially-filled blocks.
2. **`allocationAlignment` (buffer-base alignment)** — allocation *starts* can be aligned to driver memory-page boundaries (2 MB default in the llama integration). Leading slack + tail are returned to the free lists; waste stays bounded. Implemented as `BuddyAllocator::allocateAligned()` (over-allocate → align → free slack), no buddy-core changes.
3. **Chonk Chunks (size-class routing)** — requests ≤ `smallAllocThreshold` (16 MiB) are served from dedicated 64 MiB chunk blocks instead of claiming space in (or spawning) full-size 1 GiB blocks. A small tensor now costs 64 MiB of VRAM footprint, not 1 GiB.

### Results (llama.cpp b10588 + Chonk pool, pure-DEVICE_LOCAL)

| Case | ggml control | Pool v2 (parity work) | **Pool v3 (Chonk Chunks)** | Gap |
|---|---:|---:|---:|---|
| 40B split default (tg128) | 20.33 | 19.80 (−2.6%) | **20.16** | **−0.8%** |
| 40B split `-ts 1.43/1` (tg128) | 21.89 | 20.86 (−4.7%) | **21.22** | **−3.1%** |
| B70 3B solo (tg128) | 155.68 | 150.18 (−3.5%) | **155.80** | **+0.1% (parity)** |
| XTX 3B solo (tg128) | 269.0 | 249.65 (−7%) | 249.90 | −7% (XTX residual) |

Findings:
- **B70 small-model gap eliminated** (−3.5% → parity). The 2 MB buffer-base alignment was the fix there.
- **40B split improved** to −0.8%/−3.1%.
- **XTX 3B residual (−7%) is XTX-specific**: unaffected by base alignment and by pure-vs-ReBAR type. Suspects: descriptor/scheduling overhead on the primary device, or AMD-specific sub-allocation placement. Open item.
- Chunk routing's win is *capacity*, not t/s on these benches: a 18 MB ggml buffer now costs a 64 MiB chunk instead of a 1 GiB block — matters for multi-model residency and KV-heavy workloads.

### Runtime knobs (llama integration)

- `GGML_VVM_BASE_ALIGN=<bytes>` — buffer-base alignment (default 2 MiB; `0` = minAlignment)
- `GGML_VVM_CHUNK_MB=<n>` — chunk block size in MiB (default 64; `0` disables routing)
- `GGML_VVM_BLOCK_SIZE=<bytes>` — regular block size (default 1 GiB)

---

## Phase 2 - XTX residual SOLVED: the DEVICE_ADDRESS allocate flag (2026-08-23)

The remaining XTX 3B decode gap (-7%) survived an exhaustive elimination matrix:
memory type (ReBAR/pure), block size (256 MB/512 MB/1 GiB), base alignment
(256 KB/2 MB), dedicated hint, memory priority, command pool, initial block,
run order, and even **pass-through mode with byte-identical dedicated
allocations**. Interleaved A/B/A confirmed it was deterministic (250 vs 268,
no overlap) and not driver-state flakiness.

**Root cause**: `VkMemoryAllocateFlagsInfo` with `DEVICE_ADDRESS` on the
*memory allocation*. ggml always sets both the buffer's
`SHADER_DEVICE_ADDRESS` usage **and** the allocate-time flag; the pool had
only partially matched (usage flag present, allocate flag missing). On AMD
RDNA3, memory allocated without the device-address flag is placed differently
by the driver - worth ~7% of streaming-read bandwidth. Intel Arc was
unaffected (its issue was allocation *size* placement, fixed separately).

**Fix**: `pcfg.enableDeviceAddress = device->buffer_device_address` - the
pool now mirrors ggml exactly: usage flag AND allocate flag whenever the
device has the feature.

### Final matrix - all cases at parity

| Case | ggml control (pp512 / tg128) | Chonk Buffer | Gap |
|---|---:|---:|---|
| 40B split default | 520 / 20.33 | 504.4 / 19.93 | -2.0% |
| 40B split `-ts 1.43/1` | 474 / 21.89 | 486.8 / 21.50 | -1.8% |
| 40B B70 solo | 531.7 / 17.96 | 533.6 / 17.84 | -0.7% |
| XTX 3B solo | 5729 / 269.16 | 5626 / 266.96 | -0.8% |
| B70 3B solo | - / 155.68 | - / 155.83 | +0.1% |

### Robustness fixes added along the way

- **Aligned-grant fallback**: requests whose alignment padding overflows a
  full block (e.g. ~1 GiB chunk + 2 MiB alignment in a 1 GiB block) fall
  back to unaligned grants instead of failing.
- **Dedicated fallback**: if sub-allocation fails even in a fresh block, the
  pool serves the request from dedicated memory.
- **Best-fit fallthrough**: a failed sub-allocation in the best-fit block now
  falls through to new-block creation instead of failing the allocation.
- Diagnostics (env-gated, default off): `VVM_SKIP_CMDPOOL`,
  `VVM_SKIP_INITBLOCK`, `GGML_VVM_PASSTHROUGH_ALLOC`,
  `GGML_VVM_NO_DEDICATED`.

Elimination matrix (for posterity): the XTX gap was invariant to memory type,
block size, base alignment, dedicated hint, memory priority, command pool,
initial block, and run order - only the DEVICE_ADDRESS allocate flag moved it.

Research notes: AMD GPUOpen recommends ~256 MB allocations on Windows (WDDM
per-allocation costs); RDNA Performance Guide recommends staying under 80%
heap usage to avoid eviction; llama.cpp issue #22646 documents probabilistic
AMD Windows driver slow states (driver reset recovers) - worth remembering
when chasing "impossible" perf regressions. TheRock (ROCm 7.9+) is AMD's new
unified build super-repo; Linux-centric, not applicable to the Windows
Vulkan path used here.

### Consolidated runtime knobs (llama integration)

| Knob | Default | Effect |
|---|---|---|
| `GGML_VK_VVM_POOL` | `0` | `1` enables Chonk Buffer allocations |
| `GGML_VVM_BLOCK_SIZE` | `1 GiB` | regular block size (pow2, >= 256 KiB; 256 KiB = pass-through/dedicated mode) |
| `GGML_VVM_BASE_ALIGN` | `2 MiB` | buffer-base alignment (`0` = minAlignment) |
| `GGML_VVM_CHUNK_MB` | `64` | chunk block size in MiB (`0` disables size-class routing) |
| `GGML_VVM_PURE_LOCAL` | on | `0` allows ReBAR (host-visible) VRAM types |
| `GGML_VVM_NO_DEDICATED` | off | `1` drops the dedicated-allocate hint |
| `GGML_VVM_PASSTHROUGH_ALLOC` | off | `1` pool init runs, buffers via ggml native path |
| `VVM_SKIP_CMDPOOL` / `VVM_SKIP_INITBLOCK` | off | `1` skips those init steps (diagnostics) |
