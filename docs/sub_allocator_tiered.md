# Chonk Sub-Allocator: Tiered Best-Fit Design

## Overview

The Chonk Buffer allocator (`python/vulkanvm_torch/allocator/chonk_allocator.cpp`)
maintains **two pools of Vulkan memory blocks**:

1. **Main slab** — `CHONK_MIN_BLOCK_GB` floor (default 4 GB since commit `f4bb69f`,
   was 2 GB originally, 16 GB during the chunk=2048 OOM), up to
   `CHONK_MAX_BLOCKS=24` warm blocks. Handles large persistent allocations:
   model weights, KV cache, merged INT4 quant buffers, and any allocation
   exceeding the largest sub-slab tier.

2. **Tiered sub-slabs** — four additional slabs at **512 MB, 1 GB, 2 GB, 4 GB**
   floors, up to `CHONK_SUB_MAX_BLOCKS=16` blocks each,
   `CHONK_SUB_WARM_BLOCKS=4` warm. Handle the 0–4 GB range with **best-fit**
   block sizing so small/medium allocations do not get rounded up to oversized
   main-slab blocks (waste 1–1.5 GB per block). All four tiers use the **same
   `slab::Core` (best-fit, aggressive release) code** as the main slab.

## Why tiered (not a single sub-slab)

The original design had one sub-slab with a 512 MB floor. Any allocation in the
513 MB – 2 GB range fell through to the main slab (2 GB+ floor) and was rounded
up — wasting up to ~1.5 GB per block. The tiered sub-allocator closes that gap:

- **512 MB tier**: requests ≤ 1 GB → block ≈ request (no waste up to 512 MB; ≤2× up to 1 GB)
- **1 GB tier**: requests 512 MB – 2 GB → block ≈ request
- **2 GB tier**: requests 1 GB – 4 GB → block ≈ request
- **4 GB tier**: requests 2 GB – 8 GB → block ≈ request (added to keep the
  2–4 GB range from bouncing to the main slab)

Routing rule (`sub::pick_tier(sz)`):

  pick the largest tier whose floor is ≤ 2 × sz

so the worst-case waste per block is 2× the request, but the typical waste is ~0%.

## Why `CHONK_MIN_BLOCK_GB=4` (not 16) for the main slab

The original `MIN_BLOCK_GB=16` was a workaround for the **2^31-byte boundary**
on the `p/scores` tensor at full 131K context (16 GB exactly). The log:
*"Use CHUNK_SIZE = 1024 default — 2048 froze the machine, 4096-chunk backward
caused kernel panic; 512/1024 validated stable"* and *"Free pooled quant
buffers... THE fix for the position-growth steps"*.

With the Granite attention **recompute** (`vulkanvm_attn_granite.py`),
`p/scores` is NOT saved in the autograd graph — it's recomputed in backward.
So the 2^31 boundary no longer threatens the persistent pool. The only
remaining 16 GB+ allocations are the transient `p/scores` during forward/backward
and the merged quant buffer (q=14.43 GB → 16 GB block). With `MIN_BLOCK_GB=4`,
the main slab creates **4 GB blocks** on demand instead of 16 GB, which means:

- The driver GTT heap (130 GB) can hold **32 blocks** of 4 GB vs 8 blocks of 16 GB.
- The 2nd-block request at the 2^31 boundary (p/scores transient) is **4 GB
  max**, well below the 2^31 cliff.
- The aggressive `releaseEmptyBlocks(0)` (added in commit `1bbb9c0`) frees
  empty blocks immediately so the GTT heap never holds dead 16 GB blocks.

## Alignment with training chunk sizes

The four tier floors (512 MB, 1 GB, 2 GB, 4 GB) are **deliberately aligned**
with the training chunk sizes (`CHONK_CHUNK` ∈ {256, 512, 1024, 2048}) so the
per-chunk allocation footprints map cleanly to the sub-allocator tiers:

| CHUNK | Activations | Quant scratch | Staging | Tier |
|-------|-------------|---------------|---------|------|
| 256   | 0.13 GB     | small         | 0.25 GB | 512 MB |
| 512   | 0.25 GB     | small         | 0.25 GB | 512 MB |
| 1024  | 0.5 GB      | small         | 0.25 GB | 1 GB |
| 2048  | 1.0 GB      | small         | 0.25 GB | 2 GB / 4 GB |

This keeps the allocator's "exact fit" promise end-to-end.

## Driver heap math (post fix)

- 130 GB GTT heap (`gttsize=124000`)
- 4 GB blocks: **~32 fit** (vs 16 GB blocks: ~8 fit)
- 4 sub-allocator tiers × up to 16 blocks each = up to 64 sub blocks
- 1 main slab × up to 24 blocks of 4 GB = up to 96 GB of main-slab capacity
- Total possible committed: ~160 GB (driver GTT caps this)

The 2^31 boundary transient `p/scores` at chunk=512 is 1×32×512×131072×2 =
**4.3 GB** (just over 4 GB), so it fits in a single 4 GB main-slab block.

## Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `CHONK_MIN_BLOCK_GB` | 4 | Main slab floor (4 GB closes the 2^31 gap) |
| `CHONK_MAX_BLOCKS` | 24 | Main slab max blocks |
| `CHONK_WARM_BLOCKS` | 8 | Main slab warm blocks |
| `CHONK_SUB_MAX_BLOCKS` | 16 | Per-tier sub-slab max blocks |
| `CHONK_SUB_WARM_BLOCKS` | 4 | Per-tier sub-slab warm blocks |
| `CHONK_SUB_MIN_BLOCKS` | 4 | Per-tier min blocks on OOM release |

## Files / commits

- `python/vulkanvm_torch/allocator/chonk_allocator.cpp` — `sub::pick_tier` + 4-tier
  `sub::get` + tiered `chonk_allocator_alloc` routing; best-fit `slab::Core::alloc`;
  aggressive `releaseEmptyBlocks(0)` on every alloc.
- `python/vulkanvm_torch/allocator/chonk_slab.cpp` — best-fit alloc + cooldown release.
- Commits: `e2c482b` (initial 512MB/1GB/2GB tiers) → `1bbb9c0` (best-fit + release)
  → `f4bb69f` (wrapper: CHUNK=256, r=64, SAVE=10) → (this update) MIN_BLOCK=4 +
  4th 4GB tier.

## Known limitations

- The global `PoolBlockProvider` is shared across all tiers + main slab, so a
  tier's nominal "floor" is enforced via slab's `minBlockBytes` parameter,
  not the provider's. The graduated ladder can still round up by 1-2 GB.
- The `.so` (`vulkanvm_pool_test.so`) must be rebuilt for the 4th tier and
  `MIN_BLOCK_GB=4` to take effect at runtime. Source is committed; binary
  rebuild is a one-time `bash scripts/build_pool_test.sh`.

## ChunkedPoolBuffer (fine-ladder exportable splitting)

The INT4 `q`/`s`/`z` buffers (q = 14.43 GB) were each ONE monolithic
`allocateDedicatedExportable` — a single 14.43 GB `vkAllocateMemory` the radv
driver rejects under load (the recurring `VK_ERROR_OUT_OF_DEVICE_MEMORY`).

`ChunkedPoolBuffer` + `ChonkPool.alloc_chunked()` split a large region into
many <= 1 GB exportable blocks (`CHONK_MAX_EXPORTABLE_MB`, default 1024). The
quant-write loop keeps working via `narrow(start, length)` (zero-copy view
within a block; contiguous copy across a boundary — rare, since each module's
qweight ~225 MB fits one block).

- 14.43 GB q buffer -> 15 x 1 GB blocks (was 1 monolithic)
- Model buffer (0.82 GB) stays single
- KV cache is already per-layer (0.5 GB each)
- Commit: (this change)

Combined with `CHONK_MIN_BLOCK_GB=8` + `POOL=1,2,4,8`, no single large
exportable allocation remains; the pool uses a fine, flat, graduated layout.
