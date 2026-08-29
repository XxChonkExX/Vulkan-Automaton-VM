# Chonk Sub-Allocator: Tiered Best-Fit Design

## Overview

The Chonk Buffer allocator (`python/vulkanvm_torch/allocator/chonk_allocator.cpp`)
maintains **two pools of Vulkan memory blocks**:

1. **Main slab** — `CHONK_MIN_BLOCK_GB` floor (default 2 GB), up to `CHONK_MAX_BLOCKS=24`
   warm/warm blocks. Handles large persistent allocations: model weights, KV cache,
   merged INT4 quant buffers.

2. **Tiered sub-slabs** — three additional slabs at **512 MB, 1 GB, 2 GB** floors, up to
   `CHONK_SUB_MAX_BLOCKS=16` blocks each, `CHONK_SUB_WARM_BLOCKS=4` warm. Handle the
   0–2 GB range with **best-fit** block sizing so small/medium allocations do not
   get rounded up to oversized main-slab blocks (waste 1–1.5 GB per block).

## Why tiered (not a single 512 MB sub-slab)

The previous design had one sub-slab with a 512 MB floor. Any allocation in the
513 MB – 2 GB range fell through to the main slab (2 GB+ floor) and was rounded
up — wasting up to ~1.5 GB per block. The tiered sub-allocator closes that gap:

- **512 MB tier**: requests ≤ 1 GB → block ≈ request (no waste up to 512 MB; ≤2× up to 1 GB)
- **1 GB tier**: requests 512 MB – 2 GB → block ≈ request
- **2 GB tier**: requests 1 GB – 4 GB → block ≈ request

Routing rule (`sub::pick_tier(sz)`):

  pick the largest tier whose floor is ≤ 2 × sz

so the worst-case waste per block is 2× the request (same as the main slab
itself), but the typical waste for 512 MB / 1 GB / 2 GB requests is ~0%.

## Alignment with training chunk sizes

The three tier floors (512 MB, 1 GB, 2 GB) are **deliberately aligned** with the
training chunk sizes (`CHONK_CHUNK` ∈ {256, 512, 1024, 2048}) so the per-chunk
allocation footprints map cleanly to the sub-allocator tiers:

| CHUNK | Activations | Quant scratch | Staging | Tier |
|-------|-------------|---------------|---------|------|
| 256   | 0.25 GB     | small         | 0.25 GB | 512 MB |
| 512   | 0.5 GB      | small         | 0.25 GB | 512 MB / 1 GB |
| 1024  | 1.0 GB      | small         | 0.25 GB | 1 GB |
| 2048  | 2.0 GB      | small         | 0.25 GB | 2 GB |

This keeps the allocator's "exact fit" promise: the block a chunk's tensors land
in has the same shape as the chunk itself, end-to-end.

## Driver heap cap

The Vulkan driver exportable heap is ~40 GB (per `OPTIMIZATION_LOG.md`). The
tiered sub-allocator must respect this:

- 512 MB blocks: ~80 fit (rare; mostly used for 0.25–0.5 GB activations/staging)
- 1 GB blocks: ~40 fit
- 2 GB blocks: ~20 fit
- 16 GB main blocks (when `CHONK_MIN_BLOCK_GB=16`): ~2 fit

`CHONK_SUB_MAX_BLOCKS=16` per tier × 3 tiers = up to 48 sub blocks. In practice
the driver heap limits total concurrent exportable blocks, so the sum across
tiers + main is the real ceiling.

## Configuration (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `CHONK_MIN_BLOCK_GB` | 2 | Main slab floor |
| `CHONK_MAX_BLOCKS` | 24 | Main slab max blocks |
| `CHONK_WARM_BLOCKS` | 8 | Main slab warm blocks |
| `CHONK_SUB_MAX_BLOCKS` | 16 | Per-tier sub-slab max blocks |
| `CHONK_SUB_WARM_BLOCKS` | 4 | Per-tier sub-slab warm blocks |
| `CHONK_SUB_MIN_BLOCKS` | 4 | Per-tier min blocks on OOM release |

## Files / commits

- `python/vulkanvm_torch/allocator/chonk_allocator.cpp` — `sub::pick_tier` + tiered `sub::get` + tiered `chonk_allocator_alloc` routing.
- Commit: `e2c482b` — initial tiered sub-slab (512 MB / 1 GB / 2 GB).
- Original 512 MB-only sub-slab: commit `8902e2b`.

## Known limitations

- The global `PoolBlockProvider` is shared across all tiers + main slab, so a
  tier's nominal "floor" (e.g. 1 GB) is not strictly enforced — the bucket ladder
  can still round up. For exact tier floors, a per-tier provider would be needed
  (in practice the graduated ladder makes the difference small).
- The `.so` (`vulkanvm_pool_test.so`) must be rebuilt for the tiered routing to
  take effect at runtime. Source is committed; binary rebuild is a one-time
  `cmake --build . --target vulkanvm_pool_test` + copy to `_build/`.
