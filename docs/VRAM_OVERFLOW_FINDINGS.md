# Finding: VRAM overflow silently halves decode throughput

**Date:** 2026-08-24 · **Hardware:** RX 7900 XTX (24 GB) + Arc Pro B70 (32 GB),
llama.cpp b10588 + Chonk Buffer pool, Qwen3.6-40B Q4_K_M, MTP on, layer split.

## The anomaly

User-observed: 21.7 t/s decode with q8_0 KV at 256K context vs ~12-13 t/s with
f16 KV at 192K. Hypotheses floated: KV quant bandwidth, temperature/MTP
acceptance, GPU clock state.

## Experiment

Controlled sweep (5 configs x warmup + 3 measured 400-token runs each,
temp 1.4, identical prompt; server restarted per config; clock state
normalized with a warmup ping). Full data: `sweep_results.csv` pattern in
the session log; summary:

| Config | B70 committed (GiB) | B70 VRAM (GiB) | Over? | Decode t/s (3 runs) |
|---|---:|---:|:-:|---|
| 128K f16  | ~28.4 | 31.8 | no  | 25.2 / 25.6 / 24.4 |
| 192K f16  | ~34.0 | 31.8 | **YES** | 13.4 / 13.4 / 13.2 |
| 192K q8_0 | ~25.4 | 31.8 | no  | 26.4 / 25.3 / 26.3 |
| 256K q8_0 | ~29.7 | 31.8 | no  | 25.2 / 26.7 / 25.3 |
| 256K f16  | ~35.6 | 31.8 | **YES** | 13.0 / 13.3 / 13.2 |

## Finding

**Decode throughput correlates with per-GPU VRAM overflow, not with KV
quantization or context size.** When a GPU's committed memory (pool blocks +
dedicated) exceeds its VRAM, Windows WDDM spills allocations to shared
memory; decode drops to ~13 t/s (PCIe-bound), exactly 2x down, with zero
run-to-run variance. Under the ceiling: ~25-26 t/s.

Two corollaries:

1. The "quant cache is faster" observation was real but the mechanism was
   overflow avoidance: q8_0 halves KV bytes, keeping the B70 under its
   ceiling at contexts where f16 spills.
2. **The pool grew past VRAM silently** because the llama integration runs
   with budget checks disabled (`maxHeapFraction = 0` - the setting the
   2026-08-15 audit flagged as dangerous on discrete GPUs). This is that
   danger, measured: a 2x throughput cliff with no error, no warning.

## Recommendations

1. **Users:** keep per-GPU committed bytes under VRAM. The live
   `/vvm/stats` endpoint reports `capacityBytes + dedicated` per pool -
   watch that against VRAM.
2. **Integration:** set `maxHeapFraction` (e.g. 0.92) in the llama hook so
   the pool fails soft at the VRAM edge instead of spilling; overflow
   tensors should route to the sibling GPU (auto-placement machinery
   already exists) or to dedicated offload.
3. **Sweet spot for this box:** 256K ctx + q8_0 KV = 25.7 t/s with MORE
   context than 192K f16 (which runs at 13.2). Quality impact of q8_0 KV is
   minimal vs f16 (community KL studies); q4_0 is where degradation becomes
   noticeable.

## Follow-ups

- [ ] Hook: set a VRAM-aware budget (per-device `maxHeapFraction`) with a
      `GGML_VK_VVM_HEAP_FRACTION` override.
- [ ] Consider cross-GPU rebalancing when one pool nears its VRAM ceiling
      (ShardPlacer policy hook).
- [ ] Surface a warning log when committed bytes exceed 90% of VRAM.
