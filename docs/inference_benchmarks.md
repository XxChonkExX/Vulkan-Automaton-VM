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
- [ ] **Phase 1** — VVM buffer type, parity on 3B (bit-comparable outputs, t/s within noise, zero per-tensor `vkAllocateMemory`)
- [ ] **Phase 2** — tensor placement across both cards vs layer rows; KV-in-pool
- [ ] **Phase 3** — offload tiers between requests
- [ ] **Phase 4** — RDMA memory node (X2 Strix Halo)
