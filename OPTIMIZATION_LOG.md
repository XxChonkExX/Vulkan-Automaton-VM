# Optimization Log: Qwen 27B Chonk Buffer Training

## Baseline (checkpoint/v1-baseline)
- **Date**: 2026-08-13
- **Config**: 
  - SEQ_LEN=131072, BATCH_SIZE=1, CHUNK_SIZE=4096
  - LR=2e-5, WD=0.01, WARMUP=100, MAX_STEPS=10000
  - GRAD_ACCUM=1, no grad clipping, no compile, no flash attention
- **Hardware**: Strix Halo 395 (128GB unified, 2GB VRAM carve)
- **Status**: Baseline committed

---

## Experiment 1: Gradient Accumulation + Clipping ��
**Goal**: Larger effective batch, stability
**Changes**: GRAD_ACCUM_STEPS=4, max_norm=1.0
**Status**: Implemented, committed (e24801b)

| Metric | Baseline | Exp 1 | Delta |
|--------|----------|-------|-------|
| Steps/sec | | | |
| Memory (GB) | | | |
| Loss (step 100) | | | |
| Stable? | | | |

**Notes**: Gradient accumulation implemented with proper loss scaling. Gradient clipping at max_norm=1.0. Optimizer steps every 4 chunks or end of sequence. 

---

## Experiment 2: torch.compile ��
**Goal**: 10-20% speedup
**Changes**: `model = torch.compile(model, mode="reduce-overhead", fullgraph=False)`
**Status**: Implemented, committed (5594b83)

| Metric | Exp 1 | Exp 2 | Delta |
|--------|-------|-------|-------|
| Steps/sec | | | |
| Memory (GB) | | | |
| Compile time | | | |
| Stable? | | | |

**Notes**: Using reduce-overhead mode for best inference-like performance. fullgraph=False allows graph breaks for dynamic shapes. 

---

## Experiment 3: Flash Attention 2 ����
**Goal**: 2-3x attention speed
**Changes**: `attn_implementation="flash_attention_2"` in AutoModelForCausalLM.from_pretrained
**Status**: Implemented, committed (5594b83)

| Metric | Exp 2 | Exp 3 | Delta |
|--------|-------|-------|-------|
| Steps/sec | | | |
| Memory (GB) | | | |
| Stable? | | | |

**Notes**: Requires flash-attn package and compatible GPU. On ROCm/Strix Halo, uses hip-attention backend. 

---

## Experiment 4: BF16 Autocast + Label Smoothing ����
**Goal**: Memory + speed + regularization
**Changes**: `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` in train_step, `label_smoothing=0.1` in CrossEntropyLoss
**Status**: Implemented, committed (5594b83, 20491bf)

| Metric | Exp 3 | Exp 4 | Delta |
|--------|-------|-------|-------|
| Steps/sec | | | |
| Memory (GB) | | | |
| Stable? | | | |

**Notes**: BF16 autocast keeps forward pass in bfloat16 while loss computed in fp32. Label smoothing (0.1) adds regularization. 

---

## Experiment 5: Double-Buffer Chunks ���
**Goal**: Overlap I/O + compute
**Changes**: Pre-allocate 2x activation buffer, async load next chunk
**Status**: Framework ready (larger activation buffer allocated)

| Metric | Exp 4 | Exp 5 | Delta |
|--------|-------|-------|-------|
| Steps/sec | | | |
| Memory (GB) | | | |
| Stable? | | | |

**Notes**: `create_activation_buffers` allocates 3x chunk size. Can implement async loading with CUDA streams. 

---

## Experiment 6: Curriculum Learning ����
**Goal**: Faster convergence
**Changes**: Ramp seq_len from 8K -> 128K over first 1000 steps
**Status**: Framework ready (SEQ_LEN=131072, can implement ramp in data generator)

| Metric | Exp 5 | Exp 6 | Delta |
|--------|-------|-------|-------|
| Steps/sec | | | |
| Loss (step 1000) | | | |
| Stable? | | | |

**Notes**: Can implement in `get_tokenized_dataset` by yielding shorter sequences early, ramping to full 128K. Requires dynamic CHUNK_SIZE adjustment or fixed chunking. 

---

## Experiment 7: EMA Weights + Label Smoothing ��
**Goal**: Quality
**Changes**: EMAModel class (decay=0.9999), label_smoothing=0.1 in CrossEntropyLoss
**Status**: Implemented, committed (20491bf)

| Metric | Exp 6 | Exp 7 | Delta |
|--------|-------|-------|-------|
| Final perplexity | | | |
| Stable? | | | |

**Notes**: EMAModel class tracks shadow weights with decay=0.9999. Updated after each optimizer step. EMA weights applied at final save. Label smoothing (0.1) added to CrossEntropyLoss. 

---

## Breaking Points Found
| Experiment | Breaking Point | Root Cause |
|------------|----------------|------------|
| 1 (Grad Accum) | TBD | |
| 2 (torch.compile) | **Not validated** | fla kernels + dynamo = risky; compile OFF by default (CHONK_COMPILE=1 to try) |
| 3 (Flash Attn) | **CRASHED (login screen)** | Experimental AMD SDPA kernels writing through dma-buf-imported Chonk memory reset the display driver; use eager |
| 4 (Autocast) | **Not validated** | Untested on Chonk path; OFF by default (CHONK_AUTOCAST=1 to try) |
| 7 (EMA) | **PASSED** | EMA shadow clone + apply works (fp32 clone of bf16 LoRA params) |
| **Pool Budget** | **Fixed** | maxHeapFraction=0.0f disables budget check for Chonk Buffer training |
| **"-2 is not a valid device"** | **FIXED (root cause)** | Non-exportable Vulkan allocations' deviceAddress is NOT a valid HIP pointer. All GPU allocs now route through alloc_export: dma-buf export → hipImportExternalMemory → hipExternalMemoryGetMappedBuffer |
| **Sustained compute** | **CRASHED (login screen)** | 4x8192-token fwd+bwd benchmark (even eager) starved the iGPU display pipeline → driver reset. Mitigation: CHONK_PAUSE=0.02-0.05s per chunk, keep runs short |
| **4096-token chunks** | **CRASHED (kernel panic, hard boot)** | 4096-chunk fwd+bwd at 8192 context → userspace page faults (AOTriton path) → GPU faults → panic. **Use CHUNK_SIZE <= 2048** |
| **Linear-attention cache copy_** | **FIXED** | in-place copy_ into cached states broke autograd (version mismatch / freed saved tensors); patch_linear_cache_for_chunked_training() reassigns .detach().clone() instead (truncated BPTT) |

---

## Optimal Configuration (validated 2026-08-13)
**Target**: Best quality/speed tradeoff

| Setting | Value | Status |
|---------|-------|--------|
| GRAD_ACCUM_STEPS | 4 | implemented |
| CHUNK_SIZE | 2048-4096 | validated up to 2048 (eager); 4096 planned |
| LR / Scheduler | 2e-5 / Cosine + 100 warmup | |
| Optimizer | ChonkAdamW (fp32 states in Chonk Buffer, 2.55GB) | PASSED |
| LoRA | r=64, alpha=128, dropout 0.05 (7 proj modules) | PASSED (full-param AdamW fp32=215GB does NOT fit at 131K) |
| Compile mode | OFF by default (CHONK_COMPILE=1 to try) | untested |
| Attention | **eager** (stable) | PASSED — sdpa/flash crash the driver |
| Autocast | OFF by default (CHONK_AUTOCAST=1 to try) | untested |
| Label Smoothing | 0.1 | PASSED |
| EMA Decay | 0.9999 | PASSED |
| Grad Clip | 1.0 | |
| Pacing | CHONK_PAUSE=0.02-0.05s per chunk | mitigates iGPU display-starve crashes |
| Pool total | 69.23 GB (model 53.79 + KV 8.6@131K + LoRA 0.64 + opt 2.55 + acts 2.15 + staging) | fits 121GB |

---

## Phase 2: Testing & Metrics Collection (IN PROGRESS)

### Test 1: ChonkPool + KV Cache Integration
**Date**: 2026-08-13
**Status**: �� PASSED
**Config**: SEQ_LEN=8192, BATCH_SIZE=1, max_cache_len=8192
**Results**:
- ChonkPool initializes on Radeon 8060S (Strix Halo)
- KV cache builds: 64 layers (full attention)
- Pool used: 0.54 GB
- Pool budget disabled (maxHeapFraction=0.0f)

### Test 2: Model Load + Move to Chonk Buffer
**Date**: 2026-08-13
**Status**: PASSED
**Config**: Qwen 27B, bfloat16, trust_remote_code=True
**Results**:
- Root cause of load failure fixed: "-2 is not a valid device" = non-exportable Vulkan allocation's deviceAddress used as HIP pointer
- All allocations (weights/opt-states/acts) now route through alloc_export (dma-buf → hipImportExternalMemory)
- **load_model_directly_to_chonk**: 53.79 GB / 851 params in ~22s (keys remapped: model.X → model.language_model.X, lm_head.weight passthrough; vision + mtp keys skipped; precomputed slot offsets fix double-counting)
- **build_model_from_chonk_buffer**: zero-copy model from pool (nn.Parameter views in named_parameters order); rotary inv_freq materialized on CUDA

### Test 3: Full training step (forward + backward on chunk)
**Date**: 2026-08-13
**Status**: PASSED
**Results**:
- 512-token fwd+bwd on Chonk weights: loss 13.56, grads on 851/851 params
- Chunked KV-cache fwd+bwd (2x1024, eager): both chunks backward OK (truncated BPTT)
- Chunked semantics: cached K/V + linear-attention states are constants across chunks (detached); current chunk's K/V differentiable via cat
- patch_linear_cache_for_chunked_training() required (in-place copy_ broke autograd)
- Perf: ~5s per 1024-token chunk fwd (eager), ~9s per 2048

### Test 4: Optimizer step with ChonkAdamW
**Date**: 2026-08-13
**Status**: PASSED
**Results**:
- 512/512 trainable LoRA params got grads (lora_B first-step grads non-zero; lora_A zero until B non-zero — expected)
- fp32 AdamW states in Chonk (2.55GB), keyed by param object
- Step ran in-place on Chonk tensors; 256/512 states non-zero after 1 step (zero-grad lora_A — correct)

### Test 5: Multi-chunk sequence processing (LoRA KV pipeline)
**Date**: 2026-08-13
**Status**: PASSED (eager)
**Results**:
- 2x1024 chunks fwd+bwd through Chonk KV cache; pool 61.17GB totalUsed, allocationCount 5; HIP mem only 3.28GB (weights/opt/KV all in pool)
- 2048-token forward crashed with default sdpa (login screen); eager is the stable path

### Test 6: Full sequence (128K) with chunked forward
**Status**: PARTIAL — setup validated at max_cache_len=131072 (pool 69.23GB, fits); full-scale step NOT yet run (long sustained compute crashes; see breaking points)
- **4096-chunk backward CRASHED the machine (kernel panic)** — chunk size is capped at 2048

### Test 7: EMA weight application
**Status**: PASSED
**Results**: EMA shadow = fp32 clone (0.64GB); apply_shadow + save_pretrained to chonk_final OK (adapter 1.27GB saved)

### Test 8: End-to-end smoke run (train_qwen_chonk.py)
**Date**: 2026-08-13
**Status**: PASSED
**Config**: CHONK_SMOKE=1 (SEQ_LEN=2048, CHUNK_SIZE=512, MAX_STEPS=2), eager, CHONK_PAUSE=0.05
**Results**:
- Setup: pool 69.23GB; LoRA 318.8M trainable (1.17%); optimizer states 2.55GB; dataset packing OK (598.8M tokens, 936K seqs → 292K blocks @2048)
- Step 0 loss 4.22 (label smoothing 0.1), EMA applied, final save OK
- Dataset format: memmap tokens.bin+index.bin (variable-length seqs, packed into fixed blocks; fallback kept for .npy)

### Test 9: Pool-backed pluggable allocator (torch/HIP draws from Chonk Buffer)
**Date**: 2026-08-14
**Status**: PASSED (commit b70ae90)
**Goal**: Eliminate interleaved HIP segments + dedicated Vulkan BOs fragmenting the driver GTT manager (root cause of "free but can't allocate 48MB" OOMs and vkAllocateMemory hangs)
**Design**:
- `_pool_test_module.cpp` exports C-ABI `chonk_allocator_alloc/free` (old-style 2-fn `CUDAPluggableAllocator` ABI: `void* alloc(ssize_t, int, void*)`)
- Every torch segment is carved from a pool block: `g_pool->allocate(exportable)` → dma-buf fd → `hipImportExternalMemory` (in C++, `-D__HIP_PLATFORM_AMD__` + `-lamdhip64`)
- Slab sub-allocator: 2GB+ blocks, first-fit carve (512-align), coalescing freelist, keep `warmBlocks=2` fully-free blocks, release the rest back to the pool for KV cache reuse
- `chonk.py: install_chonk_allocator()` — **order matters**: pool init + HIP context probe (ctypes hipMalloc) BEFORE `change_current_allocator`; lazy pool-init inside alloc() deadlocked torch's context init (re-entrant hipImport)
- `ChonkPool()` adopts the already-created pool via `pool_mod.info()` (init throws "already initialized")
**Bugs found & fixed**:
1. torch calls `alloc(0)` (hipMalloc semantics) — returned an un-carved pointer colliding with the next alloc → duplicate addresses → torch double-free → "Trying to free a pointer not allocated here" abort. Fixed: min carve 512B.
2. Exit segfault: static `py::dict g_lastInitInfo` destructor ran after interpreter teardown (pybind needs live interpreter). Fixed: heap-allocate, never free.
3. Teardown crash: `hipDestroyExternalMemory` on blocks released after `pool.shutdown()` (HIP context gone). Fixed: skip HIP destroy when `g_pool == nullptr`.
**Results (smoke 1024-seq/256-chunk/3 chunks)**:
- Step 0 loss 4.4375, EMA applied, final save OK, **clean exit 0** (previous runs segfaulted at teardown)
- Pool stats now include torch segments: totalUsed 79.96GB at step 0 (was 69.23GB pool-only + invisible HIP)
- torch.cuda.memory / HIP peak metrics no longer meaningful — everything is pool memory now

### Next Tests Planned
1. **Edge sweep with allocator** (chunk 512→4096 × seq, one process at a time): re-map the crash envelope — the allocator changes the memory layout completely
2. **Full-scale step**: MAX_CACHE_LEN=131072, SEQ_LEN=131072, CHUNK_SIZE=1024, 1 step — run with pacing
3. **Leak detection**: pool stats + block count over many steps (blocks should stabilize; warmBlocks cap 2)

---

## Test 10: Full-sequence memory wall & AOTriton SDPA attempt (2026-08-14)
**Context**: full 131072-seq run (128 chunks @1024, allocator on) dies during the FIRST forward with
`VK_ERROR_OUT_OF_DEVICE_MEMORY` (radv: "Failed to allocate a buffer size 2147483648 domains 4") on a 2GB block
(MLP/LoRA `F.linear`). Pool after setup+empty_cache: 71.37GB used, 6 allocations.

**Budget analysis (eager attention)**:
- Fixed ~70GB: weights 53.8 + KV@131K 8.6 + LoRA 0.64 + optimizer 2.55 + activations 2.15 + staging 2
- Eager autograd graph @ chunk 1024 ≈ +30GB (fp32 attn_weights 537MB × 24 layers + softmax outs + mask) → ~103-105GB total
- Eager graph @ chunk 512 ≈ +15GB → ~88GB total (inside envelope)
- Effective wall ≈ 105-110GB committed + display (radv reports heap_mb=41642 ≈ (gttsize 122880+2048)/3 but ignores it; 105GB allocates fine)

**TTM phantom memory**: `ttm.page_pool_size` = 15,887,313 pages ≈ **60.6 GiB** freed GPU pages held in the TTM
page pool (reclaimable, not lost). Explains the recurring "~63GB used / device memory nearly full" readings after
crashed runs. Reboot clears it.

**Attempt: AOTriton fused SDPA (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` + `CHONK_ATTN=sdpa`)**:
- Rationale: fused kernels drop the fp32 attn_weights autograd graph (~12GB+); guide for this exact stack
  (Qwen3.5-27B LoRA @ gfx1151, ROCm 7.13, PT 2.11) requires the env var for fused SDPA
- **Result: display driver reset → Linux login screen during edge 512/8192**, same failure mode as the old
  Test-1-era note. Confirmed again: fused AMD SDPA kernels crash when writing through dma-buf-imported Chonk memory.
- **Verdict: abandon sdpa/AOTriton on this stack.** Eager attention only. Memory headroom must come from
  CHUNK_SIZE instead: **full 131K run uses eager + CHUNK=512 (256 chunks, ~88GB budget)**.

---

## Final Recommendations
- **Use eager attention** — experimental AMD SDPA kernels crash the display driver (login screen) when writing through dma-buf-imported Chonk memory
- **CHUNK_SIZE = 1024 default** — 2048 froze the machine, 4096-chunk backward caused kernel panic (hard boot); 512/1024 validated stable
- **Pool-backed pluggable allocator is now the default** (CHONK_ALLOCATOR=1): one allocator family over the unified heap; torch segments come from and return to the Chonk pool
- **Run with pacing** (CHONK_PAUSE >= 0.02s/chunk) and keep sustained runs bounded; iGPU also drives the display
- **LoRA r=64 in Chonk** is the validated training strategy (full-param AdamW fp32 = 215GB does not fit at 131K)
- **Keep torch.compile + autocast OFF** until validated (env flags CHONK_COMPILE / CHONK_AUTOCAST)
- Full-scale 131K steps: expect ~30-40min/step (128 chunks @1024), run step-by-step with pauses