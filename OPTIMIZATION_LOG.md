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
| 2 (torch.compile) | TBD | |
| 3 (Flash Attn) | TBD | |
| 4 (Autocast) | TBD | |
| 7 (EMA) | TBD | |
| **Pool Budget** | **Fixed** | maxHeapFraction=0.0f disables budget check for Chonk Buffer training |

---

## Optimal Configuration (to be determined)
**Target**: Best quality/speed tradeoff

| Setting | Value |
|---------|-------|
| GRAD_ACCUM_STEPS | 4 |
| CHUNK_SIZE | 4096 |
| LR / Scheduler | 2e-5 / Cosine + 100 warmup |
| Optimizer | ChonkAdamW (fp32 states in Chonk Buffer) |
| Compile mode | reduce-overhead |
| Flash Attention | Enabled (flash_attention_2) |
| Autocast | BF16 |
| Label Smoothing | 0.1 |
| EMA Decay | 0.9999 |
| Grad Clip | 1.0 |

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
**Status**: ��� IN PROGRESS (timeout on full model)
**Config**: Qwen 27B, bfloat16, trust_remote_code=True
**Partial Results**:
- Config loads: qwen3_5
- Model loads: 53.79 GB CUDA memory
- load_model_into_chonk() started but timed out (120s)
- Need: longer timeout or staged move

### Next Tests Planned
1. **Test 2b**: Staged model move (layer-by-layer or shard-by-shard)
2. **Test 3**: Full training step (forward + backward on 4K chunk)
3. **Test 4**: Optimizer step with ChonkAdamW
4. **Test 5**: Multi-chunk sequence processing
5. **Test 6**: Full sequence (128K) with chunked forward
6. **Test 7**: EMA weight application
7. **Test 8**: Memory usage over time (leak detection)

---

## Final Recommendations
*To be filled after all experiments complete*